import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from ofdm_variables import *
from ofdm_functions import *

class LLRestimator(nn.Module):
    def __init__(self, ofdm_size, snr_est):
        super(LLRestimator, self).__init__()
        
        self.ofdm_size = ofdm_size
        self.snr_est = snr_est
        
        self.initial_bias = nn.Parameter(torch.zeros(1, 2*self.ofdm_size, out=None, dtype=torch.float, requires_grad=True))
        
        self.fft_layer = nn.Linear(2*self.ofdm_size, 2*self.ofdm_size, bias=True)
        
        self.leaky_relu = nn.LeakyReLU()
        
        self.weighted_scalar = nn.Parameter(torch.zeros(1, 2*self.ofdm_size, out=None, dtype=torch.float, requires_grad=True))
        
        self.llr_layer = nn.Linear(2*self.ofdm_size, 2*self.ofdm_size, bias=True)
        
        self.final_layer = nn.Linear(2*self.ofdm_size, 2*self.ofdm_size)
        
        #initialized parameters
        self.init_parameters()
        
    def init_parameters(self):
        
        #fft layer
        self.fft_layer.weight.data = torch.tensor(DFTreal(self.ofdm_size), dtype=torch.float, requires_grad=True)
        
        #weighted scalar
        self.weighted_scalar.data = torch.tensor(2*self.snr_est, dtype=torch.float, requires_grad=True).expand_as(self.weighted_scalar.data)
        
        #1/
        #-2/(np.sqrt(2)*.05)
        #(-2 / np.sqrt(2))
        
        #llr layer
        self.llr_layer.weight.data = torch.tensor((-2/np.sqrt(2)) * np.eye(2*self.ofdm_size), dtype=torch.float, requires_grad=True)
        
        #final layer
        self.final_layer.weight.data = torch.tensor(np.eye(2*self.ofdm_size), dtype=torch.float, requires_grad=True)
        
    def forward(self, x):
        x = x + self.initial_bias
        x = self.fft_layer(x)
        x = self.leaky_relu(x)
        x = self.weighted_scalar * x
        x = self.llr_layer(x)
        x = self.final_layer(x)
        
        return x #torch.tanh(x)
    

#--- VARIABLES ---#

snrdb = 5
snr = np.power(10, snrdb / 10)

ofdm_size = 32

num_samples = np.power(2, 22)
num_bits = 2 * num_samples * ofdm_size

num_epochs = 500
batch_pct = .5
train_pct = .75

train_idx = np.int(num_samples*train_pct)
batch_size = np.int(train_idx*batch_pct)
num_batches = np.int(train_idx / batch_size)

num_qbits = 2
clip_pct = 1

#--- GENERATE DATA ---#

bits = create_bits(num_bits)

tx_symbols = modulate_bits(bits)

rx_signal = transmit_symbols(tx_symbols, ofdm_size, snr)

rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr)

agc_real = np.max(rx_signal.real)
agc_imag = np.max(rx_signal.imag)
agc_clip = np.max([agc_real, agc_imag])

qrx_signal = quantizer(rx_signal, num_qbits, clip_pct*agc_clip)
qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr)  

rx_bits = .5*np.sign(rx_llrs) + .5

ber = compute_ber(rx_bits, bits)

num_plot = 100
plt.subplots(1,2,figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(rx_signal.real[:,0:num_plot].flatten(), 'r')
plt.plot(qrx_signal.real[:,0:num_plot].flatten(), 'b')
plt.subplot(1,2,2)
plt.plot(rx_signal.imag[:,0:num_plot].flatten(), 'r')
plt.plot(qrx_signal.imag[:, 0:num_plot].flatten(), 'b')
plt.show()
#plot decision boundaries


#--- NN TRAINING ---#

LLRest = LLRestimator(ofdm_size, snr)

criterion = nn.MSELoss()
optimizer = optim.Adam(LLRest.parameters(), lr=.001, amsgrad=True)

#--- DATA ---#
signal_temp = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)

input_data = signal_temp.reshape(-1, 2*ofdm_size)
output_data = rx_llrs.reshape(-1, 2*ofdm_size) #np.tanh(rx_llrs.reshape(-1, 2*ofdm_size))

x_train = torch.tensor(input_data[0:train_idx], dtype=torch.float, requires_grad=True)
y_train = torch.tensor(output_data[0:train_idx], dtype=torch.float)

x_test = torch.tensor(input_data[train_idx:], dtype=torch.float)
y_test = torch.tensor(output_data[train_idx:], dtype=torch.float)

#--- TRAINING ---#

for epoch in range(0, num_epochs):
    train_loss = 0
    
    for batch in range(0, num_batches):
        start_idx = batch*batch_size
        end_idx = (batch+1)*batch_size
        
        x_batch = x_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        y_est_train = LLRest(x_batch)
        
        #if I use MSE then the loss should be inversely proportional to
        #the magnitude becuase I don't really care if the LLR is correct
        #and already very large, basically I need a custom loss function
        loss = criterion(y_est_train, y_batch)
        loss.backward()
        
        train_loss += loss.item()
        
    #--- TEST ---#
    
    with torch.no_grad():
        y_est_test = LLRest(x_test)
        test_loss = criterion(y_est_test, y_test)
    
    print('[epoch %d] train_loss: %.3f, test_loss: %.3f' % (epoch + 1, train_loss / num_batches, test_loss))
    
    #--- OPTIMIZER STEP ---#
    optimizer.step()
    optimizer.zero_grad()

#--- ANALYSIS ---#

#epsilon = .000001
#llr_est = np.arctanh(np.clip(LLRest(x_test).detach().numpy(), -1+epsilon, 1-epsilon))
#llr = np.arctanh(np.clip(y_test.detach().numpy(), -1+epsilon, 1-epsilon))
  
llr_est = LLRest(x_test).detach().numpy()
llr = y_test.detach().numpy()

#--- WEIGHTED MSE PER CARRIER ---#
llr_est_reshape = np.reshape(llr_est.T, (-1, 2*llr_est.shape[0]))
llr_reshape = np.reshape(llr.T, (-1, 2*llr.shape[0]))
llr_wmse = np.mean(np.power((llr_est_reshape - llr_reshape) / llr_reshape, 2), axis=1)

#plt.bar(np.arange(len(llr_wmse)), llr_wmse, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Usage')
#plt.title('Programming language usage')

#plt.show()

#--- BER PER CARRIER ---#
#BER + # of Bits Flipped from Unquantized

nnbit = .5*(np.sign(llr_est_reshape) + 1)

qbit = .5*(np.sign(qrx_llrs) + 1)
qbit = np.reshape(qbit, (-1, 2*ofdm_size))
qbit = qbit[train_idx:]
qbit = np.reshape(qbit.T, (-1, 2*qbit.shape[0]))

ubit = .5*(np.sign(llr_reshape) + 1)

bit = np.reshape(bits, (-1, 2*ofdm_size))
bit = bit[train_idx:]
bit = np.reshape(bit.T, (-1, 2*bit.shape[0]))

bit_nnvu = np.mean(abs(nnbit-ubit), axis=1)
bit_qvu = np.mean(abs(qbit-ubit), axis=1)

bit_flip_idx = np.nonzero(abs(nnbit-qbit))
num_flipped = np.sum(abs(nnbit-qbit), axis=1)
temp = abs(nnbit - ubit) * abs(nnbit-qbit)
nnbit_bad = np.sum(temp, axis=1)
nnbit_good = num_flipped - nnbit_bad

ber_nnbit = np.mean(abs(nnbit-bit), axis=1)
ber_qbit = np.mean(abs(qbit-bit), axis=1)
ber_ubit = np.mean(abs(ubit-bit), axis=1)

#NN V Q COMPARISON
fig, ax = plt.subplots(figsize=(12,5))
index = np.arange(ofdm_size)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, bit_nnvu, bar_width,
alpha=opacity,
color='g',
label='NN vs. U')

rects2 = plt.bar(index + bar_width, bit_qvu, bar_width,
alpha=opacity,
color='b',
label='Q vs. U')

plt.xlabel('Subcarrier')
plt.ylabel('Error Rate')
plt.title('NN vs. Q Comparison')
plt.legend()
plt.tight_layout()
plt.show()

#NN FLIPPED COMPARISON
fig, ax = plt.subplots(figsize=(12,5))
index = np.arange(ofdm_size)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, num_flipped, bar_width,
alpha=opacity,
color='b',
label='# NN Bits Flipped')

rects2 = plt.bar(index + bar_width, nnbit_good, bar_width,
alpha=opacity,
color='g',
label='# NN Improvement Bits')

plt.xlabel('Subcarrier')
plt.ylabel('# of Bits')
plt.title('NN Flipped Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# create plot
fig, ax = plt.subplots(figsize=(12,5))
index = np.arange(ofdm_size)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, ber_ubit, bar_width,
alpha=opacity,
color='k',
label='U')

rects2 = plt.bar(index + bar_width, ber_qbit, bar_width,
alpha=opacity,
color='b',
label='Q')

rects3 = plt.bar(index + 2*bar_width, ber_nnbit, bar_width,
alpha=opacity,
color='g',
label='NN')

plt.xlabel('Subcarrier')
plt.ylabel('BER')
plt.title('BER Comparison')
plt.legend()
plt.tight_layout()
plt.show()


#--- NN PARAMETERS ---#
#for name, param in LLRest.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)