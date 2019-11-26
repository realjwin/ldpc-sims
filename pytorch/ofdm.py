import copy
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ofdm_variables import *
from ofdm_functions import *
from parity import *

class LLRestimator(nn.Module):
    def __init__(self, ofdm_size, snr_est):
        super(LLRestimator, self).__init__()
        
        self.ofdm_size = ofdm_size
        self.snr_est = snr_est
        
        self.activation = nn.ReLU()
        
        self.fft_layer = nn.Linear(2*self.ofdm_size, 2*self.ofdm_size, bias=True)
        self.scalar = nn.Parameter(torch.ones(1, 2*self.ofdm_size, out=None, dtype=torch.float, requires_grad=True))
        
        self.hidden1 = nn.Linear(2*self.ofdm_size, 8*self.ofdm_size, bias=True)
        self.hidden2 = nn.Linear(8*self.ofdm_size, 2*self.ofdm_size, bias=True)        
        self.hidden3 = nn.Linear(2*self.ofdm_size, 16*self.ofdm_size, bias=True)
        self.hidden4 = nn.Linear(16*self.ofdm_size, 16*self.ofdm_size, bias=True)
        
        self.final = nn.Linear(16*self.ofdm_size, 2*self.ofdm_size, bias=True)
        
        #initialized parameters
        self.init_parameters()
        
    def init_parameters(self):
        
        #fft layer
        self.fft_layer.weight.data = torch.tensor(DFTreal(self.ofdm_size), dtype=torch.float, requires_grad=True)
        
        #weighted scalar
        self.scalar.data = torch.tensor(2*self.snr_est * (-2/np.sqrt(2)), dtype=torch.float, requires_grad=True).expand_as(self.scalar.data)
        
    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        
        x = self.fft_layer(x)
        x = self.scalar * x
        
        x = self.activation(self.hidden3(x))
        x = self.activation(self.hidden4(x))
        
        return self.final(x)
    

#--- VARIABLES ---#

snrdb = -3
snr = np.power(10, snrdb / 10)

ofdm_size = 32

train_samples = np.power(2, 18) #22
test_samples = np.power(2, 15)
num_epochs = 500
batch_size = np.power(2, 13) #16
num_batches = np.power(2, 5)

num_samples = train_samples + test_samples
num_bits = 2 * num_samples * ofdm_size
train_idx = train_samples

num_qbits = 2
clip_pct = 1
tanh_scale = .1

#--- GENERATE DATA ---#

bits = create_bits(num_bits//2)

bits = encode_bits(bits, G)

tx_symbols = modulate_bits(bits)

rx_signal = transmit_symbols(tx_symbols, ofdm_size, snr)

rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr)

agc_real = np.max(rx_signal.real)
agc_imag = np.max(rx_signal.imag)
agc_clip = np.max([agc_real, agc_imag])

qrx_signal = quantizer(rx_signal, num_qbits, clip_pct*agc_clip)
qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr)  

rx_bits = .5*np.sign(rx_llrs) + .5

#ber = compute_ber(rx_bits, bits)

#num_plot = 100
#plt.subplots(1,2,figsize=(12,5))
#plt.subplot(1,2,1)
#plt.plot(rx_signal.real[:,0:num_plot].flatten(), 'r')
#plt.plot(qrx_signal.real[:,0:num_plot].flatten(), 'b')
#plt.subplot(1,2,2)
#plt.plot(rx_signal.imag[:,0:num_plot].flatten(), 'r')
#plt.plot(qrx_signal.imag[:, 0:num_plot].flatten(), 'b')
#plt.show()
#plot decision boundaries

    #rx_llr_hist = np.tanh(rx_llrs)
    #
    #plt.hist(rx_llr_hist[0, 0:1000], bins=200, range=(-1,1))
    #plt.show()

#--- NN TRAINING ---#

#for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs.")
    LLRest = nn.DataParallel(LLRestimator(ofdm_size, snr))
else:
    LLRest = LLRestimator(ofdm_size, snr)
    
#send model to GPU
LLRest.to(device)

#criterion = nn.MSELoss() using weighted_mse
optimizer = optim.Adam(LLRest.parameters(), lr=.01, amsgrad=True)

#--- DATA ---#
signal_temp = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)

input_data = signal_temp.reshape(-1, 2*ofdm_size)
output_data = rx_llrs.reshape(-1, 2*ofdm_size)

x_train = torch.tensor(input_data[0:train_idx], dtype=torch.float, requires_grad=True)
y_train = torch.tensor(output_data[0:train_idx], dtype=torch.float)

#x_test = torch.tensor(input_data[train_idx:], dtype=torch.float, requires_grad=False, device=device)
#y_test = torch.tensor(output_data[train_idx:], dtype=torch.float, requires_grad=False, device=device)

#--- TRAINING ---#

for epoch in range(0, num_epochs):
    train_loss = 0
    
    for batch in range(0, num_batches):
        start_idx = batch*batch_size
        end_idx = (batch+1)*batch_size
        
        x_batch = torch.tensor(input_data[start_idx:end_idx], dtype=torch.float, requires_grad=True, device=device)
        y_batch = torch.tensor(output_data[start_idx:end_idx], dtype=torch.float, device=device)
        
        y_est_train = LLRest(x_batch)
        
        #if I use MSE then the loss should be inversely proportional to
        #the magnitude becuase I don't really care if the LLR is correct
        #and already very large, basically I need a custom loss function
        loss = weighted_mse(y_est_train, y_batch, 10e-6)
        loss.backward()
        
        train_loss += loss.item()
        
        #--- OPTIMIZER STEP ---#
        optimizer.step()
        optimizer.zero_grad()
        
        del x_batch
        del y_batch
        del y_batch_train
        del loss
        
    #--- TEST ---#
    
#    with torch.no_grad():
#        y_est_test = LLRest(x_test)
#        test_loss = weighted_mse(y_est_test, y_test, 10e-6)
    
    print('[epoch %d] train_loss: %.3f, test_loss: %.3f' % (epoch + 1, train_loss / num_batches, test_loss))
    


#--- ANALYSIS ---#

#epsilon = .000001
#llr_est = np.arctanh(np.clip(LLRest(x_test).cpu().detach().numpy(), -1+epsilon, 1-epsilon))
#llr = np.arctanh(np.clip(output_data[train_idx:], -1+epsilon, 1-epsilon))
  
#llr_est = LLRest(x_test).cpu().detach().numpy()
#llr = output_data[train_idx:]

#--- WEIGHTED MSE PER CARRIER ---#
#llr_est_reshape = np.reshape(llr_est.T, (-1, 2*llr_est.shape[0]))
#llr_reshape = np.reshape(llr.T, (-1, 2*llr.shape[0]))
#llr_wmse = np.mean(np.power((llr_est_reshape - llr_reshape) / llr_reshape, 2), axis=1)

#plt.bar(np.arange(len(llr_wmse)), llr_wmse, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Usage')
#plt.title('Programming language usage')

#plt.show()

#--- BER PER CARRIER ---#
#BER + # of Bits Flipped from Unquantized

#nnbit = .5*(np.sign(llr_est_reshape) + 1)
#
#qbit = .5*(np.sign(qrx_llrs) + 1)
#qbit = np.reshape(qbit, (-1, 2*ofdm_size))
#qbit = qbit[train_idx:]
#qbit = np.reshape(qbit.T, (-1, 2*qbit.shape[0]))
#
#ubit = .5*(np.sign(llr_reshape) + 1)
#
#bit = np.reshape(bits, (-1, 2*ofdm_size))
#bit = bit[train_idx:]
#bit = np.reshape(bit.T, (-1, 2*bit.shape[0]))
#
#bit_nnvu = np.mean(abs(nnbit-ubit), axis=1)
#bit_qvu = np.mean(abs(qbit-ubit), axis=1)
#
#bit_flip_idx = np.nonzero(abs(nnbit-qbit))
#num_flipped = np.sum(abs(nnbit-qbit), axis=1)
#temp = abs(nnbit - ubit) * abs(nnbit-qbit)
#nnbit_bad = np.sum(temp, axis=1)
#nnbit_good = num_flipped - nnbit_bad
#
#ber_nnbit = np.mean(abs(nnbit-bit), axis=1)
#ber_qbit = np.mean(abs(qbit-bit), axis=1)
#ber_ubit = np.mean(abs(ubit-bit), axis=1)
#
##NN V Q COMPARISON
#fig, ax = plt.subplots(figsize=(12,5))
#index = np.arange(ofdm_size)
#bar_width = 0.35
#opacity = 0.8
#
#rects1 = plt.bar(index, bit_nnvu, bar_width,
#alpha=opacity,
#color='g',
#label='NN vs. U')
#
#rects2 = plt.bar(index + bar_width, bit_qvu, bar_width,
#alpha=opacity,
#color='b',
#label='Q vs. U')
#
#plt.xlabel('Subcarrier')
#plt.ylabel('Error Rate')
#plt.title('NN vs. Q Comparison')
#plt.legend()
#plt.tight_layout()
#plt.show()
#
##NN FLIPPED COMPARISON
#fig, ax = plt.subplots(figsize=(12,5))
#index = np.arange(ofdm_size)
#bar_width = 0.35
#opacity = 0.8
#
#rects1 = plt.bar(index, num_flipped, bar_width,
#alpha=opacity,
#color='b',
#label='# NN Bits Flipped')
#
#rects2 = plt.bar(index + bar_width, nnbit_good, bar_width,
#alpha=opacity,
#color='g',
#label='# NN Improvement Bits')
#
#plt.xlabel('Subcarrier')
#plt.ylabel('# of Bits')
#plt.title('NN Flipped Comparison')
#plt.legend()
#plt.tight_layout()
#plt.show()
#
## create plot
#fig, ax = plt.subplots(figsize=(12,5))
#index = np.arange(ofdm_size)
#bar_width = 0.25
#opacity = 0.8
#
#rects1 = plt.bar(index, ber_ubit, bar_width,
#alpha=opacity,
#color='k',
#label='U')
#
#rects2 = plt.bar(index + bar_width, ber_qbit, bar_width,
#alpha=opacity,
#color='b',
#label='Q')
#
#rects3 = plt.bar(index + 2*bar_width, ber_nnbit, bar_width,
#alpha=opacity,
#color='g',
#label='NN')
#
#plt.xlabel('Subcarrier')
#plt.ylabel('BER')
#plt.title('BER Comparison')
#plt.legend()
#plt.tight_layout()
#plt.show()


#--- NN PARAMETERS ---#
#for name, param in LLRest.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)