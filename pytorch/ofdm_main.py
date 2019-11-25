import pickle
import numpy as np

from ofdm_nn import train_nn


#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 1 #2000
batch_size = np.power(2,12) #np.power(2, 15)
learning_rates = np.array([.01, .1])

filenames = []

#--- LOAD DATA ---#

datafile = '20191125-120932_samples.pkl'
datapath = 'data/' + datafile

with open(datapath, 'rb') as f:
    data = pickle.load(f)

snrdb = data['snrdb']
qbits = data['qbits']

tx_symbols = data['tx_symbols']

rx_signal = data['rx_signal']
rx_symbols = data['rx_symbols']
rx_llrs = data['rx_llrs']

qrx_signal = data['qrx_signal']
qrx_symbols = data['qrx_symbols']
qrx_llrs = data['qrx_llrs']

#--- TRAIN NN ---#

for qbit_idx, qbit_val in enumerate(qbits):
    for snrdb_idx, snrdb_val in enumerate(snrdb):
        for lr_idx, lr_val in enumerate(learning_rates):
            
            input_samples = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)
            input_samples = input_samples.reshape(-1, 2*ofdm_size)
            
            output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
            
            filename = train_nn(input_samples, output_samples, datafile, snrdb_val, lr_val, qbit_val, ofdm_size, num_epochs, batch_size)
            
            filenames.append(filename)
        
