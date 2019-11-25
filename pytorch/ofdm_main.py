import pickle
import numpy as np

from ofdm_nn import train_nn


#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 1000
batch_size = np.power(2, 14)
learning_rates = np.array([.01, .1])

filenames = []

#--- LOAD DATA ---#

timestamp = '20191125-140044'

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'data/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    snrdb = data['snrdb']
    qbits = data['qbits']

#--- TRAIN NN ---#

#for qbit_idx, qbit_val in enumerate(qbits):
#    for snrdb_idx, snrdb_val in enumerate(snrdb):
#        
#        datapath = 'data/' + timestamp + '_snr={}_qbits={}.pkl'.format(snrdb_val, qbit_val)
#        
#        with open(datapath, 'rb') as f:
#            data = pickle.load(f)
#            
#            rx_llrs = data['rx_llrs']
#            qrx_signal = data['qrx_signal']
#        
#        for lr_idx, lr_val in enumerate(learning_rates):
#            
#            input_samples = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)
#            input_samples = input_samples.reshape(-1, 2*ofdm_size)
#            
#            output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
#            
#            filename = train_nn(input_samples, output_samples, timestamp, snrdb_val, lr_val, qbit_val, ofdm_size, num_epochs, batch_size)
#            
#            filenames.append(filename)
        
#--- TRAIN UNQUANTIZED ---#

for snrdb_idx, snrdb_val in enumerate(snrdb):
    qbit_val = 0
    
    datapath = 'data/' + timestamp + '_snr={}_qbits={}.pkl'.format(snrdb_val, qbits[0])
     
    with open(datapath, 'rb') as f:
        data = pickle.load(f)
        
        rx_llrs = data['rx_llrs']
        rx_signal = data['rx_signal']
    
    for lr_idx, lr_val in enumerate(learning_rates):
        
        input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
        input_samples = input_samples.reshape(-1, 2*ofdm_size)
        
        output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
        
        filename = train_nn(input_samples, output_samples, timestamp, snrdb_val, lr_val, qbit_val, ofdm_size, num_epochs, batch_size)
        
        filenames.append(filename)