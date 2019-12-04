import pickle
import numpy as np
import datetime as datetime

from ofdm_nn import train_nn
from ofdm_functions import gen_data


#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 10
batch_size = np.power(2, 12) #CHANGE THIS
lr = .01

snrdb = np.linspace(0, 10, 11)

filenames = []

#--- LOAD DATA ---#

timestamp = '20191203-135513'

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'data/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    enc_bits = data['enc_bits']
    tx_symbols = data['tx_symbols']

#--- TRAIN UNQUANTIZED ---#

for snrdb_idx, snrdb_val in enumerate(snrdb):
    qbit = 0
    clipdb = 0
    
    #--- GENERATE DATA ---#
    
    rx_signal, rx_symbols, rx_llrs = gen_data(tx_symbols, snrdb_val, ofdm_size)
        
    input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
    input_samples = input_samples.reshape(-1, 2*ofdm_size)
    
    output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
    
    #--- TRAIN NETWORK ---#
    
    filename = train_nn(input_samples, output_samples, timestamp, snrdb_val, lr, qbit, clipdb, ofdm_size, num_epochs, batch_size)
    
    filenames.append(filename)

#--- SAVE LIST OF FILENAMES ---#

ts = datetime.datetime.now()

modelfile = ts.strftime('%Y%m%d-%H%M%S') + '_tx=' + timestamp + '.pkl'
modelpath = 'results/' + modelfile

with open(modelpath, 'wb') as f:
    save_dict = {
            'filenames': filenames,
            'snrdb': snrdb}
    
    pickle.dump(save_dict, f)
    
print(modelpath)