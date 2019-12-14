import pickle
import numpy as np
import datetime as datetime

from ofdm.ofdm_nn import train_nn_withSNR
from ofdm.ofdm_functions import *

#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 3000
batch_size = np.power(2, 14)
lr = .05

snrdb_low = 5
snrdb_high = 15

filenames = []

#--- LOAD DATA ---#

timestamp = '20191213-234355'

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'outputs/tx/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    enc_bits = data['enc_bits']
    tx_symbols = data['tx_symbols']
    num_samples = data['num_samples']

#--- GENERATE DATA ---#

snrdb = np.random.uniform(snrdb_low, snrdb_high, (num_samples, 1))

symbols = tx_symbols.reshape((-1, ofdm_size)).T
ofdm_symbols = np.matmul(DFT(ofdm_size).conj().T, symbols)

snr_val = np.power(10, np.broadcast_to(snrdb.T, symbols.shape)/10)

noise = (np.random.normal(0, 1/np.sqrt(snr_val)) + 
         1j*np.random.normal(0, 1/np.sqrt(snr_val))) / np.sqrt(2)

received_symbols = ofdm_symbols + noise

deofdm_symbols = np.matmul(DFT(ofdm_size), received_symbols)

noise_power = .5 * (1 / snr_val)

#this is log(Pr=1 / Pr=0) aka +inf = 1, -inf = -1
llr_bit0 = ( np.power(deofdm_symbols.real - 1/np.sqrt(2), 2) - 
            np.power(deofdm_symbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
llr_bit1 = ( np.power(deofdm_symbols.imag - 1/np.sqrt(2), 2) - 
            np.power(deofdm_symbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
  
llrs = np.concatenate((llr_bit0.T.reshape((-1,1)), llr_bit1.T.reshape((-1,1))), axis=1)
 
rx_signal = received_symbols.T.reshape((1,-1))
rx_llrs = llrs.reshape((1,-1))

#--- TRAIN UNQUANTIZED ---#

qbit = 0
clipdb = 0
    
input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
input_samples = input_samples.reshape(-1, 2*ofdm_size)
input_samples = np.concatenate((input_samples, np.power(10, snrdb/10)), axis=1)

output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
    
#--- TRAIN NETWORK ---#

filename = train_nn_withSNR(input_samples, output_samples, timestamp, snrdb_low, snrdb_high, lr, qbit, clipdb, ofdm_size, num_epochs, batch_size)

#--- SAVE LIST OF FILENAMES ---#

ts = datetime.datetime.now()

modelfile = ts.strftime('%Y%m%d-%H%M%S') + '_tx=' + timestamp + '_unquantized_withsnr.pkl'
modelpath = 'outputs/results/' + modelfile

with open(modelpath, 'wb') as f:
    save_dict = {
            'filename': filename,
            'snrdb_low': snrdb_low,
            'snrdb_high': snrdb_high}
    
    pickle.dump(save_dict, f)
    
print(modelpath)