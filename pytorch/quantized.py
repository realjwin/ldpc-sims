import pickle
import numpy as np
import datetime as datetime

from ofdm.ofdm_nn import train_nn
from ofdm.ofdm_functions import gen_data, gen_qdata


#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 1
batch_size = np.power(2, 14) #CHANGE THIS
lr = .1

qbits = np.array([3])
clipdb = np.array([10])

filenames = []

#--- LOAD PRETRAINED NETWORK ---#
    
results = '20191203-191640_tx=20191203-162534'

results_path = 'outputs/results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    pretrained_filenames = data['filenames']
    snrdb = data['snrdb']

#--- LOAD DATA ---#

timestamp = results.split('_')[1].split('=')[1]

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'outputs/tx/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    enc_bits = data['enc_bits']
    tx_symbols = data['tx_symbols']    

#--- TRAIN QUANTIZED ---#

for snrdb_idx, snrdb_val in enumerate(snrdb):
    
    if np.mod(snrdb_val, 2) == 0:    
        rx_signal, rx_symbols, rx_llrs, tx_signal = gen_data(tx_symbols, snrdb_val, ofdm_size)
        
        pretrained = pretrained_filenames[snrdb_idx]
        
        for qbits_idx, qbits_val in enumerate(qbits):
            for clipdb_idx, clipdb_val in enumerate(clipdb):
                
                print('Training SNR: {}, Q-Bits: {}, Clip: {} dB'.format(snrdb_val, qbits_val, clipdb_val))
        
                clip_ratio = np.power(10, (clipdb_val/10))
                
                #--- GENERATE QUNATIZED DATA ---#
            
                qrx_signal, qrx_symbols, qrx_llrs = gen_qdata(rx_signal, snrdb_val, qbits_val, clip_ratio, ofdm_size)
                    
                input_samples = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)
                input_samples = input_samples.reshape(-1, 2*ofdm_size)
                
                output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
                
                #--- TRAIN NETWORK ---#
                
                filename = train_nn(input_samples, output_samples, timestamp, snrdb_val, lr, qbits_val, clipdb_val, ofdm_size, num_epochs, batch_size, load_model=pretrained)
                
                filenames.append(filename)

#--- SAVE LIST OF FILENAMES ---#

snrdb = np.linspace(0, 10, 6)

modelfile = results + '_quantized.pkl'
modelpath = 'outputs/results/' + modelfile

with open(modelpath, 'wb') as f:
    save_dict = {
            'filenames': filenames,
            'snrdb': snrdb}
    
    pickle.dump(save_dict, f)
    
print(modelpath)