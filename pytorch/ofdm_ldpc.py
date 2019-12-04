import pickle
import numpy as np
import datetime as datetime

from parity import *
from ofdm_nn import *
from ofdm_functions import gen_data, gen_qdata


#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 500
batch_size = np.power(2, 8) #CHANGE THIS
lr = .05

bp_iterations = 3
clamp_value = 20

qbits = np.array([3])
clipdb = np.array([5])

filenames = []

#--- LOAD PRETRAINED NETWORK ---#
    
results = '20191203-182425_tx=20191203-135513' #'20191203-191640_tx=20191203-162534_quantized'

results_path = 'results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    pretrained_filenames = data['filenames']
    snrdb = data['snrdb']

#--- LOAD DATA ---#

timestamp = results.split('_')[1].split('=')[1]

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'data/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    enc_bits = data['enc_bits']
    tx_symbols = data['tx_symbols']    

#--- TRAIN QUANTIZED ---#

for snrdb_idx, snrdb_val in enumerate(snrdb):  
    rx_signal, rx_symbols, rx_llrs = gen_data(tx_symbols, snrdb_val, ofdm_size)
    
    pretrained = pretrained_filenames[snrdb_idx]
    
    for qbits_idx, qbits_val in enumerate(qbits):
        for clipdb_idx, clipdb_val in enumerate(clipdb):
            
            print('Training SNR: {}, Q-Bits: {}, Clip: {} dB'.format(snrdb_val, qbits_val, clipdb_val))
    
            clip_ratio = np.power(10, (clipdb_val/10))
            
            #--- GENERATE QUNATIZED DATA ---#
        
            qrx_signal, qrx_symbols, qrx_llrs = gen_qdata(rx_signal, snrdb_val, qbits_val, clip_ratio, ofdm_size)
                
            input_samples = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)
            input_samples = input_samples.reshape(-1, 2*ofdm_size)
            
            output_samples = enc_bits.reshape(-1, 2*ofdm_size)
            
            #--- TRAIN NETWORK ---#
            
            filename = train_joint(input_samples, output_samples, H, clamp_value, bp_iterations, timestamp, snrdb_val, lr, qbits_val, clipdb_val, ofdm_size, num_epochs, batch_size, load_model=pretrained)
            
            filenames.append(filename)

#--- SAVE LIST OF FILENAMES ---#

modelfile = results + '_joint.pkl'
modelpath = 'results/' + modelfile

with open(modelpath, 'wb') as f:
    save_dict = {
            'filenames': filenames,
            'snrdb': snrdb}
    
    pickle.dump(save_dict, f)
    
print(modelpath)