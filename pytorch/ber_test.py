import pickle
import numpy as np
import matplotlib.pyplot as plt

from ofdm_nn import train_nn
from ofdm_functions import gen_data

from bp import *
from masking import genMasks

from parity import *

import torch

#--- VARIABLES ---#

ofdm_size = 32

snrdb = np.linspace(0, 10, 6)

bp_iterations = 10
clamp_value = 10

filenames = []

#--- LOAD DATA ---#

timestamp = '20191203-135513'

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'data/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    enc_bits = data['enc_bits']
    tx_symbols = data['tx_symbols']

#--- NN SETUP ---#
mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(H)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    bp_model = nn.DataParallel(BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations))
else:
    bp_model = BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations)

bp_model.eval()

#send model to GPU
bp_model.to(device)

#--- BER TEST ---#

uncoded_ber = np.zeros(snrdb.shape)
coded_ber = np.zeros(snrdb.shape)
coded_bler = np.zeros(snrdb.shape)

for snrdb_idx, snrdb_val in enumerate(snrdb):
    print('running SNR: {}'.format(snrdb_val))
    
    #--- GENERATE DATA ---#
    
    rx_signal, rx_symbols, rx_llrs = gen_data(tx_symbols, snrdb_val, ofdm_size)

    input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
    input_samples = input_samples.reshape(-1, 2*ofdm_size)
    
    output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
    
    batch_size = np.power(2, 14)
    num_batches = output_samples.shape[0] // batch_size
    
    #--- DECODE ---#
    
    output_bits = np.zeros(output_samples.shape)
    
    for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx =  (batch+1)*batch_size
                    
            llr = torch.tensor(-output_samples[start_idx:end_idx, :], dtype=torch.float, device=device)                            
            x = torch.zeros(llr.shape[0], mask_cv.shape[0], dtype=torch.float, device=device)
        
            y_est = bp_model(x, llr, clamp_value)
        
            output_bits[start_idx:end_idx, :] = np.round(y_est.cpu().detach().numpy())
        
    #--- COMPUTE BER & BLER ---#
    
    enc_bits = enc_bits.reshape(-1, 2*ofdm_size)
    
    rx_bits = (np.sign(output_samples) + 1 ) // 2
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(enc_bits - rx_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(enc_bits[:, 0:32] - output_bits[:, 0:32]))

    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(enc_bits - output_bits), axis=1)))