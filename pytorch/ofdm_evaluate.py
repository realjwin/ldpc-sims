import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from parity import *
from decoder import decoder
from ofdm_functions import *
from llr import LLRestimator

#--- VARIABLES ---#

num_samples = np.power(2,20) #CHANGE THIS VALUE!
ofdm_size = 32

bp_iterations = 5
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 20

results = '20191203-191640_tx=20191203-162534'

#--- LOAD RESULTS ---#

results_path = 'results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    filenames = data['filenames']
    snrdb = data['snrdb']

#--- GENERATE BITS ---#

num_bits = 2 * num_samples * ofdm_size

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

#--- NN MODEL ---#

snr = 0 #placeholder because it currently does nothing

#for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs.")
    LLRest = nn.DataParallel(LLRestimator(ofdm_size, snr))
else:
    LLRest = nn.DataParallel(LLRestimator(ofdm_size, snr))

LLRest.eval()

#send model to GPU
LLRest.to(device)

#--- COMPUTE PERFORMANCE ---#

uncoded_ber = np.zeros(snrdb.shape)
coded_ber = np.zeros(snrdb.shape)
coded_bler = np.zeros(snrdb.shape)
uncoded_ber_nn = np.zeros(snrdb.shape)
coded_ber_nn = np.zeros(snrdb.shape)
coded_bler_nn = np.zeros(snrdb.shape)
wmse = np.zeros(snrdb.shape)

for snrdb_idx, snrdb_val in enumerate(snrdb):
    
    print(snrdb_val)
    
    #--- LOAD WEIGHTS ---#
    
    filepath = 'model/' + filenames[snrdb_idx]
    checkpoint = torch.load(filepath, map_location=device)
    
    LLRest.load_state_dict(checkpoint['model_state_dict'])
    
    #--- GENERATE DATA ---#
    
    rx_signal, rx_symbols, rx_llrs = gen_data(tx_symbols, snrdb_val, ofdm_size)

    input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
    input_samples = input_samples.reshape(-1, 2*ofdm_size)
    
    output_samples = rx_llrs.reshape(-1, 2*ofdm_size)

    #--- INFERENCE ---#
    
    llr_est = np.zeros(input_samples.shape)
    
    for batch in range(0, num_batches):
        start_idx = batch*batch_size
        end_idx =  (batch+1)*batch_size
        
        x_input = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, device=device)
        
        with torch.no_grad():
            llr_est[start_idx:end_idx, :] = LLRest(x_input).cpu().detach().numpy()
    
    #--- LLR WMSE PERFORMANCE ---#
    
    wmse[snrdb_idx] = np.mean((llr_est - output_samples)**2 / (np.abs(output_samples) + 10e-4))
    
    #compute number flipped? maybe later...
    
    #--- DECODING PERFORMANCE ---#
    
    enc_bits = enc_bits.reshape(-1, 2*ofdm_size)
    
    cbits = (np.sign(output_samples) + 1) // 2
    bits = decoder(output_samples, H, bp_iterations, batch_size, clamp_value)
    
    cbits_nn = (np.sign(llr_est) + 1) // 2
    bits_nn = decoder(llr_est, H, bp_iterations, batch_size, clamp_value)
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
    
    uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
    coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
#--- SAVE CODED INFORMATION ---#
    
ber_path = 'ber_curves/' + results + '.pkl'

with open(ber_path, 'wb') as f:
    save_dict = {
            'snrdb': snrdb,
            
            'uncoded_ber': uncoded_ber,
            'coded_ber': coded_ber,
            'coded_bler': coded_bler,
            
            'uncoded_ber_nn': uncoded_ber_nn,
            'coded_ber_nn': coded_ber_nn,
            'coded_bler_nn': coded_bler_nn,
            
            'wmse': wmse,
            }
    
    pickle.dump(save_dict, f)


plot = False 
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
             
    axes[0].semilogy(snrdb, uncoded_ber, '--*', label='Uncoded Traditional')
    axes[0].semilogy(snrdb, coded_ber, '--*', label='Coded Traditional')
    axes[0].semilogy(snrdb, uncoded_ber_nn, '--+', label='Uncoded NN')
    axes[0].semilogy(snrdb, coded_ber_nn, '--+', label='Coded NN')
    axes[0].set_title('BER')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].legend()
    
    axes[1].semilogy(snrdb, coded_bler, '--*', label='Traditional')
    axes[1].semilogy(snrdb, coded_bler_nn, '--+', label='NN')
    axes[1].set_title('BLER')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('BLER')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('unquantized_nn.eps', format='eps', bbox_inches='tight')
    #plt.show()
