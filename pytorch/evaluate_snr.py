import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bp.parity import H, G
from ofdm.ofdm_functions import *
from nn.llr import LLRestimator_withSNR

#--- VARIABLES ---#

num_samples = np.power(2,10)
ofdm_size = 32

bp_iterations = 3
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 10

results = '20191214-002518_tx=20191213-234355_unquantized_withsnr'

#--- LOAD RESULTS ---#

results_path = 'outputs/results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    filename = data['filename']
    snrdb_low = data['snrdb_low']
    snrdb_high = data['snrdb_high']
    
snrdb = np.linspace(snrdb_low, snrdb_high, 11)

#--- GENERATE BITS ---#

num_bits = 2 * num_samples * ofdm_size

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

#--- NN MODEL ---#

#for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs.")
    LLRest = nn.DataParallel(LLRestimator_withSNR(ofdm_size))
else:
    LLRest = nn.DataParallel(LLRestimator_withSNR(ofdm_size))

LLRest.eval()

#send model to GPU
LLRest.to(device)

#load weights / oss
filepath = 'outputs/model/' + filename
checkpoint = torch.load(filepath, map_location=device)

LLRest.load_state_dict(checkpoint['model_state_dict'])

train_loss = checkpoint['loss']

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
    
    #--- GENERATE DATA ---#
    snr = np.power(10, snrdb_val/10)
    snr = snr * np.ones((num_samples, 1))
    
    rx_signal, rx_symbols, rx_llrs, tx_signal = gen_data(tx_symbols, snrdb_val, ofdm_size)

    input_samples = np.concatenate((rx_signal.real.T, rx_signal.imag.T), axis=1)
    input_samples = input_samples.reshape(-1, 2*ofdm_size)
    #appends the SNR to the end of this
    input_samples = np.concatenate((input_samples, snr), axis=1)
 
    output_samples = rx_llrs.reshape(-1, 2*ofdm_size)

    #--- INFERENCE ---#
    
    llr_est = np.zeros(output_samples.shape)
    
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
    bits = decode_bits(output_samples, H, bp_iterations, batch_size, clamp_value)
    
    cbits_nn = (np.sign(llr_est) + 1) // 2
    bits_nn = decode_bits(llr_est, H, bp_iterations, batch_size, clamp_value)
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
    
    uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
    coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
#--- SAVE CODED INFORMATION ---#
    
ber_path = 'outputs/ber/' + results + '.pkl'

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


plot = True 
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
    #plt.savefig('unquantized_nn.eps', format='eps', bbox_inches='tight')

    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
             
    axes[0].plot(snrdb, wmse, '--+', label='NN WMSE')
    axes[0].set_title('WMSE')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('WMSE')
    axes[0].legend()
    
    axes[1].plot(train_loss, label='NN')
    axes[1].set_title('Train Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Train Loss')
    axes[1].legend()
    
    plt.tight_layout()
    #plt.savefig('unquantized_nn.eps', format='eps', bbox_inches='tight')