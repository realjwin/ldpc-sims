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

bp_iterations = 3
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 20

#TODO: these should automatically be able to parse from results
qbits = np.array([1, 3, 5])
clipdb = np.array([0, 5])

results = '20191203-191640_tx=20191203-162534_quantized_half'

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

uncoded_ber_quantized = np.zeros((len(snrdb), len(qbits), len(clipdb)))
coded_ber_quantized = np.zeros((len(snrdb), len(qbits), len(clipdb)))
coded_bler_quantized = np.zeros((len(snrdb), len(qbits), len(clipdb)))

uncoded_ber_nn = np.zeros((len(snrdb), len(qbits), len(clipdb)))
coded_ber_nn = np.zeros((len(snrdb), len(qbits), len(clipdb)))
coded_bler_nn = np.zeros((len(snrdb), len(qbits), len(clipdb)))

wmse_quantized = np.zeros((len(snrdb), len(qbits), len(clipdb)))
wmse_nn = np.zeros((len(snrdb), len(qbits), len(clipdb)))

for filename in filenames:
    
    print(filename)
    
    #--- LOAD WEIGHTS ---#
    
    filepath = 'model/' + filename
    checkpoint = torch.load(filepath, map_location=device)
    
    LLRest.load_state_dict(checkpoint['model_state_dict'])
    
    snrdb_val = np.float(filename.split('_')[3].split('=')[1])
    qbits_val = np.float(filename.split('_')[1].split('=')[1])
    clipdb_val = np.float(filename.split('_')[2].split('=')[1])
    clip_ratio = np.power(10, clipdb_val / 10)
    
    snrdb_idx = np.argwhere(snrdb == snrdb_val)
    qbits_idx = np.argwhere(qbits == qbits_val)
    clipdb_idx = np.argwhere(clipdb == clipdb_val)
    
    print('SNR: {}, Q-Bits: {}, Clip: {}'.format(snrdb_idx, qbits_idx, clipdb_idx))
    
    
    #--- GENERATE DATA ---#
    
    rx_signal, rx_symbols, rx_llrs = gen_data(tx_symbols, snrdb_val, ofdm_size)
    
    qrx_signal, qrx_symbols, qrx_llrs = gen_qdata(rx_signal, snrdb_val, qbits_val, clip_ratio, ofdm_size)

    input_samples = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)
    input_samples = input_samples.reshape(-1, 2*ofdm_size)

    qrx_llrs = qrx_llrs.reshape(-1, 2*ofdm_size)
    rx_llrs = rx_llrs.reshape(-1, 2*ofdm_size)
    enc_bits = enc_bits.reshape(-1, 2*ofdm_size)

    #--- INFERENCE ---#
    
    llr_est = np.zeros(input_samples.shape)
    
    for batch in range(0, num_batches):
        start_idx = batch*batch_size
        end_idx =  (batch+1)*batch_size
        
        x_input = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, device=device)
        
        with torch.no_grad():
            llr_est[start_idx:end_idx, :] = LLRest(x_input).cpu().detach().numpy()
    
    #--- LLR WMSE PERFORMANCE ---#
    
    wmse_quantized[snrdb_idx, qbits_idx, clipdb_idx] = np.mean((qrx_llrs - rx_llrs)**2 / (np.abs(rx_llrs) + 10e-4))
    
    wmse_nn[snrdb_idx, qbits_idx, clipdb_idx] = np.mean((llr_est - rx_llrs)**2 / (np.abs(rx_llrs) + 10e-4))
    
    #compute number flipped? maybe later...
    
    #--- DECODING PERFORMANCE ---#
    
    cbits = (np.sign(rx_llrs) + 1) // 2
    bits = decoder(rx_llrs, H, bp_iterations, batch_size, clamp_value)
    
    cbits_nn = (np.sign(llr_est) + 1) // 2
    bits_nn = decoder(llr_est, H, bp_iterations, batch_size, clamp_value)
    
    cbits_quantized = (np.sign(qrx_llrs) + 1) // 2
    bits_quantized = decoder(qrx_llrs, H, bp_iterations, batch_size, clamp_value)
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
    
    uncoded_ber_nn[snrdb_idx, qbits_idx, clipdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
    coded_ber_nn[snrdb_idx, qbits_idx, clipdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_nn[snrdb_idx, qbits_idx, clipdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
    uncoded_ber_quantized[snrdb_idx, qbits_idx, clipdb_idx] = np.mean(np.abs(cbits_quantized - enc_bits))
    coded_ber_quantized[snrdb_idx, qbits_idx, clipdb_idx] = np.mean(np.abs(bits_quantized[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_quantized[snrdb_idx, qbits_idx, clipdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_quantized - enc_bits), axis=1)))

#--- SAVE CODED INFORMATION ---#
    
ber_path = 'ber_curves/' + results + '.pkl'

with open(ber_path, 'wb') as f:
    save_dict = {
            'snrdb': snrdb,
            'qbits': qbits,
            'clipdb': clipdb,
            
            'uncoded_ber': uncoded_ber,
            'coded_ber': coded_ber,
            'coded_bler': coded_bler,
            
            'uncoded_ber_nn': uncoded_ber_nn,
            'coded_ber_nn': coded_ber_nn,
            'coded_bler_nn': coded_bler_nn,
            
            'uncoded_ber_quantized': uncoded_ber_quantized,
            'coded_ber_quantized': coded_ber_quantized,
            'coded_bler_quantized': coded_bler_quantized,
            
            'wmse_nn': wmse_nn,
            'wmse_quantized': wmse_quantized
            }
    
    pickle.dump(save_dict, f)

plot = True 
if plot:
    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle('NN Performance on Quantized Inputs', fontsize=16, y=1.02)
    
    axes[0].semilogy(snrdb, uncoded_ber, label='Uncoded Traditional')
    axes[0].semilogy(snrdb, coded_ber, label='Coded Traditional')
    for qbits_idx, qbits_val in enumerate(qbits):
        for clipdb_idx, clipdb_val in enumerate(clipdb):
            #axes[0].semilogy(snrdb, uncoded_ber_nn[:, qbits_idx, clipdb_idx], '--+', label='Uncoded NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
            #axes[0].semilogy(snrdb, coded_ber_nn[:, qbits_idx, clipdb_idx], '--+', label='Coded NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
            axes[0].semilogy(snrdb, uncoded_ber_quantized[:, qbits_idx, clipdb_idx], '--*', label='Uncoded Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
            #axes[0].semilogy(snrdb, coded_ber_quantized[:, qbits_idx, clipdb_idx], '--*', label='Coded Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[0].set_title('BER')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].legend()
    
    axes[1].semilogy(snrdb, coded_bler, label='Traditional')
    for qbits_idx, qbits_val in enumerate(qbits):
        for clipdb_idx, clipdb_val in enumerate(clipdb):
            axes[1].semilogy(snrdb, coded_bler_nn[:, qbits_idx, clipdb_idx], '--+', label='NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
            axes[1].semilogy(snrdb, coded_bler_quantized[:, qbits_idx, clipdb_idx], '--*', label='Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[1].set_title('BLER')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('BLER')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('quantized_nn.eps', format='eps', bbox_inches='tight')
