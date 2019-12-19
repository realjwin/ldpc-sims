import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from bp.parity import H, G
from ofdm.ofdm_functions import *
from nn.llr import LLRestimator_tanh

#--- VARIABLES ---#

num_samples = np.power(2,16)
ofdm_size = 32

bp_iterations = 5
batch_size = 2**10
num_batches = num_samples // batch_size
clamp_value = 10

qbits = 3
clip_ratio = np.power(10, 0/10)

results = '20191219-132138_tx=20191213-234355_quantized_withsnr'
#--- LOAD RESULTS ---#

results_path = 'outputs/results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    filenames = data['filenames']
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
    LLRest = nn.DataParallel(LLRestimator_tanh(ofdm_size))
else:
    LLRest = nn.DataParallel(LLRestimator_tanh(ofdm_size))

LLRest.eval()

#send model to GPU
LLRest.to(device)

#load weights / oss
filepath = 'outputs/model/' + filenames[0]
checkpoint = torch.load(filepath, map_location=device)

LLRest.load_state_dict(checkpoint['model_state_dict'])

train_loss = checkpoint['loss']

#--- COMPUTE PERFORMANCE ---#

uncoded_ber = np.zeros(snrdb.shape)
coded_ber = np.zeros(snrdb.shape)
coded_bler = np.zeros(snrdb.shape)

uncoded_ber_quantized = np.zeros(snrdb.shape)
coded_ber_quantized = np.zeros(snrdb.shape)
coded_bler_quantized = np.zeros(snrdb.shape)

uncoded_ber_nn = np.zeros(snrdb.shape)
coded_ber_nn = np.zeros(snrdb.shape)
coded_bler_nn = np.zeros(snrdb.shape)

wmse_quantized = np.zeros(snrdb.shape)
wmse_nn = np.zeros(snrdb.shape)

for snrdb_idx, snrdb_val in enumerate(snrdb):
    
    print(snrdb_val)
    
    #--- GENERATE DATA ---#
    snr_single = np.power(10, snrdb_val/10)
    snr = snr_single * np.ones((num_samples, 1))
    
    rx_signal, rx_symbols, rx_llrs, tx_signal = gen_data(tx_symbols, snrdb_val, ofdm_size)
    received_symbols = rx_signal.reshape((-1, ofdm_size)).T

    #keep clip constant, adjust input signal
    agc_clip = 10

    #compute sigma_rx per dimension (real/imag)
    #this is approximate incoming signal amplitude
    #each column is an OFDM symbol with a different SNR
    sigma_rx =  .5 * (1 + 1/snr_single)

    #compute clipping per signal (this will be very rough b/c )
    factor = agc_clip / sigma_rx * clip_ratio
    
    rx_signal_scaled = factor * received_symbols
    rx_signal_scaled = rx_signal_scaled.T.reshape((1,-1))
    
    qrx_signal = quantizer(rx_signal_scaled, qbits, agc_clip)
    
    qrx_signal_rescaled = qrx_signal.reshape((-1, ofdm_size)).T
    qrx_signal_rescaled = qrx_signal_rescaled / factor
    
    deofdm_qsymbols = np.matmul(DFT(ofdm_size), qrx_signal_rescaled)
    
    noise_power = .5 * (1 / snr_single)
    
    #this is log(Pr=1 / Pr=0) aka +inf = 1, -inf = -1
    qllr_bit0 = ( np.power(deofdm_qsymbols.real - 1/np.sqrt(2), 2) - 
                np.power(deofdm_qsymbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
    qllr_bit1 = ( np.power(deofdm_qsymbols.imag - 1/np.sqrt(2), 2) - 
                np.power(deofdm_qsymbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
      
    qllrs = np.concatenate((qllr_bit0.T.reshape((-1,1)), qllr_bit1.T.reshape((-1,1))), axis=1)

    qrx_llrs = qllrs.reshape((1,-1))

    input_samples = np.concatenate((qrx_signal_rescaled.real.T, qrx_signal_rescaled.imag.T), axis=1)
    input_samples = input_samples.reshape(-1, 2*ofdm_size)
    #appends the SNR to the end of this
    input_samples = np.concatenate((input_samples, snr), axis=1)
 
    output_samples = rx_llrs.reshape(-1, 2*ofdm_size)

    qrx_llrs = qrx_llrs.reshape(-1, 2*ofdm_size)
    rx_llrs = rx_llrs.reshape(-1, 2*ofdm_size)
    enc_bits = enc_bits.reshape(-1, 2*ofdm_size)

    #--- INFERENCE ---#
    
    llr_est = np.zeros(output_samples.shape)
    
    for batch in range(0, num_batches):
        start_idx = batch*batch_size
        end_idx =  (batch+1)*batch_size
        
        x_input = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, device=device)
        
        with torch.no_grad():
            llr_est_temp = LLRest(x_input)
            llr_est_temp = 0.5*torch.log((1+llr_est_temp)/(1-llr_est_temp))
            llr_est_temp = np.clip(llr_est_temp, -clamp_value, clamp_value)
            
        llr_est[start_idx:end_idx, :] = llr_est_temp.cpu().detach().numpy()
    
    #--- LLR WMSE PERFORMANCE ---#
    
    wrong_qidx = np.where(np.sign(qrx_llrs) != np.sign(rx_llrs))
    wrong_idx = np.where(np.sign(llr_est) != np.sign(rx_llrs))
    #llr_est[wrong_idx] = rx_llrs[wrong_idx]
    #z_wrong = np.asarray([llr_est[wrong_idx], rx_llrs[wrong_idx]]).T
    
    wmse_quantized[snrdb_idx] = np.mean((qrx_llrs[wrong_qidx] - rx_llrs[wrong_qidx])**2 / (np.abs(rx_llrs[wrong_qidx]) + 10e-4))
    
    wmse_nn[snrdb_idx] = np.mean((llr_est[wrong_idx] - rx_llrs[wrong_idx])**2 / (np.abs(rx_llrs[wrong_idx]) + 10e-4))
    
    #compute number flipped? maybe later...
    
    #--- DECODING PERFORMANCE ---#
    
    cbits = (np.sign(rx_llrs) + 1) // 2
    bits = decode_bits(rx_llrs, H, bp_iterations, batch_size, clamp_value)
    
    cbits_nn = (np.sign(llr_est) + 1) // 2
    bits_nn = decode_bits(llr_est, H, bp_iterations, batch_size, clamp_value)
    
    cbits_quantized = (np.sign(qrx_llrs) + 1) // 2
    bits_quantized = decode_bits(qrx_llrs, H, bp_iterations, batch_size, clamp_value)
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
    
    uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
    coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
    uncoded_ber_quantized[snrdb_idx] = np.mean(np.abs(cbits_quantized - enc_bits))
    coded_ber_quantized[snrdb_idx] = np.mean(np.abs(bits_quantized[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_quantized[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_quantized - enc_bits), axis=1)))

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
    fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
             
    axes[0].semilogy(snrdb, uncoded_ber, label='Uncoded Traditional')
    axes[0].semilogy(snrdb, coded_ber, label='Coded Traditional')
    axes[0].semilogy(snrdb, uncoded_ber_nn, '--+', label='Uncoded NN')
    axes[0].semilogy(snrdb, coded_ber_nn, '--+', label='Coded NN')
    axes[0].semilogy(snrdb, uncoded_ber_quantized, '--*', label='Uncoded Quantized')
    axes[0].semilogy(snrdb, coded_ber_quantized, '--*', label='Coded Quantized')
    axes[0].set_title('BER')
    axes[0].set_xlabel('SNR (dB)')
    axes[0].set_ylabel('BER')
    axes[0].legend()
    
    axes[1].semilogy(snrdb, coded_bler, label='Traditional')
    axes[1].semilogy(snrdb, coded_bler_nn, '--+', label='NN')
    axes[1].semilogy(snrdb, coded_bler_quantized, '--*', label='Quantized')
    axes[1].set_title('BLER')
    axes[1].set_xlabel('SNR (dB)')
    axes[1].set_ylabel('BLER')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('outputs/plots/' + results + '_ber.eps', format='eps', bbox_inches='tight')

    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
             
    axes[0].plot(snrdb, wmse_nn, '--+', label='NN WMSE')
    axes[0].plot(snrdb, wmse_quantized, '--*', label='Quantized WMSE')
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
    plt.savefig('outputs/plots/' + results + '_wmse.eps', format='eps', bbox_inches='tight')