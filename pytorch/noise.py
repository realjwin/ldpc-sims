import pickle
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt


from parity import *
from ofdm_functions import *

def gen_data(tx_symbols, snrdb, qbits, clip_ratio, ofdm_size):
    snr = np.power(10, snrdb/10)
    
    rx_signal = transmit_symbols(tx_symbols, ofdm_size, snr)    
        
    rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr)
    
    sigma_rx = np.max(np.std(rx_signal))
    
    agc_clip = sigma_rx * clip_ratio
    
    qrx_signal = quantizer(rx_signal, qbits, agc_clip)
    qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr)  
    
    return rx_signal, rx_symbols, rx_llrs, qrx_signal, qrx_symbols, qrx_llrs

#--- VARIABLES ---#

ts = datetime.datetime.now()

snrdb = np.array([-5, 0, 5])

qbits = np.array([3, 6, 12])
clipdb = np.array([0, 5, 10])
clip_ratio = np.power(10, clipdb/10)

ofdm_size = 32

num_samples = np.power(2,15)

num_bits = 2 * num_samples * ofdm_size

#--- GENERATE BITS ---#

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

#--- GENERATE QUANTIZED NOISE ---#


for qbit_idx, qbit_val in enumerate(qbits):
    
    fig, axes = plt.subplots(3, 3, figsize=(12,10))
    fig.suptitle('Quantization: {}-Bits'.format(qbit_val), fontsize=16, y=1.03)
    
    for snrdb_idx, snrdb_val in enumerate(snrdb):
        for clip_idx, clip_val in enumerate(clip_ratio):
            rx_signal, rx_symbols, rx_llrs, qrx_signal, qrx_symbols, qrx_llrs = gen_data(tx_symbols, snrdb_val, qbit_val, clip_val, ofdm_size)
        
            q_noise = qrx_signal - tx_symbols
            
            axes[snrdb_idx, clip_idx].hist(q_noise.real.flatten(), bins=100)
            #axes[snrdb_idx, clip_idx].set_yscale('log', nonposy='clip')
            axes[snrdb_idx, clip_idx].set_title('SNR: {} dB, Clip Ratio: {} dB'.format(snrdb_val, clipdb[clip_idx]))
            axes[snrdb_idx, clip_idx].set_xlabel('Noise Level')
            axes[snrdb_idx, clip_idx].set_ylabel('Num Samples')
            
    plt.tight_layout()
    plt.savefig('qbits={}.eps'.format(qbit_val), format='eps')
