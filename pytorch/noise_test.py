import pickle
import numpy as np
import datetime as datetime
import matplotlib.pyplot as plt


from parity import *
from ofdm_functions import *

#--- VARIABLES ---#

ts = datetime.datetime.now()

snrdb = np.array([-5, 0, 5])

qbits = np.array([1, 3, 5])
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
            rx_signal, rx_symbols, rx_llrs, tx_signal = gen_data(tx_symbols, snrdb_val, ofdm_size)
            qrx_signal, qrx_symbols, qrx_llrs = gen_qdata(rx_signal, snrdb_val, qbit_val, clip_val, ofdm_size)
        
            q_noise = qrx_signal - tx_signal
            
            axes[snrdb_idx, clip_idx].hist(q_noise.real.flatten(), bins=100)
            #axes[snrdb_idx, clip_idx].set_yscale('log', nonposy='clip')
            axes[snrdb_idx, clip_idx].set_title('SNR: {} dB, Clip Ratio: {} dB'.format(snrdb_val, clipdb[clip_idx]))
            axes[snrdb_idx, clip_idx].set_xlabel('Noise Level')
            axes[snrdb_idx, clip_idx].set_ylabel('Num Samples')
            
    plt.tight_layout()
    plt.savefig('qbits={}.eps'.format(qbit_val), format='eps')
