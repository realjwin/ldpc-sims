import pickle
import numpy as np
import datetime as datetime

from parity import *
from ofdm_variables import *
from ofdm_functions import *

#--- VARIABLES ---#

snrdb = np.array([-5, -3, 0, 3, 5])
snr = np.power(10, snrdb / 10)

qbits = np.array([2, 4, 6])
clip_pct = 1

ofdm_size = 32

num_samples = np.power(2,16) #np.power(2,22)

num_bits = 2 * num_samples * ofdm_size

#--- VARIABLES TO SAVE ---#
rx_signal_save = np.zeros((len(snr), len(qbits), num_bits//2)) + 1j*np.zeros((len(snr), len(qbits), num_bits//2))
rx_llrs_save = np.zeros((len(snr), len(qbits), num_bits))
rx_symbols_save = np.zeros((len(snr), len(qbits), num_bits//2)) + 1j* np.zeros((len(snr), len(qbits), num_bits//2))

qrx_signal_save = np.zeros((len(snr), len(qbits), num_bits//2)) + 1j* np.zeros((len(snr), len(qbits), num_bits//2))
qrx_llrs_save = np.zeros((len(snr), len(qbits), num_bits))
qrx_symbols_save = np.zeros((len(snr), len(qbits), num_bits//2)) + 1j* np.zeros((len(snr), len(qbits), num_bits//2))


#--- GENERATE DATA ---#

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

for snr_idx, snr_val in enumerate(snr):
    for qbit_idx, qbit_val in enumerate(qbits):
        rx_signal = transmit_symbols(tx_symbols, ofdm_size, snr_val)
        
        rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr_val)
        
        agc_real = np.max(rx_signal.real)
        agc_imag = np.max(rx_signal.imag)
        agc_clip = np.max([agc_real, agc_imag])
        
        qrx_signal = quantizer(rx_signal, qbit_val, clip_pct*agc_clip)
        qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr_val)  
        
        rx_bits = .5*np.sign(rx_llrs) + .5
        
        rx_signal_save[snr_idx, qbit_idx, :] = rx_signal
        rx_llrs_save[snr_idx, qbit_idx, :] = rx_llrs
        rx_symbols_save[snr_idx, qbit_idx, :] = rx_symbols
        
        qrx_signal_save[snr_idx, qbit_idx, :] = qrx_signal
        qrx_llrs_save[snr_idx, qbit_idx, :] = qrx_llrs
        qrx_symbols_save[snr_idx, qbit_idx, :] = qrx_symbols        

#--- SAVE VARIABLES ---#
        
ts = datetime.datetime.now()
        
filename = 'data/' + ts.strftime('%Y%m%d-%H%M%S') + '_samples.pkl'
    
with open(filename, 'wb') as f:
    save_dict = {
            'snrdb': snrdb,
            'qbits': qbits,
            
            'enc_bits': enc_bits,
            'tx_symbols': tx_symbols,
            
            'rx_signal': rx_signal_save,
            'rx_llrs': rx_llrs_save,
            'rx_symbols': rx_symbols_save,
            
            'qrx_signal': qrx_signal_save,
            'qrx_llrs': qrx_llrs_save,
            'qrx_symbols': qrx_symbols_save
            }
    
    pickle.dump(save_dict, f)