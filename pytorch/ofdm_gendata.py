import pickle
import numpy as np
import datetime as datetime

from parity import *
from ofdm_variables import *
from ofdm_functions import *

def gen_data(tx_symbols, snrdb, qbits, clip_pct, ofdm_size):
    snr = np.power(10, snrdb/10)
    
    rx_signal = transmit_symbols(tx_symbols, ofdm_size, snr)
        
    rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr)
    
    agc_real = np.max(rx_signal.real)
    agc_imag = np.max(rx_signal.imag)
    agc_clip = np.max([agc_real, agc_imag])
    
    qrx_signal = quantizer(rx_signal, qbits, clip_pct*agc_clip)
    qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr)  
    
    return rx_signal, rx_symbols, rx_llrs, qrx_signal, qrx_symbols, qrx_llrs

#--- VARIABLES ---#

ts = datetime.datetime.now()

snrdb = np.array([-5, -3, 0, 3, 5])

qbits = np.array([2, 4, 6])
clip_pct = 1

ofdm_size = 32

num_samples = np.power(2,22)

num_bits = 2 * num_samples * ofdm_size

#--- GENERATE BITS ---#

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

filename = 'data/' + ts.strftime('%Y%m%d-%H%M%S') + '_tx.pkl'

with open(filename, 'wb') as f:
    save_dict = {
            'snrdb': snrdb,
            'qbits': qbits,
            
            'enc_bits': enc_bits,
            'tx_symbols': tx_symbols
            }

    pickle.dump(save_dict, f)
    
print(filename)

#--- GENERATE DATA ---#

for snrdb_idx, snrdb_val in enumerate(snrdb):
    for qbit_idx, qbit_val in enumerate(qbits):
        rx_signal, rx_symbols, rx_llrs, qrx_signal, qrx_symbols, qrx_llrs = gen_data(tx_symbols, snrdb_val, qbit_val, clip_pct, ofdm_size)
    
        filename = 'data/' + ts.strftime('%Y%m%d-%H%M%S') + '_snr={}_qbits={}.pkl'.format(snrdb_val, qbit_val)
        
        with open(filename, 'wb') as f:
            save_dict = {
                    'snrdb': snrdb_val,
                    'qbits': qbit_val,
                    
                    'rx_signal': rx_signal,
                    'rx_llrs': rx_llrs,
                    'rx_symbols': rx_symbols,
                    
                    'qrx_signal': qrx_signal,
                    'qrx_llrs': qrx_llrs,
                    'qrx_symbols': qrx_symbols
                    }
            
            pickle.dump(save_dict, f)
    
        print(filename)