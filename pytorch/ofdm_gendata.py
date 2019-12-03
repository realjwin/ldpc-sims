import pickle
import numpy as np
import datetime as datetime

from parity import *
from ofdm_functions import *

#--- VARIABLES ---#

ts = datetime.datetime.now()

ofdm_size = 32

num_samples = np.power(2,16)

num_bits = 2 * num_samples * ofdm_size

#--- GENERATE BITS ---#

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

filename = 'data/' + ts.strftime('%Y%m%d-%H%M%S') + '_tx.pkl'

with open(filename, 'wb') as f:
    save_dict = {
            'enc_bits': enc_bits,
            'tx_symbols': tx_symbols
            }

    pickle.dump(save_dict, f)
    
print(filename)