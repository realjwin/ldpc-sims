import numpy as np

def create_bits(num_bits):
    return np.random.randint(2, num_bits)
    
def encode_bits(bits, generator_matrix):
    
def modulate_bits(bits):

def transmit_symbols(symbols, ofdm_size, snr):
    ofdm_symbols = np.matmul(W.conj().T, symbols)
    
    noise = (np.random.normal(0, 1/np.sqrt(snr), (block_size, num_samples)) +
        1j*np.random.normal(0, 1/np.sqrt(snr), (block_size, num_samples))) / np.sqrt(2))
    
    received_symbols = ofdm_symbols + noise
    quantized_symbols = np.sign(received_symbols)

    received_symbols = W*received_symbols

def quantizer(input, num_bits, clip_value):
    num_levels = np.power(2, num_bits)
    step_size = 2*clip_value / num_levels
    
    idx = np.int((input + clip_value) / step_size)
    quantized_input = idx*step_size - clip_value

    return np.clip(quantized_input, -clip_value, clip_value)

def demodulate_signal(symbols, ofdm_size, snr_est):

def decode_llr(llr, parity_matrix):
    
def compute_ber(bits_est, bits):
    
def compute_per(bits_est, bits):

#--- VARIABLES ---#
num_samples = 100
num_bits = num_samples * rate * block_size
 
snrdb = 10
snr = 

#generate DFT matrix
W = np.zeros((block_size, block_size))

for x in range(0,block_size):
    for y in range(0,block_size): 
        W[x,y] = np.exp(-1j*2*np.pi*x*y / block_size) / np.sqrt(block_size)
        
#load coding matrices

#QPSK mapping
qpsk = {
    (0,0) : 
        }
    
def Mapping(bits):
    return np.array([mapping_table[tuple(b)] for b in bits])

#--- FUNCTIONS ---#

#initialize from another file
ofdm_init()

bits = create_bits(num_bits)

coded_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(coded_bits)

for snr_idx, snr_val in enumerate(snr):
    rx_signal = transmit_symbols(tx_symbols, ofdm_size, snr_val)
    
    rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr_val)
    
    bits_est = decode_llr(rx_llrs, H)
    
    for q_val in quantizer_settings:
        qrx_signal = quantizer(input, q_val.num_bits, q_val.clip_value)
        
        qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr_val)

        qbits_est = decode_llr(qrx_llrs, H)
        
        #save qrx_llrs, qrx_symbols
    #save ber values
    #save llr values
