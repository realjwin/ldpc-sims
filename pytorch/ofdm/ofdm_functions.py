import torch
import numpy as np

import torch.nn as nn

from bp.bp import *

def create_bits(num_bits):
    return np.random.randint(2, size=num_bits).reshape((1,-1))
    
def encode_bits(bits, generator_matrix):
    bits = bits.reshape((-1, generator_matrix.shape[1])).T
    cbits = np.mod(np.matmul(generator_matrix,bits),2)
    
    return cbits.T.reshape((1,-1))
    
def modulate_bits(bits):
    bits = -2 * bits.reshape((-1, 2)) + 1 #(0 -> 1, 1 -> -1)
    
    symbols = (1/np.sqrt(2))*bits[:,0] + (1j/np.sqrt(2))*bits[:,1]
    
    return symbols.reshape((1, -1))
    #return np.array([qpsk_mapping[tuple(b)] for b in bits]).reshape((1,-1))
    
def transmit_symbols(symbols, ofdm_size, snr):
    symbols = symbols.reshape((-1, ofdm_size)).T
    
    ofdm_symbols = np.matmul(DFT(ofdm_size).conj().T, symbols)
    
    noise = (np.random.normal(0, 1/np.sqrt(snr), ofdm_symbols.shape) +
        1j*np.random.normal(0, 1/np.sqrt(snr), ofdm_symbols.shape)) / np.sqrt(2)
    
    received_symbols = ofdm_symbols + noise
    
    return received_symbols.T.reshape((1,-1)), ofdm_symbols.T.reshape((1,-1))

def quantizer(inputs, num_bits, clip_value):
    num_levels = np.power(2, num_bits)
    step = 2*clip_value / (num_levels - 1)
    
    idx_real = np.floor(inputs.real/step + .5)
    idx_imag = np.floor(inputs.imag/step + .5)
    
    quantized_real = np.clip(step * idx_real, -(num_levels/2)*step+1, (num_levels/2)*step-1)
    quantized_imag = np.clip(step * idx_imag, -(num_levels/2)*step+1, (num_levels/2)*step-1)
    
    quantized_temp = np.zeros(inputs.shape, dtype=np.complex)
    quantized_temp.real = quantized_real
    quantized_temp.imag = quantized_imag

    return quantized_temp
    
#    idx_real = np.round((inputs.real + clip_value) / step_size)
#    idx_imag = np.round((inputs.imag + clip_value) / step_size)
#    
#    quantized_real = np.clip(idx_real*step_size - clip_value, -clip_value, clip_value)
#    quantized_imag = np.clip(idx_imag*step_size - clip_value, -clip_value, clip_value)
#
#    quantized_temp = np.zeros(inputs.shape, dtype=np.complex)
#    quantized_temp.real = quantized_real
#    quantized_temp.imag = quantized_imag

def demodulate_signal(symbols, ofdm_size, snr_est):
    symbols = symbols.reshape((-1, ofdm_size)).T
    
    #de-ofdm
    received_symbols = np.matmul(DFT(ofdm_size),symbols)
    
    #compute LLRs
    noise_power = .5 * (1 / snr_est) #assuming Pr = 1 / subcarrier, and power per channel (real/imag)
    
    #this is log(Pr=1 / Pr=0) aka +inf = 1, -inf = -1
    llr_bit0 = ( np.power(received_symbols.real - 1/np.sqrt(2), 2) - np.power(received_symbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
    llr_bit1 = ( np.power(received_symbols.imag - 1/np.sqrt(2), 2) - np.power(received_symbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
    
    llrs = np.concatenate((llr_bit0.T.reshape((-1,1)), llr_bit1.T.reshape((-1,1))), axis=1)
    
    return llrs.reshape((1,-1)), received_symbols.T.reshape((1,-1)) #this is not properly SNR scaled

def weighted_mse(llr_est, llr, epsilon):
    return torch.mean((llr_est - llr)**2 / (torch.abs(llr) + epsilon))
    
def compute_ber(bits_est, bits):
    return np.sum(np.abs(bits_est - bits)) / bits.size

def DFT(N):  
    W = np.zeros((N, N), dtype=np.complex)
    
    for x in range(0,N):
        for y in range(0,N): 
            W[x,y] = np.exp(-1j*2*np.pi*x*y / N) / np.sqrt(N)
            
    return W

def DFTreal(N):
    W = DFT(N)
    
    Wr = np.zeros((2*N, 2*N), dtype=np.float)
    
    for x in range(0,N):
        for y in range(0,N):
            Wr[2*x, 2*y] = W[x,y].real
            Wr[2*x, 2*y+1] = -W[x,y].imag
            Wr[2*x+1, 2*y] = W[x,y].imag
            Wr[2*x+1, 2*y+1] = W[x,y].real
            
    return Wr

def gen_data(tx_symbols, snrdb, ofdm_size):
    snr = np.power(10, snrdb/10)
    
    rx_signal, tx_signal = transmit_symbols(tx_symbols, ofdm_size, snr)    
        
    rx_llrs, rx_symbols = demodulate_signal(rx_signal, ofdm_size, snr)

    return rx_signal, rx_symbols, rx_llrs, tx_signal

def gen_qdata(rx_signal, snrdb, qbits, clip_ratio, ofdm_size):
    snr = np.power(10, snrdb/10)
    
    sigma_rx = np.max(np.std(rx_signal))
    
    agc_clip = sigma_rx * clip_ratio
    
    qrx_signal = quantizer(rx_signal, qbits, agc_clip)
    qrx_llrs, qrx_symbols = demodulate_signal(qrx_signal, ofdm_size, snr)  
    
    return qrx_signal, qrx_symbols, qrx_llrs

#batch_size must be divisible!
def decode_bits(llrs, H, bp_iterations, batch_size, clamp_value):
    
    output_bits = np.zeros(llrs.shape)
    
    num_batches = llrs.shape[0] // batch_size
    
    #--- NN SETUP ---#
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bp_model = nn.DataParallel(BeliefPropagation(H, bp_iterations))
    else:
        bp_model = BeliefPropagation(H, bp_iterations)
    
    bp_model.eval()
    
    #send model to GPU
    bp_model.to(device)
        
    for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx =  (batch+1)*batch_size
                    
            llr = torch.tensor(llrs[start_idx:end_idx, :], dtype=torch.float, device=device)                            
            x = torch.zeros(llr.shape[0], bp_model.layer_size() , dtype=torch.float, device=device)
        
            y_est = bp_model(x, llr, clamp_value)
        
            output_bits[start_idx:end_idx, :] = np.round(y_est.cpu().detach().numpy())
            
    return output_bits