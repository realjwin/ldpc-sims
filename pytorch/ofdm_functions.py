import torch
import numpy as np

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
    
    return received_symbols.T.reshape((1,-1))

def quantizer(inputs, num_bits, clip_value):
    num_levels = np.power(2, num_bits) - 1
    step_size = 2*clip_value / num_levels
    
    idx_real = np.round((inputs.real + clip_value) / step_size)
    idx_imag = np.round((inputs.imag + clip_value) / step_size)
    
    quantized_real = np.clip(idx_real*step_size - clip_value, -clip_value, clip_value)
    quantized_imag = np.clip(idx_imag*step_size - clip_value, -clip_value, clip_value)

    quantized_temp = np.zeros(inputs.shape, dtype=np.complex)
    quantized_temp.real = quantized_real
    quantized_temp.imag = quantized_imag

    return quantized_temp

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

def decode_llr(llr, parity_matrix, bp_iterations):
    #setup BP network
    mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(parity_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations))
    else:
        model = BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations)
        
    #send model to GPU
    model.to(device)
    
#finish this LATER if we need it
#    llr_train = torch.tensor(rx_bits_llr_train, dtype=torch.double, requires_grad=True, device=device)
#    y_train = torch.tensor(tx_bits_train, dtype=torch.double, device=device)
#                    
#    x_train = torch.zeros(llr_train.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True, device=device)
#
#    #--- MODEL ---#
#    y_est_train = model_train(x_train, llr_train, clamp_value)


def weighted_mse(llr_est, llr, epsilon):
    return torch.mean((llr_est - llr)**2 / (torch.abs(llr) + epsilon))
    
def compute_ber(bits_est, bits):
    return np.sum(np.abs(bits_est - bits)) / bits.size
    
def compute_per(bits_est, bits, block_size):
    return 0

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