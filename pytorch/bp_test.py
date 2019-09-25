import torch
import numpy as np
import matplotlib.pyplot as plt

from bp import *
from masking import genMasks


G = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 1, 0, 1],
              [0, 1, 1, 1],
              [1, 1, 1, 0]])
    
H = np.array([[1, 1, 0, 1, 1, 0, 0],
              [0, 1, 1, 1, 0, 1, 0],
              [1, 1, 1, 0, 0, 0, 1]])


def gen_codebook():
    messages = []
    
    for i in range(0,16):
        messages.append(np.array(list("{0:04b}".format(i)), dtype=np.int))

    messagebook = np.array(messages)
    
    codebook = np.transpose(np.mod(np.matmul(G, np.transpose(messagebook)), 2))
    
    return messagebook, codebook


def md_decoder(rx_codeword, codebook):
    
    distance = []
    
    #check distance from each codeword
    for codeword in codebook:
        distance.append(np.sum(np.abs(codeword - rx_codeword)))
    
    #choose the minimum distance codeword
    return np.argmin(distance)


if __name__ == "__main__":
    
    #--- NN SETUP ---#
    mask_cv, mask_vc, mask_cv_final, llr_expander = genMasks(H)
    model1 = BeliefPropagation(mask_cv, mask_vc, mask_cv_final, llr_expander, 1)
    model2 = BeliefPropagation(mask_cv, mask_vc, mask_cv_final, llr_expander, 2)
    model4 = BeliefPropagation(mask_cv, mask_vc, mask_cv_final, llr_expander, 4)
    
    x = torch.zeros(mask_cv.shape[0], 1, dtype=torch.double, requires_grad=True)
    
    #--- VARIABLES ---#
    
    clamp_value = 1000
    num_samples = 10000
    snr_list = [0, 2, 4, 6, 8, 10]
    
    per_md = np.zeros((len(snr_list),1))
    ber_md = np.zeros((len(snr_list),1))
    
    per_nn1 = np.zeros((len(snr_list),1))
    ber_nn1 = np.zeros((len(snr_list),1))
    per_nn2 = np.zeros((len(snr_list),1))
    ber_nn2 = np.zeros((len(snr_list),1))
    per_nn4 = np.zeros((len(snr_list),1))
    ber_nn4 = np.zeros((len(snr_list),1))
    
    #--- TX BITS ---#
    
    #generate book of hamming codes
    #each row is a message / codeword
    messagebook, codebook = gen_codebook()
    
    #generate indices corresponding to each message
    messages = np.random.randint(0, 16, num_samples)
    
    tx_bits = 2 * codebook[messages] - 1
    
    for snr_idx, snr in enumerate(snr_list):
    
        #--- CHANNEL ---#
        
        sigma = 1 / np.sqrt(np.power(10, snr/10))
        
        noise = np.random.normal(0, sigma, tx_bits.shape)
        
        rx_bits = tx_bits + noise
        
        #--- RX BITS ---#
        
        rx_bits_llr = ( np.power(rx_bits + 1, 2) - np.power(rx_bits - 1, 2) ) / (2*np.power(sigma, 2))
        rx_bits_hard = rx_bits > 0
        
        #--- DECODING ---#
        
        for idx in range(0, num_samples):
            
            message_md = md_decoder(rx_bits_hard[idx], codebook)
            
            per_md[snr_idx] += np.float(np.abs(message_md - messages[idx]) > 0)
            ber_md[snr_idx] += np.float(np.sum(np.abs(messagebook[message_md] - messagebook[messages[idx]])))        
        
            llr = torch.tensor(rx_bits_llr[idx], dtype=torch.double, requires_grad=True)
            
            cbits_nn1 = np.transpose(np.round(model1(x, llr, clamp_value).detach().numpy()))[0]
            per_nn1[snr_idx] += np.float(np.sum(np.abs(cbits_nn1 - codebook[messages[idx]])) > 0)
            ber_nn1[snr_idx] += np.float(np.sum(np.abs(cbits_nn1[0:4] - messagebook[messages[idx]])))
            
            cbits_nn2 = np.transpose(np.round(model2(x, llr, clamp_value).detach().numpy()))[0]
            per_nn2[snr_idx] += np.float(np.sum(np.abs(cbits_nn2 - codebook[messages[idx]])) > 0)
            ber_nn2[snr_idx] += np.float(np.sum(np.abs(cbits_nn2[0:4] - messagebook[messages[idx]])))
            
            cbits_nn4 = np.transpose(np.round(model4(x, llr, clamp_value).detach().numpy()))[0]
            per_nn4[snr_idx] += np.float(np.sum(np.abs(cbits_nn4 - codebook[messages[idx]])) > 0)
            ber_nn4[snr_idx] += np.float(np.sum(np.abs(cbits_nn4[0:4] - messagebook[messages[idx]])))
        
    per_md /= num_samples
    ber_md /= num_samples * 4
    per_nn1 /= num_samples
    ber_nn1 /= num_samples * 4
    per_nn2 /= num_samples
    ber_nn2 /= num_samples * 4
    per_nn4 /= num_samples
    ber_nn4 /= num_samples * 4
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 10)
    fig.suptitle('PER and BER')
    
    ax1.set_title('PER')
    ax1.semilogy(snr_list, per_md, '*-')
    ax1.semilogy(snr_list, per_nn1, '+-')
    ax1.semilogy(snr_list, per_nn2)
    ax1.semilogy(snr_list, per_nn4)
    
    ax2.set_title('BER')
    ax2.semilogy(snr_list, ber_md)
    ax2.semilogy(snr_list, ber_nn1)
    ax2.semilogy(snr_list, ber_nn2)
    ax2.semilogy(snr_list, ber_nn4)
    
print('hold up - compare')