import pickle
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from bp import *
from masking import genMasks

def pyd(tensor):
    return tensor.detach().numpy()

def ber(y_est, y):
    y_est_np = -1*np.round(pyd(y_est)) + 1
    y = pyd(y)
    
    return np.mean(np.abs(y - y_est_np).flatten())

def importData(filename):
    mat_contents = sio.loadmat(filename)
    cbits = mat_contents['cbits_final']
    rscbits = mat_contents['rscbits_final']
    snrdb = mat_contents['snrdb'].flatten()
    iterations = mat_contents['iterations'].flatten()
    ber = mat_contents['ber_final']
    num_blocks = mat_contents['num_blocks'].flatten()[0]
    return cbits, rscbits, snrdb, iterations, ber, num_blocks

if __name__ == "__main__":
    
    #--- LOAD MATLAB ---#
    
    #parity check matrix
    filename = '../parity.mat'
    mat_contents = sio.loadmat(filename)
    H = mat_contents['H']
    
    #data (input: 1 codeword, output: 1 codeword, retrain each SNR - for now)
    filename = '../data/ldpc/20191010-1439_n=10000.mat'
    cbits, rscbits, snrdb, iterations, ber_matlab, num_blocks = importData(filename)

    #--- VARIABLES ---#

    num_samples_train = 1000
    bp_idx = 0
    snr_idx = 1
    
    layers = iterations[bp_idx]
    sigma = 1 / np.sqrt(np.power(10, snrdb[snr_idx]/10))

    clamp_value = 10000   
    epochs = 2
    batch_size = 10

    #np.random.seed(0)
    
    #--- NN SETUP ---#
    mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(H)
    model = BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, layers)

    #--- TRAINING DATA ---#
    #input: rx_bits_llr
    #output: tx_bits
    
    tx_bits_train = np.transpose(cbits[:,0:num_samples_train])
    rx_bits_train_llr = np.transpose(rscbits[0][snr_idx][:, 0:num_samples_train])
    #rx_bits_train_llr = ( np.power(rx_bits_train + 1, 2) - np.power(rx_bits_train -x 1, 2) ) / (2*np.power(sigma, 2) )
    
    #--- TRAINING ---#
    
    llr_train = torch.tensor(rx_bits_train_llr, dtype=torch.double)
    y_train = torch.tensor(tx_bits_train, dtype=torch.double)
            
    #training accuracy
    x_train = torch.zeros(llr_train.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True)
    y_est_train = model(x_train, llr_train, clamp_value)
    
    
    
    train_ber = ber(y_est_train, y_train)
                   
    print('train_ber: %.3f' % (train_ber))
    print('matlab_ber: %.3f' % (ber[bp_idx][snr_idx]))