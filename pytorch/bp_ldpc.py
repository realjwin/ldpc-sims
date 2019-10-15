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
    y_est_np = np.round(pyd(y_est))
    y = pyd(y)
    
    return np.mean(np.abs(y - y_est_np).flatten())

def importData(filename):
    mat_contents = sio.loadmat(filename)
    cbits = mat_contents['cbits_final']
    rbits = mat_contents['rbits_final']
    snrdb = mat_contents['snrdb'].flatten()
    iterations = mat_contents['iterations'].flatten()
    ber = mat_contents['ber_final']
    num_blocks = mat_contents['num_blocks'].flatten()[0]
    return cbits, rbits, snrdb, iterations, ber, num_blocks

if __name__ == "__main__":
    
    #--- LOAD MATLAB ---#
    
    #parity check matrix
    filename = '../parity.mat'
    mat_contents = sio.loadmat(filename)
    H = mat_contents['H']
    
    #data (input: 1 codeword, output: 1 codeword, retrain each SNR - for now)
    filename = '../data/ml/20190328-1443_relay_n=1000.mat'
    cbits, rbits, snrdb, iterations, ber_matlab, num_blocks = importData(filename)

    #--- VARIABLES ---#

    num_samples_train = 1
    num_samples_test = 999
    bp_idx = 2
    snr_idx = 6
    
    layers = iterations[bp_idx]
    sigma = 1 / np.sqrt(np.power(10, snrdb[snr_idx]/10))

    clamp_value = 10000    
    epochs = 2
    batch_size = 10



    #np.random.seed(0)
    
    #--- NN SETUP ---#
    mask_cv, mask_vc, mask_cv_final, llr_expander = genMasks(H)
    model = BeliefPropagation(mask_cv, mask_vc, mask_cv_final, llr_expander, layers)

    #--- TRAINING DATA ---#
    #input: rx_bits_llr
    #output: tx_bits
    
    tx_bits_train = np.transpose(cbits[:,0:num_samples_train])
    rx_bits_train = np.transpose(rbits[bp_idx][snr_idx][:, 0:num_samples_train])
    rx_bits_train_llr = ( np.power(rx_bits_train + 1, 2) - np.power(rx_bits_train - 1, 2) ) / (2*np.power(sigma, 2) )
    
    tx_bits_test = np.transpose(cbits[:,num_samples_train:])
    rx_bits_test = np.transpose(rbits[bp_idx][snr_idx][:, num_samples_train:])
    rx_bits_test_llr = ( np.power(rx_bits_test + 1, 2) - np.power(rx_bits_test - 1, 2) ) / (2*np.power(sigma, 2) )
    
    #--- TRAINING ---#
    
    llr_train = torch.tensor(rx_bits_train_llr, dtype=torch.double)
    y_train = torch.tensor(tx_bits_train, dtype=torch.double)
    
    llr_test = torch.tensor(rx_bits_test_llr, dtype=torch.double)
    y_test = torch.tensor(tx_bits_test, dtype=torch.double)
    
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=.05)
    
    for epoch_idx in range(epochs):
        epoch_loss = 0
#        
#        for batch_idx in range(np.int(np.floor(num_samples_train/batch_size))):
#            
#            x = torch.zeros(batch_size, mask_cv.shape[0], dtype=torch.double, requires_grad=True)
#            
#            llr_input = llr_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
#            
#            y_est = model(x, llr_input, clamp_value)
#            
#            loss = criterion(y_est, y_train[batch_idx*batch_size:(batch_idx+1)*batch_size])
#            loss.backward()
#            epoch_loss += loss.item()
#
#            optimizer.step()
#            
#            print('one batch step')
            
        #training accuracy
        x_train = torch.zeros(llr_train.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True)
        y_est_train = model(x_train, llr_train, clamp_value)
        
        train_ber = ber(y_est_train, y_train)
    
#        #validation accuracy
#        x_test = torch.zeros(llr_test.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True)
#        y_est_test = model(x_test, llr_test, clamp_value)
#        
#        test_ber = ber(y_est_test, y_test)
        test_ber = 0
                   
        print('[epoch %d] loss: %.3f, train_ber: %.3f, test_ber: %.3f' % (epoch_idx + 1, epoch_loss, train_ber, test_ber))