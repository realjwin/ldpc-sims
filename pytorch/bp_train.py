import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

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


def pyd(tensor):
    return tensor.detach().numpy()

def ber(y_est, y):
    y_est_np = np.round(pyd(y_est))
    y = pyd(y)
    
    return np.mean(np.abs(y - y_est_np).flatten())

def gen_codebook():
    messages = []
    
    for i in range(0,16):
        messages.append(np.array(list("{0:04b}".format(i)), dtype=np.int))

    messagebook = np.array(messages)
    
    codebook = np.transpose(np.mod(np.matmul(G, np.transpose(messagebook)), 2))
    
    return messagebook, codebook

if __name__ == "__main__":
    
    #--- VARIABLES ---#
    
    layers = 1
    clamp_value = 10000
    
    num_samples = 50000
    snr_min = 3
    snr_max = 3
    
    epochs = 20
    batch_size = 25000
    
    #np.random.seed(0)
    
    #--- NN SETUP ---#
    mask_cv, mask_vc, mask_cv_final, llr_expander = genMasks(H)
    model = BeliefPropagation(mask_cv, mask_vc, mask_cv_final, llr_expander, layers)

    #--- TRAINING DATA ---#
    #input: rx_bits_llr
    #output: tx_bits
    
    #generate book of hamming codes
    #each row is a message / codeword
    messagebook, codebook = gen_codebook()
    
    #generate indices corresponding to each message
    messages_train = np.zeros(num_samples, dtype=np.int)
    #messages_train = np.random.randint(0, 16, num_samples)
    messages_test = np.zeros(num_samples, dtype=np.int)
    #messages_test = np.random.randint(0, 16, num_samples)
    
    tx_bits_train = codebook[messages_train]
    tx_bits_test = codebook[messages_test]
    
    snr = np.random.uniform(snr_min, snr_max, (num_samples,1))
    sigma = 1 / np.sqrt(np.power(10, snr/10))
    noise_train = np.random.normal(0, np.broadcast_to(sigma, tx_bits_train.shape))
    noise_test = np.random.normal(0, np.broadcast_to(sigma, tx_bits_test.shape))
    
    rx_bits_train = 2 * tx_bits_train - 1 + noise_train
    rx_bits_train_llr = ( np.power(rx_bits_train + 1, 2) - np.power(rx_bits_train - 1, 2) ) / (2*np.power(sigma, 2) )

    rx_bits_test = 2 * tx_bits_test - 1 + noise_test
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
        
        for batch_idx in range(np.int(np.floor(num_samples/batch_size))):
            
            x = torch.zeros(batch_size, mask_cv.shape[0], dtype=torch.double, requires_grad=True)
            
            llr_input = llr_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
            
            y_est = model(x, llr_input, clamp_value)
            
            loss = criterion(y_est, y_train[batch_idx*batch_size:(batch_idx+1)*batch_size])
            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            
        #training accuracy
        x_train = torch.zeros(llr_train.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True)
        y_est_train = model(x_train, llr_train, clamp_value)
        
        train_ber = ber(y_est_train, y_train)
    
        #validation accuracy
        x_test = torch.zeros(llr_test.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True)
        y_est_test = model(x_test, llr_test, clamp_value)
        
        test_ber = ber(y_est_test, y_test)
                   
        print('[epoch %d] loss: %.3f, train_ber: %.3f, test_ber: %.3f' % (epoch_idx + 1, epoch_loss, train_ber, test_ber))
    
    #--- SAVE MODEL ---#