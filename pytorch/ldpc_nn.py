#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:36:04 2019

@author: jacobwinick
"""

import pickle
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim

def importData(filename):
    mat_contents = sio.loadmat(filename)
    cbits = mat_contents['cbits_final']
    rbits = mat_contents['rbits_final']
    snrdb = mat_contents['snrdb'].flatten()
    iterations = mat_contents['iterations'].flatten()
    ber = mat_contents['ber_final']
    num_blocks = mat_contents['num_blocks'].flatten()[0]
    return cbits, rbits, snrdb, iterations, ber, num_blocks
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(648, 1296)
        self.hidden = nn.Linear(1296, 648)
        self.output = nn.Linear(648, 324)
        
    def forward(self, x):
        x = torch.sigmoid(self.input(x))
        x = torch.sigmoid(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

def main():
    #data (input: 1 codeword, output: 1 codeword, retrain each SNR - for now)
    filename = '../data/ml/20190328-1443_relay_n=1000.mat'
    cbits, rbits, snrdb, iterations, ber, num_blocks = importData(filename)
    bp_iter = 2
    
    #training variables
    epochs = 1000
    N_train = int(num_blocks / 2)
    N_test = int(num_blocks / 2)
    iterations = 2
    batch_size = int(N_train / iterations)
    
    #outputs
    BER_train_nn = np.empty(snrdb.shape)
    BER_test_nn = np.empty(snrdb.shape)
    BER_train_bp = np.empty(snrdb.shape)
    BER_test_bp = np.empty(snrdb.shape)
    
    for idx, val in enumerate(snrdb):
        print('The value is: %.3f' % val)
        
        #setup NN (SWITCH TO ADAM OPTIMIZER!)
        net = Net()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(net.parameters(), lr=.05)
        
        bits_train = np.transpose(cbits[0:324, :int(num_blocks/2)])
        bits_test = np.transpose(cbits[0:324, int(num_blocks/2):])
        rbits_train = np.transpose(rbits[bp_iter, idx][:, :int(num_blocks/2)])
        rbits_test = np.transpose(rbits[bp_iter, idx][:, int(num_blocks/2):])
    
        #convert codewords / outputs to tensor and batch them
        train_input = torch.tensor(rbits_train, dtype=torch.float)
        train_output = torch.tensor(bits_train, dtype=torch.float)
        test_input = torch.tensor(rbits_test, dtype=torch.float)

        #train NN (epochs, batch, iterations)
        for epoch in range(epochs):
            epoch_loss = 0
            
            for iter in range(iterations):
                #what are the dimension of torch.tensor again?
                data_in = train_input[iter*batch_size:(iter+1)*batch_size, :]
                data_out = net(data_in)
                correct_out = train_output[iter*batch_size:(iter+1)*batch_size, :]
                
                loss = criterion(data_out, correct_out)
                loss.backward()
                optimizer.step()                
                epoch_loss += loss.item()
                
            print('[epoch %d] loss: %.3f' % (epoch + 1, epoch_loss / iterations))
        
        #NN validation
        with torch.no_grad():
            train_nn = net(train_input)
            test_nn = net(test_input)
        
        train_nn_raw = train_nn.numpy()
        test_nn_raw = test_nn.numpy()
        train_nn = np.round(train_nn_raw)
        test_nn = np.round(test_nn_raw)

        BER_train_nn[idx] = sum(sum(abs(train_nn - bits_train))) / (N_train*324)
        BER_test_nn[idx] = sum(sum(abs(test_nn - bits_test))) / (N_test*324)
        
        BER_train_bp[idx] = ber[bp_iter, idx]
        BER_test_bp[idx] = ber[bp_iter, idx]

    #saving data
    
    
    with open('objs.pkl', 'wb') as f:
        pickle.dump([BER_train_nn, BER_test_nn, BER_train_bp, BER_test_bp], f)
    
main()