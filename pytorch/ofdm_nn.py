import numpy as np
import datetime as datetime

import torch
import torch.optim as optim

from llr import LLRestimator
from ofdm_functions import *

def train_nn(input_samples, output_samples, data_timestamp, snrdb, learning_rate, qbits, ofdm_size, num_epochs, batch_size, load_model=None):
    #--- VARIABLES ---#
    
    snr = np.power(10, snrdb / 10)
    num_samples = input_samples.shape[0]
    num_batches = num_samples // batch_size

    #--- INIT NN ---#

    #for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs.")
        LLRest = nn.DataParallel(LLRestimator(ofdm_size, snr))
    else:
        LLRest = LLRestimator(ofdm_size, snr)
    
    #send model to GPU
    LLRest.to(device)

    optimizer = optim.SGD(LLRest.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(LLRest.parameters(), lr=learning_rate, amsgrad=True)

    #--- TRAINING ---#
    
    train_loss = np.zeros(num_epochs)

    for epoch in range(0, num_epochs):
        
        #shuffle data each epoch
        p = np.random.permutation(num_samples)
        input_samples = input_samples[p]
        output_samples = output_samples[p]
        
        for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx = (batch+1)*batch_size
            
            x_batch = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, requires_grad=True, device=device)
            y_batch = torch.tensor(output_samples[start_idx:end_idx], dtype=torch.float, device=device)
            
            y_est_train = LLRest(x_batch)
            
            #if I use MSE then the loss should be inversely proportional to
            #the magnitude becuase I don't really care if the LLR is correct
            #and already very large, basically I need a custom loss function
            loss = weighted_mse(y_est_train, y_batch, 10e-6)
            loss.backward()
            
            train_loss[epoch] += loss.item()
            
            #--- OPTIMIZER STEP ---#
            optimizer.step()
            optimizer.zero_grad()
            
            del x_batch
            del y_batch
            del y_est_train
            del loss
    
        if np.mod(epoch, 1) == 0:    
            print('[epoch %d] train_loss: %.5f, snr: %.2f, qbits: %d, lr: %.3f' % (epoch + 1, train_loss[epoch] / num_batches, snrdb, qbits, learning_rate))

    #--- RETURN MODEL PARAMETERS ---#
    
    ts = datetime.datetime.now()
        
    filename = ts.strftime('%Y%m%d-%H%M%S') + '_qbits={}_snr={}_lr={}.pth'.format(qbits, snrdb, learning_rate)
    filepath = 'model/' + filename
    
    torch.save({
            'epoch': epoch,
            'data_timestamp': data_timestamp,
            'batch_size': batch_size,
            'model_state_dict': LLRest.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, filepath)
    
    del LLRest
    
    return filename