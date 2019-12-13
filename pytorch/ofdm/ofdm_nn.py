import numpy as np
import datetime as datetime
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from nn.llr import LLRestimator
from ofdm.ofdm_functions import *
from bp.bp import BeliefPropagation
import nn.joint as j

def train_nn(input_samples, output_samples, data_timestamp, snrdb, learning_rate, qbits, clipdb, ofdm_size, num_epochs, batch_size, load_model=None):
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

    #--- LOAD MODEL ---#
    
    if load_model:
        model_path = 'model/' + load_model
        
        checkpoint = torch.load(model_path, map_location=device)
        
        LLRest.load_state_dict(checkpoint['model_state_dict'])

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
            loss = weighted_mse(y_est_train, y_batch, 10e-4)
            loss.backward()
            
            train_loss[epoch] += loss.item()
            
            #--- OPTIMIZER STEP ---#
            optimizer.step()
            optimizer.zero_grad()
            
            del x_batch
            del y_batch
            del y_est_train
            del loss
        
        #--- TEST ---#
        
        if np.mod(epoch, 100) == 0:
            with torch.no_grad():
                random_sample = np.random.choice(num_samples, np.power(2, 10))
                
                x_test = torch.tensor(input_samples[random_sample], dtype=torch.float, device=device)
                y_test = torch.tensor(output_samples[random_sample], dtype=torch.float, device=device)
                
                y_est_test = LLRest(x_test)
                test_loss = weighted_mse(y_est_test, y_test, 10e-4)
                
            y_est_bits = np.sign(y_est_test.cpu().detach().numpy())
            y_bits = np.sign(output_samples[random_sample])
            
            num_flipped = np.mean(np.abs(y_est_bits - y_bits))
            temp = output_samples[random_sample]
            flipped_values= np.abs(temp[np.where(np.abs(y_est_bits - y_bits) == 2)])
            
            print('flipped mean: {}, median: {}, max: {}'.format(np.mean(flipped_values), np.median(flipped_values), np.amax(flipped_values)))
    
            print('[epoch %d] train_loss: %.3f, test_loss: %.3f, flipped_ber: %.3f' % (epoch + 1, train_loss[epoch] / num_batches, test_loss, num_flipped))
            
            del x_test
            del y_test
            del y_est_test
            del test_loss
        

    #--- RETURN MODEL PARAMETERS ---#
    
    ts = datetime.datetime.now()
        
    filename = ts.strftime('%Y%m%d-%H%M%S') + '_qbits={}_clipdb={}_snr={}_lr={}.pth'.format(qbits, clipdb, snrdb, learning_rate)
    filepath = 'outputs/model/' + filename
    
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

def train_joint(input_samples, output_samples, test_input, test_output, H, bp_iterations, clamp_value, data_timestamp, snrdb, learning_rate, qbits, clipdb, ofdm_size, num_epochs, batch_size, load_model=None):
    #--- VARIABLES ---#
    
    snr = np.power(10, snrdb / 10)
    num_samples = input_samples.shape[0]
    num_batches = num_samples // batch_size
    minibatch_size = 2**9
    num_minibatches = batch_size // minibatch_size

    #--- INIT NN ---#
    
    mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(H)

    #for cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs.")

    #model = j.Joint(ofdm_size, snr, mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations)

    model = nn.DataParallel(j.Joint(ofdm_size, snr, mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations))
    
    #send model to GPU
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.SGD([
        {'params': model.module.LLRest.parameters(), 'lr': 5*learning_rate},
        {'params': model.module.BP.parameters()}]
        , lr=learning_rate)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)#amsgrad=True)

    #--- LOAD MODEL ---#
    
    if load_model:
        model_path = 'model/' + load_model
        
        checkpoint = torch.load(model_path, map_location=device)
        
        num_items = len(checkpoint['model_state_dict'])
        
        d = collections.OrderedDict()
        
        for idx in range(0, num_items):
            (old_key, value) = checkpoint['model_state_dict'].popitem(last=False)
            key_split = old_key.split('.')
            key_split.insert(1, 'LLRest')
            deref = '.'
            new_key = deref.join(key_split)
            d[new_key] = value
    
        model.load_state_dict(d, strict=False)

    #--- TRAINING ---#
    
    train_loss = np.zeros(num_epochs)

    for epoch in range(0, num_epochs):
        
        #shuffle data each epoch
        p = np.random.permutation(num_samples)
        input_samples = input_samples[p]
        output_samples = output_samples[p]
        
        for batch in range(0, num_batches):
            for minibatch in range(0, num_minibatches):
                #print('batch: {}'.format(batch))
                
                start_idx = batch*batch_size+minibatch*minibatch_size
                end_idx = batch*batch_size+(minibatch+1)*minibatch_size
                
                x_batch = torch.tensor(input_samples[start_idx:end_idx], dtype=torch.float, requires_grad=True, device=device)
                y_batch = torch.tensor(output_samples[start_idx:end_idx], dtype=torch.float, device=device)
                
                x_temp = torch.zeros(x_batch.shape[0], mask_cv.shape[0], dtype=torch.float, device=device)
                
                y_est_train = model(x_batch, x_temp, clamp_value)
    
                loss = criterion(y_est_train, y_batch) / num_minibatches
                loss.backward()
                
                train_loss[epoch] += loss.item()
            
                del x_batch
                del y_batch
                del x_temp
                del y_est_train
                del loss
                
            #--- OPTIMIZER STEP ---#
            optimizer.step()
            optimizer.zero_grad()
        
        #--- TEST ---#
        
        if np.mod(epoch, 1) == 0:
            with torch.no_grad():
                
                x_test = torch.tensor(test_input, dtype=torch.float, device=device)
                y_test = torch.tensor(test_output, dtype=torch.float, device=device)
                
                x_temp = torch.zeros(x_test.shape[0], mask_cv.shape[0], dtype=torch.float, device=device)
                
                y_est_test = model(x_test, x_temp, clamp_value)
                test_loss = criterion(y_est_test, y_test)
                
            y_est_bits = np.round(y_est_test.cpu().detach().numpy())
            y_bits = np.round(test_output)

            ber = np.mean(np.abs(y_est_bits - y_bits))
    
            print('[epoch %d] train_loss: %.3f, test_loss: %.3f, test_ber: %.3f' % (epoch + 1, train_loss[epoch] / num_batches, test_loss, ber))
            
            del x_test
            del y_test
            del x_temp
            del y_est_test
            del test_loss
        

    #--- RETURN MODEL PARAMETERS ---#
    
    ts = datetime.datetime.now()
        
    filename = ts.strftime('%Y%m%d-%H%M%S') + '_qbits={}_clipdb={}_snr={}_lr={}_joint.pth'.format(qbits, clipdb, snrdb, learning_rate)
    filepath = 'outputs/model/' + filename
    
    torch.save({
            'epoch': epoch,
            'data_timestamp': data_timestamp,
            'batch_size': batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, filepath)
    
    del model
    
    return filename
