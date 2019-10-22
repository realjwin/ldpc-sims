import pickle
import numpy as np
import scipy.io as sio
import datetime as datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from bp import *
from masking import genMasks

def pyd(tensor):
    return tensor.detach().numpy()

def ber_sum(y_est, y):
    y_est_np = np.round(pyd(y_est))
    y = pyd(y)
    
    return np.sum(np.abs(y - y_est_np).flatten())

def importData(filename):
    mat_contents = sio.loadmat(filename)
    cbits = mat_contents['cbits_final']
    rscbits = mat_contents['rscbits_final']
    snrdb = mat_contents['snrdb'].flatten()
    iterations = mat_contents['iterations'].flatten()
    ber = mat_contents['ber_final']
    num_blocks = mat_contents['num_blocks'].flatten()[0]
    rbits_final = mat_contents['rbits_final']
    return cbits, rscbits, snrdb, iterations, ber, num_blocks, rbits_final

if __name__ == "__main__":
    
    #--- LOAD MATLAB ---#
    
    #parity check matrix
    filename = '../parity1.mat'
    mat_contents = sio.loadmat(filename)
    H = mat_contents['H']
    
    #data (input: 1 codeword, output: 1 codeword, retrain each SNR - for now)
    filename = '../data/ldpc/20191022-1400_n=10000.mat'
    cbits, rscbits, snrdb, iterations, ber_matlab, num_blocks, rbits_final = importData(filename)

    #--- VARIABLES ---#

    num_samples = 10000
    batch_size = 200
    
    clamp_value = 10000
    epsilon = 1/clamp_value #does nothing right now, but fix later
    
    ber_nn = np.zeros(ber_matlab.shape)
    
    #for cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #--- EVALUATION ---#
    
    for bp_idx, bp_value in enumerate(iterations):
        for snr_idx, snr_value in enumerate(snrdb):
            layers = bp_value
            sigma = 1 / np.sqrt(np.power(10, snr_value/10))
    
            #--- NN SETUP ---#
            mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(H)
            
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = nn.DataParallel(BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, layers))
            else:
                model = BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, layers)
            
            #send model to GPU
            model.to(device)
            
            #--- INFERENCE ---#
            
            for batch in range(0, np.int(num_samples/batch_size)):
                #--- DATA SETUP ---#
                
                tx_bits = np.transpose(cbits[:,batch*batch_size:(batch+1)*batch_size])
                rx_bits_llr = np.transpose(rscbits[0][snr_idx][:, batch*batch_size:(batch+1)*batch_size])
                
                llr = torch.tensor(rx_bits_llr, dtype=torch.double)
                y = torch.tensor(tx_bits, dtype=torch.double)
                    
                #training accuracy
                x = torch.zeros(llr.shape[0], mask_cv.shape[0], dtype=torch.double, requires_grad=True)
            
                #send tensors to device
                x.to(device)
                llr.to(device)
                
                with torch.no_grad():
                    y_est = model(x, llr, clamp_value)
                    
                    ber_nn[bp_idx][snr_idx] += ber_sum(y_est.cpu()[:, 0:32], y[:, 0:32])
                    
                    #ber_matlab[bp_idx][snr_idx] += ber_sum(np.transpose(rbits_final[bp_idx][snr_idx][:, batch*batch_size:(batch+1)*batch_size]), y)
                    
                del x
                del llr
                del y
                del y_est
                
            print('finished BER inference with Iterations: {} and SNR: {}'.format(bp_value, snr_value))
            
            del model
            
    ber_nn /= (num_samples*32)
    
    #save ber numbers for plotting
    ts = datetime.datetime.now()
    filename = 'results/' + ts.strftime('%Y%m%d-%H%M%S') + '_ber.pkl'
    
    with open(filename, 'wb') as f:
        save_dict = {'ber_nn': ber_nn, 'ber_matlab': ber_matlab, 'bp': iterations, 'snr': snrdb}
        pickle.dump(save_dict, f)