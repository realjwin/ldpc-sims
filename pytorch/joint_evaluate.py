import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections

import torch
import torch.nn as nn

from parity import *
from decoder import decoder
from ofdm_functions import *
from llr import LLRestimator
from joint import Joint
from masking import genMasks

#--- VARIABLES ---#

num_samples = np.power(2,8) #CHANGE THIS VALUE!
ofdm_size = 32

bp_iterations = 20 #3
batch_size = num_samples
clamp_value = 3 #20

#TODO: these should automatically be able to parse from results
qbits_val = 3 
clip_ratio = 1

trained_model = '20191204-203947_qbits=3_clipdb=0_snr=5.0_lr=1_joint'

#--- GENERATE BITS ---#

num_bits = 2 * num_samples * ofdm_size

bits = create_bits(num_bits//2)

enc_bits = encode_bits(bits, G)

tx_symbols = modulate_bits(enc_bits)

#--- NN MODEL ---#

snr = 0 #placeholder because it currently does nothing

mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(H)

#for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--- LOAD RESULTS ---#

results_path = 'model/' + trained_model + '.pth'

checkpoint = torch.load(results_path, map_location=device)
data_timestamp = checkpoint['data_timestamp']

data_timestamp = 

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs.")

model = nn.DataParallel(Joint(ofdm_size, snr, mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations))

#send model to GPU
model.to(device)

model.load_state_dict(checkpoint['model_state_dict'])

for name, param in model.named_parameters():
    if param.requires_grad and name.split('.')[1] == 'LLRest':
        print(name, param.data)

#--- COMPUTE PERFORMANCE ---#

#uncoded_ber = np.zeros(snrdb.shape)
#coded_ber = np.zeros(snrdb.shape)
#coded_bler = np.zeros(snrdb.shape)
#
#uncoded_ber_quantized = np.zeros(snrdb.shape)
#coded_ber_quantized = np.zeros(snrdb.shape)
#coded_bler_quantized = np.zeros(snrdb.shape)
#
#uncoded_ber_nn = np.zeros(snrdb.shape)
#coded_ber_nn = np.zeros(snrdb.shape)
#coded_bler_nn = np.zeros(snrdb.shape)
#
#wmse_quantized = np.zeros(snrdb.shape)
#wmse_nn = np.zeros(snrdb.shape)
    
#--- GENERATE DATA ---#
    
snrdb_val = 5
qbits_val = 3

rx_signal, rx_symbols, rx_llrs = gen_data(tx_symbols, snrdb_val, ofdm_size)

qrx_signal, qrx_symbols, qrx_llrs = gen_qdata(rx_signal, snrdb_val, qbits_val, clip_ratio, ofdm_size)

input_samples = np.concatenate((qrx_signal.real.T, qrx_signal.imag.T), axis=1)
input_samples = input_samples.reshape(-1, 2*ofdm_size)

qrx_llrs = qrx_llrs.reshape(-1, 2*ofdm_size)
rx_llrs = rx_llrs.reshape(-1, 2*ofdm_size)
enc_bits = enc_bits.reshape(-1, 2*ofdm_size)

#--- INFERENCE ---#

x_input = torch.tensor(input_samples, dtype=torch.float, device=device)
x_temp = torch.zeros(x_input.shape[0], mask_cv.shape[0], dtype=torch.float, device=device)

with torch.no_grad():
    bits_nn = model(x_input, x_temp, clamp_value)
    
bits_nn = np.round(bits_nn.cpu().detach().numpy())

#--- DECODING PERFORMANCE ---#

cbits = (np.sign(rx_llrs) + 1) // 2
bits = decoder(rx_llrs, H, bp_iterations, batch_size, clamp_value)

cbits_quantized = (np.sign(qrx_llrs) + 1) // 2
bits_quantized = decoder(qrx_llrs, H, bp_iterations, batch_size, clamp_value)

coded_ber_nn= np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
coded_bler_nn = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))

#uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
#coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
#coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
#
##uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
#coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
#coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
#
#uncoded_ber_quantized[snrdb_idx] = np.mean(np.abs(cbits_quantized - enc_bits))
#coded_ber_quantized[snrdb_idx] = np.mean(np.abs(bits_quantized[:, 0:32] - enc_bits[:, 0:32]))
#coded_bler_quantized[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_quantized - enc_bits), axis=1)))
