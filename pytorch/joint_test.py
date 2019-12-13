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

bp_iterations = 3
batch_size = num_samples
clamp_value = 20

#TODO: these should automatically be able to parse from results
qbits_val = 3 
clip_ratio = 1

results = '20191203-191640_tx=20191203-162534_quantized'

#--- LOAD RESULTS ---#

results_path = 'results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    filenames = data['filenames']
    snrdb = data['snrdb']

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

if torch.cuda.device_count() > 1:
    print("Using ", torch.cuda.device_count(), "GPUs.")

model = nn.DataParallel(Joint(ofdm_size, snr, mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations))

#send model to GPU
model.to(device)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)#, param.data)

#--- COMPUTE PERFORMANCE ---#

uncoded_ber = np.zeros(snrdb.shape)
coded_ber = np.zeros(snrdb.shape)
coded_bler = np.zeros(snrdb.shape)

uncoded_ber_quantized = np.zeros(snrdb.shape)
coded_ber_quantized = np.zeros(snrdb.shape)
coded_bler_quantized = np.zeros(snrdb.shape)

uncoded_ber_nn = np.zeros(snrdb.shape)
coded_ber_nn = np.zeros(snrdb.shape)
coded_bler_nn = np.zeros(snrdb.shape)

wmse_quantized = np.zeros(snrdb.shape)
wmse_nn = np.zeros(snrdb.shape)

for snrdb_idx, snrdb_val in enumerate(snrdb):
    
    #--- LOAD WEIGHTS ---#
    
    filepath = 'model/' + filenames[snrdb_idx]
    checkpoint = torch.load(filepath, map_location=device)
    
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
    
    #--- GENERATE DATA ---#
    
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
    
    uncoded_ber[snrdb_idx] = np.mean(np.abs(cbits - enc_bits))
    coded_ber[snrdb_idx] = np.mean(np.abs(bits[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits - enc_bits), axis=1)))
    
    #uncoded_ber_nn[snrdb_idx] = np.mean(np.abs(cbits_nn - enc_bits))
    coded_ber_nn[snrdb_idx] = np.mean(np.abs(bits_nn[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_nn[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_nn - enc_bits), axis=1)))
    
    uncoded_ber_quantized[snrdb_idx] = np.mean(np.abs(cbits_quantized - enc_bits))
    coded_ber_quantized[snrdb_idx] = np.mean(np.abs(bits_quantized[:, 0:32] - enc_bits[:, 0:32]))
    coded_bler_quantized[snrdb_idx] = np.mean(np.sign(np.sum(np.abs(bits_quantized - enc_bits), axis=1)))
