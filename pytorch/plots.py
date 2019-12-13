import pickle
import numpy as np
import matplotlib.pyplot as plt

#results_quantized = '20191203-191640_tx=20191203-162534_quantized'
#
#ber_path = 'ber_curves/' + results_quantized + '.pkl'
#
#with open(ber_path, 'rb') as f:
#    data = pickle.load(f)
#    
#    snrdb = data['snrdb']
#    
#    uncoded_ber = data['uncoded_ber']
#    coded_ber = data['coded_ber']
#    coded_bler = data['coded_bler']
#    
#    uncoded_ber_nn = data['uncoded_ber_nn']
#    coded_ber_nn = data['coded_ber_nn']
#    coded_bler_nn = data['coded_bler_nn']
#    
#    uncoded_ber_quantized = data['uncoded_ber_quantized']
#    coded_ber_quantized = data['coded_ber_quantized']
#    coded_bler_quantized = data['coded_bler_quantized']
#    
#    wmse_nn = data['wmse_nn']
#    wmse_quantized = data['wmse_quantized']
#
#fig, axes = plt.subplots(1, 2, figsize=(15,7))
#fig.suptitle('NN Performance on Quantized Inputs', fontsize=16, y=1.02)
#         
#axes[0].semilogy(snrdb, uncoded_ber, label='Uncoded Traditional')
#axes[0].semilogy(snrdb, coded_ber, label='Coded Traditional')
#axes[0].semilogy(snrdb, uncoded_ber_nn, '--+', label='Uncoded NN')
#axes[0].semilogy(snrdb, coded_ber_nn, '--+', label='Coded NN')
#axes[0].semilogy(snrdb, uncoded_ber_quantized, '--*', label='Uncoded Quantized')
#axes[0].semilogy(snrdb, coded_ber_quantized, '--*', label='Coded Quantized')
#axes[0].set_title('BER')
#axes[0].set_xlabel('SNR (dB)')
#axes[0].set_ylabel('BER')
#axes[0].legend()
#
#axes[1].semilogy(snrdb, coded_bler, label='Traditional')
#axes[1].semilogy(snrdb, coded_bler_nn, '--+', label='NN')
#axes[1].semilogy(snrdb, coded_bler_quantized, '--*', label='Quantized')
#axes[1].set_title('BLER')
#axes[1].set_xlabel('SNR (dB)')
#axes[1].set_ylabel('BLER')
#axes[1].legend()
#
#plt.tight_layout()
#plt.savefig('quantized_nn.eps', format='eps', bbox_inches='tight')
##plt.show()
#
##--- UNQUANTIZED TRAINING ---#
#
results = '20191203-191640_tx=20191203-162534'

ber_path = 'ber_curves/' + results + '.pkl'

with open(ber_path, 'rb') as f:
    data = pickle.load(f)
    
    snrdb_uquant = data['snrdb']
    
    uncoded_ber = data['uncoded_ber']
    coded_ber = data['coded_ber']
    coded_bler = data['coded_bler']
    
    uncoded_ber_nn = data['uncoded_ber_nn']
    coded_ber_nn = data['coded_ber_nn']
    coded_bler_nn = data['coded_bler_nn']
    
    wmse = data['wmse']

fig, axes = plt.subplots(1, 2, figsize=(15,7))
fig.suptitle('NN Performance on Unquantized Inputs', fontsize=16, y=1.02)
         
axes[0].semilogy(snrdb_uquant, uncoded_ber, '--o', label='Uncoded Traditional')
axes[0].semilogy(snrdb_uquant, coded_ber, '--o', label='Coded Traditional')
axes[0].semilogy(snrdb_uquant, uncoded_ber_nn, '--s', label='Uncoded NN')
axes[0].semilogy(snrdb_uquant, coded_ber_nn, '--s', label='Coded NN')
axes[0].set_title('BER', fontsize = 14)
axes[0].set_xlabel('SNR (dB)', fontsize = 14)
axes[0].set_ylabel('BER', fontsize = 14)
axes[0].legend(fontsize = 12)

axes[1].semilogy(snrdb_uquant, coded_bler, '--o', label='Traditional')
axes[1].semilogy(snrdb_uquant, coded_bler_nn, '--s', label='NN')
axes[1].set_title('BLER', fontsize = 14)
axes[1].set_xlabel('SNR (dB)', fontsize = 14)
axes[1].set_ylabel('BLER', fontsize = 14)
axes[1].legend(fontsize = 12)

plt.tight_layout()
plt.savefig('unquantized_nn.eps', format='eps', bbox_inches='tight')
#plt.show()

fig, ax = plt.subplots(figsize=(10,7))
fig.suptitle('Weighted Mean Square Error Comparison', fontsize=16, y=1.02)

ax.plot(snrdb_uquant, wmse, label='Unquantized NN')
ax.plot(snrdb_uquant, wmse_nn, label='Quantized NN')
ax.plot(snrdb_uquant, wmse_quantized, label='Quantized Traditional')
ax.set_xlabel('SNR (dB)', fontsize = 14)
ax.set_ylabel('WMSE', fontsize = 14)
ax.legend()

plt.tight_layout()

#--- GRID SEARCH ---#

results = '20191203-191640_tx=20191203-162534_quantized_half'

ber_path = 'ber_curves/' + results + '.pkl'

with open(ber_path, 'rb') as f:
    data = pickle.load(f)
    
    snrdb = data['snrdb']
    qbits = data['qbits']
    clipdb = data['clipdb']
    
    uncoded_ber = data['uncoded_ber']
    coded_ber = data['coded_ber']
    coded_bler = data['coded_bler']
    
    uncoded_ber_nn = data['uncoded_ber_nn']
    coded_ber_nn = data['coded_ber_nn']
    coded_bler_nn = data['coded_bler_nn']
    
    uncoded_ber_quantized = data['uncoded_ber_quantized']
    coded_ber_quantized = data['coded_ber_quantized']
    coded_bler_quantized = data['coded_bler_quantized']
    
    wmse_nn = data['wmse_nn']
    wmse_quantized = data['wmse_quantized']

fig, axes = plt.subplots(1, 2, figsize=(15,7))
fig.suptitle('NN Performance on Quantized Inputs', fontsize=18, y=1.02)

axes[0].semilogy(snrdb, uncoded_ber, '-vk', label='Uncoded Traditional')
axes[0].semilogy(snrdb, coded_ber, '-^k', label='Coded Traditional')
for qbits_idx, qbits_val in enumerate(qbits):
    #for clipdb_idx, clipdb_val in enumerate(clipdb):
    clipdb_idx = 0
    clipdb_val = 0
    axes[0].semilogy(snrdb, uncoded_ber_nn[:, qbits_idx, clipdb_idx], '--s', label='Uncoded NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[0].semilogy(snrdb, coded_ber_nn[:, qbits_idx, clipdb_idx], '--s', label='Coded NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[0].semilogy(snrdb, uncoded_ber_quantized[:, qbits_idx, clipdb_idx], '--o', label='Uncoded Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[0].semilogy(snrdb, coded_ber_quantized[:, qbits_idx, clipdb_idx], '--o', label='Coded Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
axes[0].set_title('\u03BC= 0 dB', fontsize = 16)
axes[0].set_xlabel('SNR (dB)', fontsize = 14)
axes[0].set_ylabel('BER', fontsize = 14)
axes[0].legend(fontsize = 12)

axes[0].tick_params(axis="x", labelsize=14)
axes[0].tick_params(axis="y", labelsize=14)

axes[1].semilogy(snrdb, uncoded_ber, '-vk', label='Uncoded Traditional')
axes[1].semilogy(snrdb, coded_ber, '-^k', label='Coded Traditional')
for qbits_idx, qbits_val in enumerate(qbits):
    #for clipdb_idx, clipdb_val in enumerate(clipdb):
    clipdb_idx = 1
    clipdb_val = 5
    axes[1].semilogy(snrdb, uncoded_ber_nn[:, qbits_idx, clipdb_idx], '--s', label='Uncoded NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[1].semilogy(snrdb, coded_ber_nn[:, qbits_idx, clipdb_idx], '--s', label='Coded NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[1].semilogy(snrdb, uncoded_ber_quantized[:, qbits_idx, clipdb_idx], '--o', label='Uncoded Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
    axes[1].semilogy(snrdb, coded_ber_quantized[:, qbits_idx, clipdb_idx], '--o', label='Coded Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
axes[1].set_title('\u03BC= 5 dB', fontsize = 16)
axes[1].set_xlabel('SNR (dB)', fontsize = 14)
axes[1].set_ylabel('BER', fontsize = 14)
axes[1].legend(fontsize = 12)

axes[1].tick_params(axis="x", labelsize=14)
axes[1].tick_params(axis="y", labelsize=14)

#axes[1].semilogy(snrdb, coded_bler, '-vk', label='Traditional')
#for qbits_idx, qbits_val in enumerate(qbits):
#    for clipdb_idx, clipdb_val in enumerate(clipdb):
#        axes[1].semilogy(snrdb, coded_bler_nn[:, qbits_idx, clipdb_idx], '--s', label='NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
#        axes[1].semilogy(snrdb, coded_bler_quantized[:, qbits_idx, clipdb_idx], '--o', label='Quantized, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
#        axes[1].set_title('BLER', fontsize = 16)
#        axes[1].set_xlabel('SNR (dB)', fontsize = 14)
#        axes[1].set_ylabel('BLER', fontsize = 14)
#        axes[1].legend(fontsize = 12)
#
#axes[1].tick_params(axis="x", labelsize=14)
#axes[1].tick_params(axis="y", labelsize=14)

plt.tight_layout()
plt.savefig('quantized_nn.eps', format='eps', bbox_inches='tight')


#--- WMSE PLOTS ---#

#fig, axes = plt.subplots(1, 2, figsize=(15,7))
#fig.suptitle('Weighted Mean Square Error Comparison', fontsize=16, y=1.02)
#
#axes[0].plot(snrdb_uquant, wmse, '-vk', label='Unquantized NN')
#axes[1].plot(snrdb_uquant, wmse, '-vk', label='Unquantized NN')
#
#for qbits_idx, qbits_val in enumerate(qbits):
#    clipdb_idx = 0
#    clipdb_val = 0
#    axes[0].plot(snrdb, wmse_nn[:, qbits_idx, clipdb_idx], '--s', label='Quantized NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
#    axes[0].plot(snrdb, wmse_quantized[:, qbits_idx, clipdb_idx], '--o', label='Quantized Traditional, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
#
#axes[0].set_xlabel('SNR (dB)', fontsize = 14)
#axes[0].set_ylabel('WMSE', fontsize = 14)
#axes[0].legend(fontsize = 12)
#axes[0].tick_params(axis="x", labelsize=14)
#axes[0].tick_params(axis="y", labelsize=14)
#
#for qbits_idx, qbits_val in enumerate(qbits):
#    clipdb_idx = 1
#    clipdb_val = 5
#    axes[1].plot(snrdb, wmse_nn[:, qbits_idx, clipdb_idx], '--s', label='Quantized NN, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
#    axes[1].plot(snrdb, wmse_quantized[:, qbits_idx, clipdb_idx], '--o', label='Quantized Traditional, {}-bits, \u03BC={}dB'.format(qbits_val, clipdb_val))
#
#axes[1].set_xlabel('SNR (dB)', fontsize = 14)
#axes[1].set_ylabel('WMSE', fontsize = 14)
#axes[1].legend(fontsize = 12)
#axes[1].tick_params(axis="x", labelsize=14)
#axes[1].tick_params(axis="y", labelsize=14)
#
#plt.tight_layout()
#plt.savefig('wmse.eps', format='eps', bbox_inches='tight')