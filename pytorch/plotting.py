import pickle
import numpy as np
import matplotlib.pyplot as plt

filename = 'results/20191016-122838_ber.pkl'
with open(filename, 'rb') as f:
    ber_dict = pickle.load(f)

ax = plt.gca()
fig = plt.gcf()
fig.set_size_inches(10, 8)

for bp_idx, bp_value in enumerate(ber_dict['bp']):
    ax.semilogy(ber_dict['snr'], ber_dict['ber_nn'][bp_idx], '*-', label='NN BP={}'.format(bp_value))    
    ax.semilogy(ber_dict['snr'], ber_dict['ber_matlab'][bp_idx], '+-', label='MATLAB BP={}'.format(bp_value))
        
ax.legend()

fig.suptitle('BER Comparison (Untrained)', fontsize=20)
plt.xlabel('SNR (dB)', fontsize=18)
plt.ylabel('BER', fontsize=16)
fig.savefig('ber.jpg')