import pickle
import numpy as np
import datetime as datetime

from ofdm.ofdm_nn import train_nn_withSNR
from ofdm.ofdm_functions import *

#--- VARIABLES ---#

ofdm_size = 32
num_epochs = 20000
batch_size = np.power(2, 14)
lr = .1

qbits = np.array([3])
clipdb = np.array([0])

filenames = []

#--- LOAD PRETRAINED NETWORK ---#
    
results = '20191214-002518_tx=20191213-234355_unquantized_withsnr'

results_path = 'outputs/results/' + results + '.pkl'

with open(results_path, 'rb') as f:
    data = pickle.load(f)

    pretrained_model = data['filename']
    snrdb_low = data['snrdb_low']
    snrdb_high = data['snrdb_high']

#--- LOAD DATA ---#

timestamp = results.split('_')[1].split('=')[1]

tx_file = timestamp + '_tx.pkl'
tx_filepath = 'outputs/tx/' + tx_file

with open(tx_filepath, 'rb') as f:
    data = pickle.load(f)

    enc_bits = data['enc_bits']
    tx_symbols = data['tx_symbols']    
    num_samples = data['num_samples']

#--- GENERATE DATA ---#

snrdb = np.random.uniform(snrdb_low, snrdb_high, (num_samples, 1))

symbols = tx_symbols.reshape((-1, ofdm_size)).T
ofdm_symbols = np.matmul(DFT(ofdm_size).conj().T, symbols)

snr_val = np.power(10, np.broadcast_to(snrdb.T, symbols.shape)/10)

noise = (np.random.normal(0, 1/np.sqrt(snr_val)) + 
         1j*np.random.normal(0, 1/np.sqrt(snr_val))) / np.sqrt(2)

received_symbols = ofdm_symbols + noise

deofdm_symbols = np.matmul(DFT(ofdm_size), received_symbols)

noise_power = .5 * (1 / snr_val)

#this is log(Pr=1 / Pr=0) aka +inf = 1, -inf = -1
llr_bit0 = ( np.power(deofdm_symbols.real - 1/np.sqrt(2), 2) - 
            np.power(deofdm_symbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
llr_bit1 = ( np.power(deofdm_symbols.imag - 1/np.sqrt(2), 2) - 
            np.power(deofdm_symbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
  
llrs = np.concatenate((llr_bit0.T.reshape((-1,1)), llr_bit1.T.reshape((-1,1))), axis=1)
 
rx_signal = received_symbols.T.reshape((1,-1))
rx_llrs = llrs.reshape((1,-1))

#--- TRAIN QUANTIZED ---#

for qbits_idx, qbits_val in enumerate(qbits):
    for clipdb_idx, clipdb_val in enumerate(clipdb):
        print('Q-Bits: {}, Clip: {} dB'.format(qbits_val, clipdb_val))
        
        #how does automatic gain control work in the context of an OFDM system?
        clip_ratio = np.power(10, (clipdb_val/10))        

        snr = np.power(10, snrdb/10)
        
        #keep clip constant, adjust input signal
        agc_clip = 10

        #compute sigma_rx per dimension (real/imag)
        #this is approximate incoming signal amplitude
        #each column is an OFDM symbol with a different SNR
        sigma_rx =  .5 * (1 + 1/snr)

        #compute clipping per signal (this will be very rough b/c )
        factor = agc_clip / sigma_rx * clip_ratio
        
        rx_signal_scaled = np.broadcast_to(factor.T, received_symbols.shape) * received_symbols
        rx_signal_scaled = rx_signal_scaled.T.reshape((1,-1))
        
        qrx_signal = quantizer(rx_signal_scaled, qbits, agc_clip)
        
        qrx_signal_rescaled = qrx_signal.reshape((-1, ofdm_size)).T
        qrx_signal_rescaled = qrx_signal_rescaled / np.broadcast_to(factor.T, qrx_signal_rescaled.shape)
        
        deofdm_qsymbols = np.matmul(DFT(ofdm_size), qrx_signal_rescaled)
        
        noise_power = .5 * (1 / snr_val)
        
        #this is log(Pr=1 / Pr=0) aka +inf = 1, -inf = -1
        qllr_bit0 = ( np.power(deofdm_qsymbols.real - 1/np.sqrt(2), 2) - 
                    np.power(deofdm_qsymbols.real + 1/np.sqrt(2), 2) ) / (2*noise_power)
        qllr_bit1 = ( np.power(deofdm_qsymbols.imag - 1/np.sqrt(2), 2) - 
                    np.power(deofdm_qsymbols.imag + 1/np.sqrt(2), 2) ) / (2*noise_power)
          
        qllrs = np.concatenate((qllr_bit0.T.reshape((-1,1)), qllr_bit1.T.reshape((-1,1))), axis=1)

        qrx_llrs = qllrs.reshape((1,-1))
        
        input_samples = np.concatenate((qrx_signal_rescaled.real.T, qrx_signal_rescaled.imag.T), axis=1)
        input_samples = input_samples.reshape(-1, 2*ofdm_size)
        input_samples = np.concatenate((input_samples, snr), axis=1)
        
        output_samples = rx_llrs.reshape(-1, 2*ofdm_size)
            
        #--- TRAIN NETWORK ---#
        
        filename = train_nn_withSNR(input_samples, output_samples, timestamp, snrdb_low, snrdb_high, lr, qbits_val, clipdb_val, ofdm_size, num_epochs, batch_size, pretrained_model)

        filenames.append(filename)

#--- SAVE LIST OF FILENAMES ---#

ts = datetime.datetime.now()

modelfile = ts.strftime('%Y%m%d-%H%M%S') + '_tx=' + timestamp + '_quantized_withsnr.pkl'
modelpath = 'outputs/results/' + modelfile

with open(modelpath, 'wb') as f:
    save_dict = {
            'filenames': filenames,
            'snrdb_low': snrdb_low,
            'snrdb_high': snrdb_high}
    
    pickle.dump(save_dict, f)
    
print(modelpath)