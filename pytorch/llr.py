import torch
import torch.nn as nn
import numpy as np
from ofdm_functions import DFTreal

class LLRestimator(nn.Module):
    def __init__(self, ofdm_size, snr_est):
        super(LLRestimator, self).__init__()
        
        self.ofdm_size = ofdm_size
        self.snr_est = snr_est
        
        self.activation = nn.Tanh()
        
        self.fft_layer = nn.Linear(2*self.ofdm_size, 2*self.ofdm_size, bias=False)
        self.scalar = nn.Parameter(torch.ones(1, 2*self.ofdm_size, out=None, dtype=torch.float, requires_grad=True))
        
        self.hidden1 = nn.Linear(2*self.ofdm_size, 8*self.ofdm_size, bias=True)
        self.hidden2 = nn.Linear(8*self.ofdm_size, 2*self.ofdm_size, bias=True)        
        self.hidden3 = nn.Linear(2*self.ofdm_size, 16*self.ofdm_size, bias=True)
        self.hidden4 = nn.Linear(16*self.ofdm_size, 16*self.ofdm_size, bias=True)
        
        self.final = nn.Linear(16*self.ofdm_size, 2*self.ofdm_size, bias=True)
        
        self.bn3 = nn.BatchNorm1d(16*self.ofdm_size)
        self.bn4 = nn.BatchNorm1d(16*self.ofdm_size)
        
        
        #initialized parameters
        self.init_parameters()
        
    def init_parameters(self):
        
        #fft layer
        self.fft_layer.weight.data = torch.tensor(DFTreal(self.ofdm_size), dtype=torch.float, requires_grad=True)
        
        #weighted scalar
        self.scalar.data = torch.tensor(2*self.snr_est * (-2/np.sqrt(2)), dtype=torch.float, requires_grad=True).expand_as(self.scalar.data)
        
    def forward(self, x):
        x = self.fft_layer(x)
        
        #x = self.activation(self.hidden1(x))
        #x = self.activation(self.hidden2(x))
        
        #x = self.fft_layer(x)
        #x = self.scalar * x
        
        x = self.activation(self.hidden3(x))
        #x = self.bn3(x)
        x = self.activation(self.hidden4(x))
        #x = self.bn4(x)
        
        return self.final(x)