import numpy as np
import datetime as datetime

import torch
import torch.nn as nn
import torch.optim as optim

from nn.llr import LLRestimator
from ofdm.ofdm_functions import *
from bp.bp import BeliefPropagation

class Joint(nn.Module):
    def __init__(self, H, iterations):
        super(Joint, self).__init__()
        
        self.LLRest = LLRestimator(ofdm_size, snr)
        
        self.BP = BeliefPropagation(H, iterations)
        
        self.layer_size_val = self.BP.layer_size()
        
    def forward(self, signal, x, clamp_value):
        
        llr = self.LLRest(signal)
        
        return self.BP(x, llr, clamp_value)
    
    def layer_size(self):
        return self.layer_size_val