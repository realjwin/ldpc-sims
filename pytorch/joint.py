import numpy as np
import datetime as datetime

import torch
import torch.nn as nn
import torch.optim as optim

from llr import LLRestimator
from ofdm_functions import *
from masking import genMasks
from bp import *

class Joint(nn.Module):
    def __init__(self, ofdm_size, snr, mask_c, mask_v, mask_v_final, llr_expander, iterations):
        super(Joint, self).__init__()
        
        self.LLRest = LLRestimator(ofdm_size, snr)
        
        self.BP = BeliefPropagation(mask_c, mask_v, mask_v_final, llr_expander, iterations)
        
    def forward(self, signal, x, clamp_value):
        
        llr = self.LLRest(signal)
        
        return self.BP(x, llr, clamp_value)