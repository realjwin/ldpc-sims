#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:48:40 2019

@author: jacobwinick
"""

import numpy as np
import torch.nn as nn
from bp_cv import BeliefPropagationCV
from bp_vc import BeliefPropagationVC

class BeliefPropagation(nn.Module):
    def __init__(self, mask_cv, mask_vc, mask_cv_final, llr_expander, iterations):
        super(BeliefPropagation, self).__init__()
        
        #do i need to setup internal variables?
        #self.mask_cv = mask_cv ...etc
        
        self.BeliefPropagationIter = nn.Sequential(
                BeliefPropagationVC(mask_vc, llr_expander),
                nn.Tanh(),
                BeliefPropagationCV(mask_cv)
                )
        
        self.layers = nn.ModuleList([self.BeliefPropagationIter 
                                     for i in range(0,iterations)])
    
        self.final_layer = nn.Sequential(
                BeliefPropagationVC(mask_v_final, np.eye(mask_v_final.shape[0])),
                nn.Sigmoid()
                )

    def forward(self, x, llr, clamp_value):
        
        #BP algorithm
        for layer in self.layers:
            x = layer([x, llr]).clamp(-clamp_value, clamp_value)

        #output layer
        return self.final_layer([x, llr])