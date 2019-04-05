#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:07:25 2019

@author: jacobwinick
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


#load data
with open('objs.pkl', 'rb') as f:
    variables = pickle.load(f)
    
BER_train_nn = variables[0]
BER_test_nn = variables[1]
BER_train_bp = variables[2]
BER_test_bp = variables[3]
snrdb = np.linspace(0.0, 5.0, 11)

plt.semilogy(snrdb, BER_test_nn)
plt.semilogy(snrdb, BER_test_bp)
plt.show