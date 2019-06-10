#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:38:01 2019

@author: jacobwinick
"""

import numpy as np
import scipy.io as sio

def genMasks(parity_matrix):
    #the nodes for each layer of the trellis are defined by 
    #c(m,n) & v(m,n) where m denotes which check (variable) node
    #and n denotes what variable (check) node is not included in
    #its current LLR. So c(0,1) means check node 0, which does not
    #contain any information from variable node v(c(0,1), *)
    
    #node intersection matrix
    c = []
    v = []
    max_clen = 0
    max_vlen = 0
    
    #nn layer size (mask size)
    num_nodes = np.sum(np.sum(H)) 
    mask_cv = np.zeros((num_nodes, num_nodes))
    mask_vc = np.zeros((num_nodes, num_nodes))
    
    #for each row in parity check matrix
    for idx, val in enumerate(H):
        c.append(np.nonzero(val)[0])
        if max_clen < len(c[idx]):
            max_clen = len(c[idx])

    #for each column in parity check matrix
    for idx, val in enumerate(np.transpose(H)):
        v.append(np.nonzero(val)[0])
        if max_vlen < len(v[idx]):
            max_vlen = len(v[idx])

    clookup = np.full((len(c),max_clen), -1)
    vlookup = np.full((len(v),max_vlen), -1)

    #check node lookup table
    counter = 0
    for m in range(0, len(c)):
        for n in range(0, len(c[m])):
            clookup[m][n] = counter
            counter += 1
    
    #variable node lookup table
    counter = 0
    for m in range(0, len(v)):
        for n in range(0, len(v[m])):
            vlookup[m][n] = counter
            counter += 1
    
    #mask c -> v
    #for each check node
    for m in range(0, len(c)):
        for n in range(0, len(c[m])):
            for idx, val in enumerate(v[c[m][n]]):
                #if the v node is not going back to c[m]
                if val != m:
                    c_node = clookup[m][n]
                    v_node = vlookup[c[m][n]][idx]
                    
                    mask_cv[v_node][c_node] = 1
                        
    #mask v -> c
    for m in range(0, len(v)):
        for n in range(0, len(v[m])):
            for idx, val in enumerate(c[v[m][n]]):
                #if the c node is not going back to v[m]
                if val != m:
                    v_node = vlookup[m][n]
                    c_node = clookup[v[m][n]][idx]
                    
                    mask_vc[c_node][v_node] = 1
    
    return mask_cv, mask_vc


#create c & v connection matrices
filename = '../parity.mat'
mat_contents = sio.loadmat(filename)
H = mat_contents['H']

mask_cv, mask_vc = genMasks(H)