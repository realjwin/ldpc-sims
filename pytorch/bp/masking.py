#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:38:01 2019

@author: jacobwinick
"""

import numpy as np
import scipy.io as sio

def generate_masks(H):
    """
    the nodes for each layer of the trellis are defined by 
    c(m,n) & v(m,n) where m denotes which check (variable) node
    and n denotes what variable (check) node is not included in
    its current LLR. So c(0,1) means check node 0, which does not
    contain any information from variable node v(c(0,1), *)
    
    the parity check matrix H is such that each row represents a
    parity check equation and each column is the set of parity check
    equations that the variable is part of
    """
    
    #node intersection matrix
    c = []
    v = []
    
    #these track the max number of
    #connections a v-node or c-node has
    max_clen = 0
    max_vlen = 0
    
    #nn layer size (mask size)
    num_nodes = np.sum(np.sum(H)) 
    mask_c = np.zeros((num_nodes, num_nodes))
    mask_v = np.zeros((num_nodes, num_nodes))
    mask_v_final = np.zeros((H.shape[1], num_nodes))
    
    #for each row in parity check matrix
    #get array of non-zero indices and append to c
    #simultaneously track the row with most connections
    #note: this could be done much faster but this works
    for idx, val in enumerate(H):
        c.append(np.nonzero(val)[0])
        if max_clen < len(c[idx]):
            max_clen = len(c[idx])

    #for each column in parity check matrix
    #repeat the above process but for the v nodes
    #this is what I want to use for the LLR expander!
    for idx, val in enumerate(np.transpose(H)):
        v.append(np.nonzero(val)[0])
        if max_vlen < len(v[idx]):
            max_vlen = len(v[idx])

    #create llr expander
    #this is made on the basis that
    #the v-nodes are ordered & collated (aka v0 v0 v0 v1 v1 v1 v1 ... etc)
    #this is true based on how I built the masking nodes
    llr_expander_list = [len(i) for i in v]
    llr_expander = np.zeros((np.sum(llr_expander_list),H.shape[1]))
    
    counter = 0
    for idx, val in enumerate(llr_expander_list):
        for i in range(0,val):
            llr_expander[counter][idx] = 1
            counter += 1

    #wtf does all this do from here & below??
    #it works but a little complicated so I
    #really need to comment and explain it
    #here we go...time to figure this out
    
    clookup = np.full((len(c),max_clen), -1)
    vlookup = np.full((len(v),max_vlen), -1)

    #check node lookup table
    #creates a linear index of a c-node
    #and which v-node it connects to
    #so clookup[m][n] is the linear index of
    #the m-th c-node connecting to the v-node
    #at at location c[m][n]
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
    
    #mask v
    #this is the message variable nodes construct
    #based on previous c-messages to send to c
    #for each check node
    for m in range(0, len(c)):
        #for each v-c connection
        for n in range(0, len(c[m])):
            #for each v-node that c connects to
            for idx, val in enumerate(v[c[m][n]]):
                #if that v node is not going back to c[m]
                #then connect the c-node to the v-node
                if val != m:
                    #clookup[m][n] gives the linear index
                    c_node = clookup[m][n]
                    v_node = vlookup[c[m][n]][idx]
                    
                    mask_v[v_node][c_node] = 1
                    
    #mask v final
    #this needs to take every message that connects to node v
    #but should not exclude the message of the edge it's going back to
    #if that makes sense. this sucks because I need to understand how
    #I wrote this goddamn code above from earlier, but this will have many
    #fewer final connections on the output so this is going to be the only
    #non-square matrix  coming out of here, what does this look like?
    
    for m in range(0, len(c)):
        for n in range(0, len(c[m])):    
            c_node = clookup[m][n]
            v_node = c[m][n]
            mask_v_final[v_node][c_node] = 1
            
    #mask c
    for m in range(0, len(v)):
        for n in range(0, len(v[m])):
            for idx, val in enumerate(c[v[m][n]]):
                #if the c node is not going back to v[m]
                if val != m:
                    v_node = vlookup[m][n]
                    c_node = clookup[v[m][n]][idx]
                    
                    mask_c[c_node][v_node] = 1

    
    
    #mask_vc: num_nodes x num_nodes
    #mask_cv: num_nodes x num_nodes
    #mask_c_final: codeword x num_nodes
    #llr_expander: num_nodes x codeword
    
    return mask_c, mask_v, mask_v_final, llr_expander

if __name__ == '__main__':
    #create c & v connection matrices
    filename = '../parity.mat'
    mat_contents = sio.loadmat(filename)
    H = mat_contents['H']
    
#    H = np.array([[1, 1, 0, 1, 1, 0, 0],
#              [0, 1, 1, 1, 0, 1, 0],
#              [1, 1, 1, 0, 0, 0, 1]])
    
    mask_c, mask_v, mask_v_final, llr_expander = genMasks(H)