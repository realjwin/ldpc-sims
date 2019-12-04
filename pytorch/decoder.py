import numpy as np

import torch
import torch.nn as nn

from bp import *
from masking import genMasks

#batch_size must be divisible!
def decoder(llrs, H, bp_iterations, batch_size, clamp_value=10):
    
    output_bits = np.zeros(llrs.shape)
    
    num_batches = llrs.shape[0] // batch_size
    
    #--- NN SETUP ---#
    mask_vc, mask_cv, mask_v_final, llr_expander = genMasks(H)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        bp_model = nn.DataParallel(BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations))
    else:
        bp_model = BeliefPropagation(mask_vc, mask_cv, mask_v_final, llr_expander, bp_iterations)
    
    bp_model.eval()
    
    #send model to GPU
    bp_model.to(device)
        
    for batch in range(0, num_batches):
            start_idx = batch*batch_size
            end_idx =  (batch+1)*batch_size
                    
            llr = torch.tensor(-llrs[start_idx:end_idx, :], dtype=torch.float, device=device)                            
            x = torch.zeros(llr.shape[0], mask_cv.shape[0], dtype=torch.float, device=device)
        
            y_est = bp_model(x, llr, clamp_value)
        
            output_bits[start_idx:end_idx, :] = np.round(y_est.cpu().detach().numpy())
            
    return output_bits