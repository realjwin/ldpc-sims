#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:27:59 2019

@author: jacobwinick
"""

import math
import numpy as np

import torch
import torch.nn as nn

#################################
# Define custome autograd function for masked connection.

class BeliefPropagationVC_Function(torch.autograd.Function):
    """
    implements x_i,e=(v,c) = 2 atanh ( mult x_i-1, e )
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, mask):
        #convert tensor to row vector
        #broadcast input to the mask size (aka repeat rows)
        input_temp1 = input.view(1, -1).expand_as(mask)
        
        #create "inverse" mask to convert 0's to 1's & vice-versa
        mask_add = -1 * ( mask - 1 )
        
        #get input & connections ready for multiplication
        input_temp2 = mask * input_temp1 + mask_add
        
        #element wise product, where some values are "zero'd" (aka 1)
        #this is creating the non-fully connected architecture
        #this is now a COLUMN vector
        output_temp = torch.prod(input_temp2, dim=1, keepdim=True)
        
        #implementation of 2 * atanh (this may not be numerically stable)
        #it's required that | output_temp | < 1, is this enforced here?
        output = np.log(torch.div( 1 + output_temp, 1 - output_temp ))

        ctx.save_for_backward(input, mask, output_temp)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        
        input, mask, output_temp = ctx.saved_tensors
        grad_input = None
        
        #no gradient required for mask, so not implementing
        grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            #compute the matrix using mask
            #a row in a mask corresponds to the first column of my result
            #so if a row has a 1 in it, then it means that x_i-1,e connects to
            #x_i,e and that means that it should be included in the list
            #otherwise it should be zero
            
            #basically generate output_temp, and now you have the product for
            #each output node, so use that for each column and then divide by
            #the input for that derivative (this is probably like...not the best way
            #but should be close enough that we don't have an issue). This means
            #divide each ROW (not column) by the corresponding input
            
            #output temp is a column vector which represents the 
            #multiplication of messages but before atanh
            #we then need to broadcast this (repeat rows)
            output_temp = output_temp.view(1,-1)
            deriv_matrix_pre = 2 / 1 - (output_temp^2)
            
            #so we need to take output temp and skip the one not being used 
            for row in mask:
                for col in mask:
                    if mask[row][col] == 0:
                        deriv_matrix[row][col] = 0
                    else:
                        deriv_matrix[row][col] = output_temp[row] / input[col]
                                
            grad_input = deriv_matrix.t().matmul(grad_output)

        return grad_input, grad_mask


class BeliefPropagationVC(nn.Module):
    def __init__(self, mask):
        """
        extended torch.nn module which mask connection.

        mask [torch.tensor]:
            the shape is (n_output_feature, n_input_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        """
        
        super(BeliefPropagationVC, self).__init__()
        
        self.output_features = mask.shape[0]
        self.input_features = mask.shape[1]

        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float)
        else:
            self.mask = torch.tensor(mask, dtype=torch.float)

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


#THIS IS HOW TO CHECK THINGS ARE WORKING RIGHT
if __name__ == 'check grad':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    customlinear = CustomizedLinearFunction.apply

    input = (
            torch.randn(20,20,dtype=torch.double,requires_grad=True),
            torch.randn(30,20,dtype=torch.double,requires_grad=True),
            None,
            None,
            )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)