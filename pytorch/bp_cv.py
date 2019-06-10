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

class BeliefPropagationCV_Function(torch.autograd.Function):
    """
    implements x_i,e=(v,c) = tanh ( 1/2 ( w_i,v * l_v + sum( w_i,e,e' x_i-1,e' ) ) )
    this will be challenging...
    this is effectively the message which the variable nodes create based on
    the received info from the check nodes, which is why it's called C->V
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, input_weight, expanded_llr, expanded_llr_weight, mask):
        
        #convert tensor to column vector
        weighted_input = input_weight.mm(input.view(-1, 1))
        output = .5 * mask.mm(weighted_input) + .5 * expanded_llr.mm(expanded_llr_weight)

        ctx.save_for_backward(input, input_weight, expanded_llr, expanded_llr_weight, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        
        input, input_weight, expanded_llr, expanded_llr_weight, mask = ctx.saved_tensors
        grad_input = .5 * input_weight.t().mm(grad_output)
        grad_input_weight = .5 * 
        grad_expanded_llr = None
        grad_expanded_llr_weight = .5 *
        grad_mask = None
        
        return grad_input, grad_input_weight, grad_expanded_llr, grad_expanded_llr_weight, grad_mask


class BeliefPropagationCV(nn.Module):
    def __init__(self, mask, llr):
        """
        extended torch.nn module which mask connection.

        mask [torch.tensor]:
            the shape is (n_output_feature, n_input_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        """
        
        super(BeliefPropagationCV, self).__init__()
        
        self.output_features = mask.shape[0]
        self.input_features = mask.shape[1]

        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float)
        else:
            self.mask = torch.tensor(mask, dtype=torch.float)

        self.mask = nn.Parameter(self.mask, requires_grad=False)
        
        
        if isinstance(llr, torch.Tensor):
            self.llr = llr.type(torch.float)
        else:
            self.llr = torch.tensor(llr, dtype=torch.float)
            
        self.llr = nn.Parameter(self.mask, requires_grad=False)

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