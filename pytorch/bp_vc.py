import math
import numpy as np

import torch
import torch.nn as nn

#################################
# Define custome autograd function for masked connection.

class BeliefPropagationVC_Function(torch.autograd.Function):
    """
    implements x_i,e=(v,c) = 2 atanh ( mult x_i-1, e )
    
    this is effectively the message which the check nodes create based on
    the received info from the variable nodes, which is why it's called V->C
    
    note: it may be that the implementation of atanh is not stable
    """

    @staticmethod
    def forward(ctx, input, mask):
        
        #create "inverse" mask to convert 0's to 1's & vice-versa
        #since we're multiplying, after we mask the connections we want
        #we must convert post-mask 0's to the multiplicative identity (i.e. 1)
        inverse_mask = -1 * ( mask - 1 )
        
        masked_input = mask * input.view(1, -1).expand_as(mask) + inverse_mask
        
        #element wise product of the masked input
        #this is creating the non-fully connected architecture
        multiplied_input = torch.prod(masked_input, dim=1, keepdim=True)
        
        #if min(abs(1-multiplied_input.numpy())) < 1e-6:
        #    print('may have divide by zero error')
        #    print(min(abs(1-multiplied_input.numpy())))
        
        #it's required that | output_temp | < 1, is this enforced here?
        multiplied_input = torch.clamp(multiplied_input, -1, 1)
        
        #implementation of 2 * atanh (this may not be numerically stable)
        output = np.log(torch.div( 1 + multiplied_input, 1 - multiplied_input ))
        
        #save the output for the backprop
        ctx.save_for_backward(input, mask, multiplied_input)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        #load saved tensors
        input, mask, multiplied_input = ctx.saved_tensors
        
        #constant factor for derivative of function
        grad_matrix_const = 2 / (1 - multiplied_input**2)
        
        #create matrix which represents "almost-correct" derivative
        #each column has the same "almost-correct" derivative
        grad_matrix_temp = (grad_matrix_const * multiplied_input).view(1,-1).expand_as(mask.t())
        
        #divide each row by x_i-1,e to get correct derivative since we're
        #calculating d x_i,e' / d x_i-1,e where e is row index, e' is col index
        #then mask the connections which don't exist. it is the transpose of mask
        #because we're doing it "backwards" from the forward direction
        grad_matrix = mask.t() * ( grad_matrix_temp / input.view(-1,1).expand_as(mask.t()) )
        
        #compute the gradient wrt grad_output
        grad_input = grad_matrix.mm(grad_output)
        
        #mask has no gradient
        grad_mask = None
        
        return grad_input, grad_mask


class BeliefPropagationVC(nn.Module):
    def __init__(self, mask):
        """
        Computes messages at check nodes in BP
        
        Inputs:
            - mask [torch.tensor]:
                - shape: input x output
                - content: elts are 0 or 1 corresponding to a connection
                           from one layer node to the next
                
        Notes:
            - nn.Parameter is a special kind of tensor that will get
            automatically registered as the nn.Module's parameter once
            it's assigned as an attribute. This needs to be done to appear
            in the .parameters() and to be converted when calling .cuda()
            nn.Parameter requires gradients by default
        """
        
        super(BeliefPropagationVC, self).__init__()
        
        self.input_dim = mask.shape[0]
        self.output_dim = mask.shape[1]

        #setup mask tensor
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.double)
        else:
            self.mask = torch.tensor(mask, dtype=torch.double)

        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        return BeliefPropagationVC_Function.apply(input, self.mask)

#this checks that the gradients are working properly
if __name__ == '__main__':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    bp_vc = BeliefPropagationVC_Function.apply

    mask = torch.rand(20,20,dtype=torch.double,requires_grad=False)
    mask = torch.round(mask)

    input = (
            torch.randn(20,1,dtype=torch.double,requires_grad=True),
            mask
            )
    
    test = gradcheck(bp_vc, input, eps=1e-6, atol=1e-4)
    print(test)