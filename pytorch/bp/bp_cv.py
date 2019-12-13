import math
import numpy as np

import torch
import torch.nn as nn

#################################
##NOTE: This will be so slow on a big matrix due to masking both in fwd & rev

class BeliefPropagationCV_Function(torch.autograd.Function):
    """
    implements x_i,e=(v,c) = 2 atanh ( mult x_i-1, e )
    
    this is effectively the message which the check nodes create to send
    to the variable nodes based on previously received info from the
    variable nodes, which is why it's called C->V
    
    note: it may be that the implementation of atanh is not stable
    """

    @staticmethod
    def forward(ctx, input, mask):
        
        input_expanded = input.unsqueeze(1).expand([-1, mask.shape[0], -1])
        
        #create "inverse" mask to convert 0's to 1's & vice-versa
        #since we're multiplying, after we mask the connections we want
        #we must convert post-mask 0's to the multiplicative identity (i.e. 1)
        inverse_mask = -1 * ( mask - 1 )
        
        #this is to prevent the issue of a 1 being the output of a mult x_i-1, e
        #when no message is being sent back to the variable node (i.e. when the
        #variable node is the check node). this corrects the issue by placing
        #a zero in the product to zero it out.
        #inverse_mask_correction = torch.prod(inverse_mask, dim=1)
        #inverse_mask[:,0] = inverse_mask[:,0] - inverse_mask_correction
        
        masked_input = mask.expand_as(input_expanded) * input_expanded + inverse_mask.expand_as(input_expanded)
        
        #element wise product of the masked input
        #this is creating the non-fully connected architecture
        multiplied_input = torch.prod(masked_input, dim=2)

        epsilon = .0000001

        #it's required that | output_temp | < 1, enforced with epislon
        multiplied_input = torch.clamp(multiplied_input, -(1-epsilon), (1-epsilon))
        
        #implementation of 2 * atanh (this may not be numerically stable)
        output = torch.div( 1 + multiplied_input, 1 - multiplied_input ).log_()
        
        #save the output for the backprop
        ctx.save_for_backward(input, mask, multiplied_input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        #load saved tensors
        input, mask, multiplied_input= ctx.saved_tensors
        
        #constant factor for derivative of function
        grad_matrix_const = 2 / (1 - multiplied_input**2)
        
        #cuda stuff
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #creates 3-d matrix where m[i][:][i] = 1, else m[i][j][k] = 0
        #this is used to eliminate x[i] during multiplication for gradient purposes
        grad_matrix_eye = torch.eye(mask.shape[1], dtype=torch.float, device=device).unsqueeze(1).expand([-1, mask.shape[0], -1])
        
        #this creates a mask for each x dimension which does not include the x
        #when multiplying the mask, by effectively computing an xor elt-by-elt
        grad_matrix_xor = torch.clamp(mask.unsqueeze(0).expand_as(grad_matrix_eye) - grad_matrix_eye, 0)
        
        #expand to support multiple x inputs
        grad_matrix_xor = grad_matrix_xor.unsqueeze(0).expand(input.shape[0], -1, -1, -1)
        
        #create the inverse mask since we are multiplying
        grad_matrix_xor_inverse = -1 * ( grad_matrix_xor - 1)
        
        #expand x to the right size
        input_expanded_xor = input.unsqueeze(1).unsqueeze(1).expand_as(grad_matrix_xor)
        
        grad_matrix_temp = torch.transpose(torch.prod(grad_matrix_xor * input_expanded_xor + grad_matrix_xor_inverse, dim=3), 1, 2)
        
        grad_matrix = torch.transpose(grad_matrix_const.unsqueeze(1), 1, 2).expand_as(grad_matrix_temp) * mask.expand_as(grad_matrix_temp) * grad_matrix_temp
        
        #compute the gradient wrt grad_output
        grad_input = torch.bmm(grad_output.unsqueeze(1), grad_matrix).squeeze(1)
        
        #mask has no gradient
        grad_mask = None
        
        return grad_input, grad_mask


class BeliefPropagationCV(nn.Module):
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
        
        super(BeliefPropagationCV, self).__init__()
        
        self.output_dim = mask.shape[0]
        self.input_dim = mask.shape[1]

        #setup mask tensor
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float)
        else:
            self.mask = torch.tensor(mask, dtype=torch.float)

        self.mask = nn.Parameter(self.mask, requires_grad=False)

    def forward(self, input):
        return BeliefPropagationCV_Function.apply(input, self.mask)

#this checks that the gradients are working properly
if __name__ == '__main__':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    bp_cv = BeliefPropagationCV_Function.apply
    
    #torch.manual_seed(3)
    mask = torch.empty(20, 20, dtype=torch.float,requires_grad=False).uniform_(0, 1)
    mask = torch.round(mask)
    
    input_temp = np.random.uniform(-1,1,(2,20))

    input = (
            torch.tensor(input_temp, dtype=torch.float, requires_grad=True),
            mask
            )
    
    test = gradcheck(bp_cv, input, eps=1e-3, atol=1e-4)
    print(test)