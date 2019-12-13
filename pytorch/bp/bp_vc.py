import numpy as np

import torch
import torch.nn as nn

class BeliefPropagationVC_Function(torch.autograd.Function):
    """
    implements x_i,e=(v,c) = 1/2 ( w_i,v * l_v + sum( w_i,e,e' x_i-1,e' ) )
    
    this is effectively the message which the variable nodes create to send
    to the check nodes based on previously received info from the
    check nodes, which is why it's called V->C
    """

    @staticmethod
    def forward(ctx, input, input_weight, mask, llr, llr_weight, llr_expander):
        
        #weight & mask the input values
        weighted_input = input.mm((mask * input_weight).t_())
        
        #create expanded version of weighted initial LLR estimate
        #repeat the llr for each message which needs that LLR,
        #but weights are the same for each LLR
        expanded_llr = (llr_weight * llr).mm(llr_expander.t())
        
        #the output is the sum of the other LLRs, weighted & masked plus the LLR
        output = .5 * ( expanded_llr + weighted_input )

        #save the output for the backprop
        ctx.save_for_backward(input, input_weight, mask, llr, llr_weight, llr_expander)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        #load tensors from forward method
        input, input_weight, mask, llr, llr_weight, llr_expander = ctx.saved_tensors
        
        grad_input = .5 * grad_output.mm(mask*input_weight)
    
        grad_input_weight = .5 * mask * (grad_output.t().mm(input))
        
        grad_mask = None
        
        #this is WRONG why???
        grad_llr_temp = .5 * grad_output * llr_weight.mm(llr_expander.t())
        grad_llr = grad_llr_temp.mm(llr_expander)
        
        #llr weights are only per node, but there's "duplicate" nodes
        #so we need to add their gradients together to get the correct grad
        grad_llr_weight_temp = .5 * grad_output * llr.mm(llr_expander.t())
        grad_llr_weight = grad_llr_weight_temp.mm(llr_expander)
        
        grad_llr_expander = None
        
        #note that trailing None in return statment are ignored
        return grad_input, grad_input_weight, grad_mask, grad_llr, grad_llr_weight, grad_llr_expander

class BeliefPropagationVC(nn.Module):
    def __init__(self, mask, llr_expander):
        """
        Computes messages at variable nodes in BP
        
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
        
        self.output_dim = mask.shape[0]
        self.input_dim = mask.shape[1]
        
        #setup mask tensor
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float)
        else:
            self.mask = torch.tensor(mask, dtype=torch.float)

        self.mask = nn.Parameter(self.mask, requires_grad=False)
        
        #setup llr tensor
        if isinstance(llr_expander, torch.Tensor):
            self.llr_expander = llr_expander.type(torch.float)
        else:
            self.llr_expander = torch.tensor(llr_expander, dtype=torch.float)
            
        self.llr_expander = nn.Parameter(self.llr_expander, requires_grad=False)

        #setup input_weight tensor
        self.input_weight = nn.Parameter(torch.ones([self.output_dim, self.input_dim], dtype=torch.float))
        
        #setup llr_weight tensor
        self.llr_weight = nn.Parameter(torch.ones([1, self.llr_expander.shape[1]], dtype=torch.float))

        #mask weights
        self.input_weight.data = self.mask.data * self.input_weight.data

    """    
    - input [torch.tensor]:
        - shape: expanded / duplicated variable node size
        - content: messages at parity check node 
    
    - llr [torch.tensor]:
        - shape: (# of variable nodes) x 1
        - content: contains initial llr values received
        
    - note:
        - the above items have been cat'd into a list 0 is input, 1 is llr
    """
    def forward(self, input):
        return BeliefPropagationVC_Function.apply(input[0], self.input_weight, self.mask, input[1], self.llr_weight, self.llr_expander)


#this checks that the gradients are working properly
if __name__ == '__main__':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    bp_vc = BeliefPropagationVC_Function.apply
    
    mask = torch.rand(20,20,dtype=torch.float,requires_grad=False)
    mask = torch.round(mask)
    
    llr_expander = np.zeros((20,5))
    for idx, row in enumerate(llr_expander):
        llr_expander[idx][np.random.randint(5)] = 1
        
    llr_expander = torch.tensor(llr_expander, dtype=torch.float, requires_grad=False)

    input = (
            torch.randn(100,20,dtype=torch.float,requires_grad=True), #input
            torch.randn(20,20,dtype=torch.float,requires_grad=True), #input weight
            mask, #input mask
            torch.randn(1,5,dtype=torch.float,requires_grad=True), #llr
            torch.randn(1,5,dtype=torch.float,requires_grad=True), #llr weight
            llr_expander #llr expander
            )
    
    test = gradcheck(bp_vc, input, eps=1e-6, atol=1e-4)
    print(test)