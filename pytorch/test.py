#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:36:04 2019

@author: jacobwinick
"""

import torch

mask = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 0]])

input = torch.tensor([[1], [2], [3]])

mask_add = -1 * ( mask - 1 )

input_temp1 = input.view(1, -1).expand_as(mask)

input_temp2 = mask * input_temp1 + mask_add

output = torch.prod(input_temp2, dim=1, keepdim=True)

print(output.numpy())

