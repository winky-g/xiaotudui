#-*- coding: utf-8 -*-
import torch
from torch import nn

class Shiyan(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input +1
        return output

shiyan = Shiyan()
x = torch.tensor(1.0)
output = shiyan(x)
print(output)