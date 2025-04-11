import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# Exact HuggingFace NewGELUActivation
class NewGELUActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x.pow(3))))


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    
    def __repr__(self):
        return f"Conv1D(nf={self.nf}, nx={self.weight.size(0)})"
    
class GPT2Config:
    def __init__(self, 
                 vocab_size=50257,
                 n_ctx=1024,
                 n_embd=768,
                 n_layer=12,
                 n_head=12):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return f"GPT2Config {self.to_dict()}"