import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import utils as utils 
import LayerNorm

class Encoder(nn.Module):
    "Encoder Class"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = utils.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        inputs
        x    : source embedding 
        mask : source mask
        """
        for layer in self.layers:
            x =  layer(x, mask)
        return self.norm(x)
