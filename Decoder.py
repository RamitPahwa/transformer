import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import utils as utils 
import LayerNorm

class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = utils.clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, ecoder, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, ecoder, src_mask, tgt_mask)
        return self.norm(x)


