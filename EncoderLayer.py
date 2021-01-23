import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import utils as utils 
import LayerNorm
import SubLayer

class EncoderLayer(nn.Module):

    def __init__(self, attention, feed_forward, dropout, size):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.feed_forward = feed_forward
        self.sublayers = utils.clones(Sublayer(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x : self.attention(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
