import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import utils as utils 
import LayerNorm
import SubLayer

class DecoderLayer(nn.Module):

    def __init__(self, attention, feed_forward, size, dropout):
        super(DecoderLayer, self).__init__()
        self.attention = attention 
        self.feed_forward = feed_forward
        self.sublayers = utils.clones(SubLayer(size, dropout), 3)
        self.size = size

    def forward(self, x, encode, src_mask, tgt_mask):
        m = encode
        x = self.sublayers[0](x, lambda x: self.attention(x, x, x, tgt_mask))
        x = self.sublayes[1](x, lambda x: self.attention(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)

