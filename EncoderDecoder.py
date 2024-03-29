# This is Transformer Code by RAMIT PAHWA
# This is insipired by the implementation by https://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    """
    Encoder Decoder Model 
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Inputs
        src : this is the input sentence 
        tgt : this is the ouput sentence 
        src_mask: this is source mask
        tgt_maks: this is target mask 
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        "Encode Model"
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, encode, src_mask, tgt, tgt_mask):
        "Decoder Model"
        return self.decoder(self.tgt_embed(tgt), encode, src_mask, tgt_mask)

# Questions 
# why doe we need source mask for decoding ?