# coding: utf-8
import sys
import os
sys.path.append(os.getcwd())
# sys.path.append("./home/gyuseonglee/anaconda/envs/tch/lib/")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from torch.nn.utils import clip_grad_norm_ as clip_grad
# from tqdm.auto import tqdm as tq
import warnings
warnings.filterwarnings(action='ignore')

from layers import *

class BidirectionalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BidirectionalEncoder, self).__init__()
        self.name = "Bidirectional Encoder"
        self.encoder_forward = TransformerEncoder(
            num_layers = 4,
            num_heads  = 4,
            dim_model  = input_dim,
            dim_feedforward = hidden_dim,
            dim_q = hidden_dim//4,
            dim_k = hidden_dim//4,
            dropout = 0.11
        )
        self.encoder_backward = TransformerEncoder(
            num_layers = 4,
            num_heads  = 4,
            dim_model  = input_dim,
            dim_feedforward = hidden_dim,
            dim_q = hidden_dim//4,
            dim_k = hidden_dim//4,
            dropout = 0.11
        )
        self.linear = nn.Linear(
            in_features  = input_dim,
            out_features = output_dim
        )

    
    
        
    def forward(self, x):        
        forward_x  = self.encoder_forward(x)
        backward_x = self.encoder_backward(forward_x.flip(dims=[1]))   
        x = self.linear(backward_x)[:, -1, :]
        return x



class OnesidedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(OnesidedEncoder, self).__init__()
        self.name = "One-sided Encoder"

        self.attention_forward = TransformerEncoder(
            num_layers = 4,
            num_heads  = 6,
            dim_model  = input_dim,
            dim_feedforward = hidden_dim,
            dim_q = hidden_dim//2,
            dim_k = hidden_dim//2,
            dropout = 0.11
        )
            
        self.linear = nn.Linear(
            in_features  = input_dim,
            out_features = output_dim
        )
        
    def forward(self, x):        
        forward_x  = self.attention_forward(x)
#         backward_x = self.attention_backward(x.flip(dims=[1]))
#         con = forward_x + backward_x    
        x = self.linear(forward_x)[:, -1, :]
        return x