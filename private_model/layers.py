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

'''
    1. Residual blocks
'''
class Residual_original(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return tensors[0] + self.dropout(self.norm(self.sublayer(*tensors)))
    

'''
    2. Feed forward & positional encodings
'''
def feed_forward(dim_input, dim_feedforward) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.GELU(), # activation function of original model was relu 
        nn.Linear(dim_feedforward, dim_input),
)


def position_encoding(seq_len, dim_model, 
                      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")) -> torch.Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def scaled_dot_product_attention(query:torch.Tensor, key:torch.Tensor, value:torch.Tensor)->torch.Tensor:  # value -> value
    # temp : Q @ K.T   -> (N.T.D) @ (N,D,T)  (batch size만 그대로 두고, D,T 와 T, D를 )
    # Q @ K.T -> (N, T, T)
    QKT     = query.bmm(key.transpose(1, 2))  # bmm : batch matrix multiplication (X, O, O)-> O에 해당되는 dim에 대해서만 matmul 진행
    root_dk = query.size(-1)**0.5            # squared root of D
    softmax = f.softmax( QKT / root_dk, dim= -1 ) # softmax for "T of Key", not for "T of Query", so dim = -1 is right
                                                  # dim = -2 로 맞추면 Key 에 대한 쿼리 결과(세로축)으로 1을 합산하는 꼴임


    return softmax.bmm(value) # (N,T,T)@(N,T,D) -> (N, T, D)


'''
    3. Attention
'''

class AttentionHead(nn.Module): # X = (N, D)  / Wq, Wk, Wv : (D, Q) or (D, K) / 일반적으로 K, Q는 같은 dimension 사용
    def __init__(self, input_dim:int, query_dim:int, key_dim:int):
        super().__init__()
        
        self.q_linear = nn.Linear(input_dim, query_dim) # generate Q 
        self.k_linear = nn.Linear(input_dim, key_dim) # generate K
        self.v_linaer = nn.Linear(input_dim, key_dim) # generate V


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        # 인풋으로 들어온 query, key, value 텐서에 대해 linear forward를 진행하고
        # 그 과정을 통해 만들어진 Q, K, V를 그대로 scaled_dot_product_attention 에 forward 시킴        
        return scaled_dot_product_attention(
            self.q_linear(query), # query @ q_linear : (Xq : N, D) @ (Wq : D, Q) -> (N, Q)
            self.k_linear(key),   # key   @ k_linear : (Xk : N, D) @ (Wk : D, K) -> (N, K)
            self.v_linaer(value)) # value @ v_linear : (Xv : N, D) @ (Wv : D, K) -> (N, K)
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, input_dim: int, query_dim: int, key_dim: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(input_dim, query_dim, key_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * key_dim, input_dim)  
        # num_heads 만큼 horizontally concat 되므로, 
        # multiheadAttention의 forward 결과 나오는 concated V의 dimension 은 numm_heads배 늘어난다. 
        # 즉, (N, T, K) * num_heads -> (N, T, num_heads * K)


    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return self.linear(
            torch.cat([ head(query, key, value) for head in self.heads], dim=-1)
        )

'''
    4. Transformer encoder
'''

class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_heads, dim_model, dim_feedforward, dim_q, dim_k, dropout):
        super().__init__()
        # dim_q = max(dim_model // num_heads, 1)
        # dim_k = dim_q
        self.attention = MultiHeadAttention(
                num_heads = num_heads,
                input_dim = dim_model,
                query_dim = dim_q, 
                key_dim   = dim_k
        )
        self.residual_AT = Residual(
            sublayer  = self.attention,
            dimension = dim_model,
            dropout   = dropout
        )
        self.feed_forward = feed_forward(
            dim_input       = dim_model,
            dim_feedforward = dim_feedforward
        )
        self.residual_FF = Residual(
            sublayer = self.feed_forward,
            dimension = dim_model,
            dropout   = dropout,
        )

    ''' source shape : (N, T, D)'''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_AT(x, x, x)
        return self.residual_FF(x)


class TransformerEncoder(nn.Module):
    def __init__(self, 
                 num_layers, 
                 num_heads, 
                 dim_model, 
                 dim_feedforward, 
                 dim_q,
                 dim_k,
                 dropout):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    num_heads = num_heads, 
                    dim_model = dim_model, 
                    dim_feedforward = dim_feedforward, 
                    dim_q = dim_q,
                    dim_k = dim_k,
                    dropout = dropout
                )
                for _ in range(num_layers)
            ]
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape
        if(x.ndim==2):
            x = x.reshape(1, x.size(0), x.size(1))
        N, T, D = x.shape
        
        # positional encoding
        x += position_encoding(T, D)
        for layer in self.layers:
            x = layer(x)

        return x