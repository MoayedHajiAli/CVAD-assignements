import math
from turtle import hideturtle

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import utils
import numpy as np


class MLP(nn.Module):
    def __init__(self, inp_dim, output_dim, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = inp_dim
        self.mlp = nn.Sequential(nn.Linear(inp_dim, hidden_size),
                                 nn.ReLU(), 
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(), 
                                 nn.Linear(hidden_size, output_dim))
    
    def forward(self, x):
        return self.mlp(x)

class NormMLP(nn.Module):
    def __init__(self, inp_dim, output_dim, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = inp_dim
        self.mlp = nn.Sequential(nn.Linear(inp_dim, hidden_size),
                                 nn.LayerNorm(hidden_size),
                                 nn.ReLU(), 
                                 nn.Linear(hidden_size, output_dim))
    
    def forward(self, x):
        return self.mlp(x)
        
        

class Global_Graph(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_lin = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_lin = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_lin = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)
        # self.future_proj = MLP(hidden_size, hidden_size)


    def forward(self, x, attention_mask=None, mapping=None):
        # 
        # x (B, max(#poly), hidden)
        # print(x.shape, attention_mask.shape)
        # repeat the mask number of heads
        # attention_mask = attention_mask.repeat((self.num_heads, 1, 1))

        # manual implementation
        q, k, v = self.q_lin(x), self.k_lin(x), self.v_lin(x)
        attn = torch.bmm(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # B x #poly x #poly
        # replace 0 with -inf to avoid erros
        attn = attn.masked_fill(attention_mask == 0, -2e9)
        attn = F.softmax(attn, dim=-1) # B, #poly x #poly
        z = torch.bmm(attn, v) # B, #poly, #hidden_size

        # else:
        #     z = self.selfAtten(query=self.q_lin(x), key=self.k_lin(x), value=self.v_lin(x), attn_mask=1-attention_mask)[0] # B x max(#poly) x hidden

        return self.proj(z)


class Sub_Graph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(Sub_Graph, self).__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList([NormMLP(hidden_size, hidden_size//2) for i in range(depth)])
        
    def forward(self, x, lengths):
        """
        Args:
            x (_type_): #polylines x max(#nodes) x hidden_size
            lengths (list): the number of vectors in each polyline
        """
        # print("Hidden states shape:", x.shape, flush=True)
        # print("lengths shape", lengths.shape, flush=True)

        # create a mask out of lengths 
        mask = torch.zeros(x.shape[0], x.shape[1], 1).to(x.device)
        for i, ln in enumerate(lengths):
                mask[i, :ln, :] = 1
        mask = mask.repeat((1, 1, self.hidden_size//2)) == 0

        for i, layer in enumerate(self.layers):
            x = layer(x) # #polys x max(#nodes) x hidden_size//2
            max_x = torch.max(x.masked_fill(mask, -2e9), dim=1, keepdim=True)[0]  #polys x 1 x hidden_size//2
            max_x = max_x.repeat(1, x.size(1), 1) # max(#nodes) x hidden_size//2
            x = torch.cat([max_x, x], dim=-1) # #polys x max(#nodes) x hidden_size

        # take maxpool
        # x = self.mask_for_max(x, lengths)
        x = torch.max(x.masked_fill(mask.repeat((1, 1, 2)), -2e9), dim=1)[0] # #polys x hidden_size
        # l2 normalize for the auxillary loss 
        # x = x / torch.norm(x, dim=-1, keepdim=True)
        return x 
