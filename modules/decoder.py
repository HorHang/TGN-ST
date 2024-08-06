"""
Decoder modules for dynamic link prediction

"""

import torch
from torch.nn import Linear
import torch.nn.functional as F


class LinkPredictor(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels, act='relu'):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)
        self.act = act
        self.reset_parameters()

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        if self.act == 'relu':
            h = h.relu() # dim [200,100]
        elif self.act == 'gelu':
            h = F.gelu(h) # dim [200,100]
        elif self.act == 'leakyrelu':
            h = F.leaky_relu(h) # dim [200,100]
        return self.lin_final(h).sigmoid()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_final.reset_parameters()

class NodePredictor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node = Linear(in_dim, in_dim)
        self.out = Linear(in_dim, out_dim)

    def forward(self, node_embed):
        h = self.lin_node(node_embed)
        h = h.relu()
        h = self.out(h)
        # h = F.log_softmax(h, dim=-1)
        return h
