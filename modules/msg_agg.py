"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch_geometric.utils import scatter
from torch_scatter import scatter_max
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.aggr import SetTransformerAggregation, LSTMAggregation, DeepSetsAggregation
from torch.nn import Linear

class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int): # [num_edge, msg_dim (320)], [num_edge], [num_edge], n_id.size(0)
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size) # index of the maximum value in t
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out # [unique_nodes, msg_dim (320)]

class Aggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.msg_aggr = SetTransformerAggregation(320, heads=2)
        
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size) # index of the maximum value in t
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        val, ind = index.sort()
        if ind.size(0) == 1:
            ret = self.msg_aggr(msg[ind], val)
            out[mask] = ret
        return out

class SetAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.set_aggr = SetTransformerAggregation(320, heads=2)
        
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size) # index of the maximum value in t
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        val, indices = index.sort()
        if indices.size(0) == 1:
            ret = self.set_aggr(msg[indices], val)
            out[mask] = ret
        return out

class LSTMAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.set_aggr = LSTMAggregation(320, 320)
        
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size) # index of the maximum value in t
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        val, indices = index.sort()
        if indices.size(0) == 1:
            ret = self.set_aggr(msg[indices], val)
            out[mask] = ret
        return out

class DeepSetsAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.set_aggr = DeepSetsAggregation()
        
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size) # index of the maximum value in t
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        val, indices = index.sort()
        if indices.size(0) == 1:
            ret = self.set_aggr(msg[indices], val)
            out[mask] = ret
        return out

class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")
