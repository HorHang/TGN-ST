"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch.nn import Linear, Parameter
import numpy as np


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()

class TimeEncoderMixer(torch.nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, out_channels):
        super(TimeEncoderMixer, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.lin.weight = Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.out_channels, dtype=np.float32))).reshape(self.out_channels, -1))
        self.lin.weight.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        return self.lin(t.view((-1, 1))).cos()