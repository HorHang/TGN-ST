"""
Message Function Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import torch
from torch import Tensor


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim # + 4

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor): 
        # z_src.shape torch.Size([0, 100]) z_dst.shape torch.Size([0, 100]) raw_msg.shape torch.Size([0, 20]) t_enc.shape torch.Size([0, 100])
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)

# class IdentityMessage(torch.nn.Module):
#     def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
#         super().__init__()
#         self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim + 4

#     def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor): 
#         # z_src.shape torch.Size([0, 100]) z_dst.shape torch.Size([0, 100]) raw_msg.shape torch.Size([0, 20]) t_enc.shape torch.Size([0, 100])
#         return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
