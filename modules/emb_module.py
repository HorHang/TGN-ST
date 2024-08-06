"""
GNN-based modules used in the architecture of MP-TG models

"""

import math
from torch_geometric.nn import TransformerConv
import torch

class GraphAttentionEmbedding(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc, co_dim=4, dropout=0.1):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels# + co_dim
        self.conv = TransformerConv(
            in_channels, out_channels // 2, heads=2, dropout=dropout, edge_dim=edge_dim
        )
        self.reset_parameters()

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t # retrieve the last update time of the source node and subtract the time of the edge to get the relative time
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([msg, rel_t_enc], dim=-1)
        return self.conv(x, edge_index, edge_attr) # x shall be h_k-1_u(t); edge_attr a_k_u(t)
    
    def reset_parameters(self):
        self.conv.reset_parameters()

# class GraphAttentionEmbedding(torch.nn.Module):
#     """
#     Reference:
#     - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
#     """

#     def __init__(self, in_channels, out_channels, msg_dim, time_enc, co_dim=4):
#         super().__init__()
#         self.time_enc = time_enc
#         edge_dim = msg_dim + time_enc.out_channels + co_dim
#         self.conv = TransformerConv(
#             in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim
#         )

#     def forward(self, x, last_update, edge_index, co_occur, t, msg):
#         rel_t = last_update[edge_index[0]] - t # retrieve the last update time of the source node and subtract the time of the edge to get the relative time
#         rel_t_enc = self.time_enc(rel_t.to(x.dtype))
#         edge_attr = torch.cat([msg, co_occur, rel_t_enc], dim=-1)
#         # print("feature matrix x:", x.shape, x)
#         # print("edge_index:", edge_index.shape, edge_index)
#         return self.conv(x, edge_index, edge_attr)

class TimeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class NormalLinear(torch.nn.Linear):
            # From TGN code: From JODIE code
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.out_channels)

    def forward(self, x, last_update, t):
        rel_t = last_update - t
        embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

        return embeddings
