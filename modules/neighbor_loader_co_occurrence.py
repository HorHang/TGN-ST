"""
Neighbor Loader

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import numpy as np


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.device = device
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        # self._assoc is used to keep track of a unique identifier for each node encountered so far

        # DyGFormer (bs, len, dim) ==> (unique_nodes, dim)

        self.reset_state()

    def __call__(self, n_id: Tensor, src: Tensor, dst: Tensor) -> Tuple[Tensor, Tensor, Tensor]: # n_id is the way to get back the src & dst nodes
        # retrive the request neighbor & e_id of node n_id
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask] # (num_edges), (num_edges), (num_edges) not unique edges

        # Relabel node indices. 
        # Handles potential invalid neighbors and relabels node indices for efficient processing.
        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device) # ensure unique & consecutive node id including neighbor nodes
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        src_node_appearances, dst_node_appearances = self.co_occurrence(neighbors, nodes) # (num_edges, 2), (num_edges, 2)
        batch_src_node_appearances, batch_dst_node_appearances = self.co_occurrence(src, dst)
        return n_id, torch.stack([neighbors, nodes]), e_id, src_node_appearances, dst_node_appearances, batch_src_node_appearances, batch_dst_node_appearances
    
    def co_occurrence(self, src: Tensor, dst: Tensor): # count neighbor co_occurrence of nodes within each batch
        
        src_unique_keys, src_inverse_indices, src_counts = np.unique(src.cpu(), return_inverse=True, return_counts=True)
        src_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float().to(self.device)
        src_mapping_dict = dict(zip(src_unique_keys, src_counts))

        dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst.cpu(), return_inverse=True, return_counts=True)
        dst_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float().to(self.device)
        dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))
        
        src_neighbor_counts_in_dst = src.cpu().apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)
        dst_neighbor_counts_in_src = dst.cpu().apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float().to(self.device)

        src_node_appearances = torch.stack([src_neighbor_counts_in_src, src_neighbor_counts_in_dst], dim=1)
        dst_node_appearances = torch.stack([dst_neighbor_counts_in_src, dst_neighbor_counts_in_dst], dim=1)
        
        return src_node_appearances, dst_node_appearances

    def insert(self, src: Tensor, dst: Tensor):
        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
        )

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)