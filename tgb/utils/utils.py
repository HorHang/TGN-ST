import numpy as np
import random
import os
import pickle
from typing import Any
import sys
import argparse
import json
import io
import torch
import torch.nn as nn
from torch import Tensor
from typing import List
from einops import rearrange

# import torch
def save_pkl(obj: Any, fname: str) -> None:
    r"""
    save a python object as a pickle file
    """
    with open(fname, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fname: str) -> Any:
    r"""
    load a python object from a pickle file
    """
    with open(fname, "rb") as handle:
        return pickle.load(handle)


def set_random_seed(seed: int):
    r"""
    setting random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_args():
    parser = argparse.ArgumentParser('*** TGB ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-flight')
    parser.add_argument('-m', '--model', type=str, help='Model name', default='TGN')
    parser.add_argument('--model_name', type=str, help='Model name', default='TGN')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--wd', type=float, help='Weight decay', default=1e-5) # Hang added
    parser.add_argument('--t_0', type=int, help='restart iteration for first restart', default=5)
    parser.add_argument('--t_mult', type=int, help='restart time multiplier', default= 1)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=100)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=5)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('--sub_sampling', type= bool, help='graph subsampling indicator', default=False)
    parser.add_argument('--region', type=str, help='training sumpling region', default="NA")
    parser.add_argument('--drift', type=bool, help='test region drift', default=False)
    parser.add_argument('--num_workers', type=int, help='Number of workers', default=27)
    parser.add_argument('--num_neighbor', type=int, default=10, help='Number of neighbors to consider')
    parser.add_argument('--optimizer', type=str, help='Optimizer', default='adam')
    parser.add_argument('--resume', type=bool, help='Resume training', default=False)
    parser.add_argument('--start_epoch', type=int, help='Start epoch', default=1)
    parser.add_argument('--start_run', type= int, help='Start run', default=0)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)
    parser.add_argument('--act', type=str, help= 'activation function', default= 'relu')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv

def save_results(new_results: dict, filename: str):
    r"""
    save (new) results into a json file
    :param: new_results (dictionary): a dictionary of new results to be saved
    :filename: the name of the file to save the (new) results
    """
    if os.path.isfile(filename):
        # append to the file
        with open(filename, 'r+') as json_file:
            file_data = json.load(json_file)
            # convert file_data to list if not
            if type(file_data) is dict:
                file_data = [file_data]
            file_data.append(new_results)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
    else:
        # dump the results
        with open(filename, 'w') as json_file:
            json.dump(new_results, json_file, indent=4)

def enlarge_as(src : Tensor, other : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()

class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True
    ):
        self._padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self._padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, inp : Tensor) -> Tensor:
        # Handle the case where input has only two dimensions
        # we expect them to have semantics (batch, channels),
        # so we add the missing dimension manually
        if inp.dim() == 2: inp = rearrange(inp, 'b i -> b 1 i')
        
        result = super(CausalConv1d, self).forward(inp)
        if self._padding != 0: return result[..., :-self._padding]
        return result
    
class BlockLinear(nn.Module):
    def __init__(
        self,
        block_dims : List[int | List[int]],
        bias : bool = False,
    ):
        super(BlockLinear, self).__init__()
        
        self._blocks = nn.ParameterList([
            nn.Parameter(torch.randn(size, requires_grad=True))
            for size in block_dims
        ])
        
        self._bias = nn.Parameter(torch.zeros(sum(block_dims))) if bias else None
        
    def forward(self, inp : Tensor) -> Tensor:
        # Assemble the blocks into a block-diagonal matrix
        full = torch.block_diag(*self._blocks)
        
        out = torch.matmul(full, inp)
        
        if self._bias is not None:
            out = out + self._bias
        
        return out