"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python models/TGN_ST_Sweep.py --data "tgbl-flight" --num_run 1 --seed 1
"""

import math
import timeit
import random

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

# Hang added
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

wandb.login()

sweep_config = {
    'method': 'random'
    }

metric = {
    'name': 'best_val_mrr',
    'goal': 'maximize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd', 'adamw']
        },
    'act': {
        'values': ['relu', 'gelu', 'leaky_relu']},
    'emb_dim': {
        'values': [100, 128, 172, 256]},
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'patience': {
        'values': [5,10,15]},
    'dropout': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
    'num_neighbor': {
        'values': [10, 20, 30]},
    'lr': {
        'distribution': 'q_log_uniform_values',
        'q': 0.00003,
        'min': 0.00001,
        'max': 0.001,
      },
    'bs': {
        'distribution': 'q_log_uniform_values',
        'q': 32,
        'min': 64,
        'max': 512,
      },
    'wd': {
        'distribution': 'q_log_uniform_values',
        'q': 0.0005,
        'min': 0.0001,
        'max': 1,
      },
    't_0': {
        'distribution': 'int_uniform',
        'min': 5,
        'max': 15,
      },
    't_mult': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 5,
      },
    })

sweep_id = wandb.sweep(sweep_config, project="tgn_st_sweep")

def data_preprocess(config, device):
    dataset = PyGLinkPropPredDataset(name=config.data, sub_sampling= config.sub_sampling, method= config.region,
                                    drift= config.drift, root="datasets")
    metric = dataset.eval_metric
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    data = dataset.get_TemporalData()
    data = data.to(device)
    return data, dataset, train_mask, val_mask, test_mask, metric

def build_dataset(config, device):
    data, dataset, train_mask, val_mask, test_mask, metric = data_preprocess(config, device)

    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    train_loader = TemporalDataLoader(train_data, batch_size=config.bs)
    val_loader = TemporalDataLoader(val_data, batch_size=config.bs)
    test_loader = TemporalDataLoader(test_data, batch_size=config.bs)
    return train_data, train_loader, val_loader, test_loader, data, dataset, metric

args, _ = get_args()

def train(config=None, args= args):
    with wandb.init(project= "tgn_st_sweep", config=args):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_data, train_loader, val_loader, test_loader, data, dataset, metric = build_dataset(config, device)
        min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
        criterion = torch.nn.BCEWithLogitsLoss()
        neighbor_loader = LastNeighborLoader(data.num_nodes, size=config.num_neighbor, device=device)
        assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
        evaluator = Evaluator(name=config.data)
        neg_sampler = dataset.negative_sampler
        model = build_network(data, config, device)
        optimizer = build_optimizer(model, config)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=config.t_0, T_mult=config.t_mult)
        save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
        save_model_id = f'{config.model}_{config.data}_{config.seed}_{sweep_id}'
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                        tolerance=config.tolerance, patience=config.patience)
        wandb.watch(model, criterion, log="all", log_freq=10)
        batch_ct = 0
        example_ct = 0
        epoch_lr = []
        dataset.load_val_ns()
        for epoch in tqdm(range(config.num_epoch)):
            avg_loss = train_batch(data, train_data, model, train_loader, criterion, optimizer, assoc, neighbor_loader, 
                                   min_dst_idx, max_dst_idx, device, epoch, batch_ct, example_ct)
            perf_metrics = test(data, model, criterion, val_loader, neg_sampler, assoc, neighbor_loader, 
                                                          device, metric, evaluator, split_mode= "val")
            
            for param_group in optimizer.param_groups:
                epoch_lr.append(param_group['lr'])
            
            wandb.log({"train_loss": avg_loss, "val_mrr": perf_metrics, 'scheduler': epoch_lr[-1]}, step= epoch)
            
            if config.optimizer != 'adam':
                scheduler.step(epoch)
            
            if early_stopper.step_check(perf_metrics, model):
                break

        wandb.log({"best_val_mrr": early_stopper.best_sofar})

def train_batch(data, train_data, model, train_loader, criterion, optimizer, assoc, neighbor_loader, 
                min_dst_idx, max_dst_idx, device, epoch, batch_ct, example_ct):
    
    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()
    
    model['memory'].reset_state() # Reset memory. This is where the error of different device happens if lack of it.
    neighbor_loader.reset_state()
    
    cumu_loss = 0
    for _, batch in tqdm(enumerate(train_loader)):
        batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        
        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])
        
        loss = criterion(torch.cat([pos_out, neg_out]), torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)]))

        example_ct += batch.num_events
        batch_ct += 1

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()

        cumu_loss += float(loss) * batch.num_events

    return cumu_loss / train_data.num_events

def build_network(data, config, device):
    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        config.mem_dim,
        config.time_dim,
        message_module=IdentityMessage(data.msg.size(-1), config.mem_dim, config.time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=config.mem_dim,
        out_channels=config.emb_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
        dropout= config.dropout,
    ).to(device)

    link_pred = LinkPredictor(in_channels=config.emb_dim, act = config.act).to(device)

    model = torch.nn.ModuleDict({'memory': memory,
            'gnn': gnn,
            'link_pred': link_pred})
    
    model['gnn'].reset_parameters()
    model['memory'].reset_state()
    model['link_pred'].reset_parameters()

    return model

def build_optimizer(model, config):
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
            lr=config.lr,
        )
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
            lr=config.lr, weight_decay= config.wd,
        )
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()), 
            lr=config.lr, weight_decay=config.wd)
    return optimizer

    
@torch.no_grad()
def test(data, model, criterion, loader, neg_sampler, assoc, neighbor_loader, 
         device, metric, evaluator, split_mode):

    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = []
    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode) # [200, 20]
        
        src = pos_src.unsqueeze(1).expand(-1, len(neg_batch_list[0]) + 1) # [200, 21]
        dst = torch.cat((pos_dst.unsqueeze(1), torch.tensor(neg_batch_list, device= device)), dim=1)   # [200, 21]

        n_id = torch.cat([pos_src, pos_dst, torch.tensor(neg_batch_list, device= device).flatten()]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        
        for idx in range(src.size(0)):
            y_pred = model['link_pred'](z[assoc[src[idx]]], z[assoc[dst[idx]]])

            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = float(torch.tensor(perf_list).mean())
    return perf_metrics

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    wandb.agent(sweep_id, train, count=200)