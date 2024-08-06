"""
Dynamic Link Prediction with a TGN model with Early Stopping
Reference: 
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

command for an example run:
    python models/TGN.py --data "tgbl-flight" --num_run 1 --seed 1
"""

import math
import timeit

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
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb

wandb.login()
# ==========
# ========== Define helper function...
# ==========

def train(epoch= 1):
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        # mw.begin()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg #dim 200; 200; 200; [200,16]
        
        # TODO implement neighbor co-occurrence
        
        
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
            data.t[e_id].to(device), # get time of corresponding edges
            data.msg[e_id].to(device), # get msg of corresponding edges
        )
        
        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(torch.cat([pos_out, neg_out]), torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)]))
        # loss = criterion(pos_out, torch.ones_like(pos_out))
        # loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)
        
        # insert only portion of new encountered interactions
        # probability = np.arange(0, len(src))/len(src)
        # prob = torch.softmax(torch.tensor(probability), dim=0)
        # indices = np.random.choice(len(src), int(len(src) * 0.9), p= prob, replace=False)
        # indices = np.sort(indices)
        # neighbor_loader.insert(src[indices], pos_dst[indices])

        # mw.zero_grad()
        # loss.backward(create_graph= True)
        loss.backward()
        optimizer.step()
        # scheduler.step(epoch + i/iters)
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, neg_sampler, split_mode):
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
    total_val_loss = 0
    val_list = []
    auc_roc_list = []
    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode) # shape (200, 20)

        val_loss = []
        ap_list = []
        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
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

            y_pred = model['link_pred'](z[assoc[src]], z[assoc[dst]]) # shape [21, 1]
            
            # compute validation loss; TODO check if number of edge validation is the same as training
            labels = torch.zeros_like(y_pred)
            labels[0, :] = 1
            loss = criterion(y_pred, labels)
            val_loss.append(loss.item()) # return 21 losses of 1 pos + 20 neg edges
            
            # compute average precision
            ap = average_precision_score(y_true= labels.cpu().numpy(), y_score= y_pred.cpu().numpy())
            ap_list.append(ap)
            
            auc_roc_list.append(roc_auc_score(y_true= labels.cpu().numpy(), y_score= y_pred.cpu().numpy()))

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)
        
        val_list.append(torch.tensor(val_loss).mean()) # list of all batch

    perf_metrics = float(torch.tensor(perf_list).mean())
    total_val_loss = float(torch.tensor(val_list).mean()) # total loss per epoch
    mean_aps = float(torch.tensor(ap_list).mean())
    auc_roc = float(torch.tensor(auc_roc_list).mean())
    
    return perf_metrics, total_val_loss, mean_aps, auc_roc

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = "tgbl-flight"
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 20
WEIGHT_DECAY = args.wd
MODEL_NAME = 'TGN'
T_0 = args.t_0
T_MULT = args.t_mult
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data loading. method must be 'random', 'AF', 'AN', 'AS', 'EU', 'NA', 'OC', 'SA'
dataset = PyGLinkPropPredDataset(name=DATA, sub_sampling= False, method= 'AS', drift= True, root="datasets")
# dataset_test = PyGLinkPropPredDataset(name=DATA, sub_sampling= True, method= 'random', root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)

model = torch.nn.ModuleDict({'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred})

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)

# optimizer = torch.optim.AdamW(set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
# scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_MULT)

# optim = gdtuo.Adam(optimizer = gdtuo.SGD(LR))
# mw = gdtuo.ModuleWrapper(torch.nn.ModuleList([memory, gnn, link_pred]), optimizer=optim)
# mw.initialize()

criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")
# print("training & validation @asia, test @oceania")
evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'


for run_idx in range(NUM_RUNS):
    with wandb.init(project='mamba_module_test', config=args):
        
        wandb.watch(model, criterion, log="all", log_freq=10)
        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.info(f'model -> {torch.nn.Sequential(memory, gnn, link_pred)}')
        
        print('-------------------------------------------------------------------------------')
        print(f"INFO: >>>>> Run: {run_idx} <<<<<")
        
        start_run = timeit.default_timer()

        # set the seed for deterministic results...
        torch.manual_seed(run_idx + SEED)
        set_random_seed(run_idx + SEED)

        # define an early stopper
        save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
        save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
        early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                        tolerance=TOLERANCE, patience=PATIENCE)

        # ==================================================== Train & Validation
        # loading the validation negative samples
        dataset.load_val_ns()

        val_per_l = []
        train_loss_l, val_loss_l, aps_l, auc_roc_l = [], [], [], [] # Hang added
        train_times_l, val_times_l = [], []
        free_mem_l, total_mem_l, used_mem_l = [], [], []
        start_train_val = timeit.default_timer()
        iters = len(train_loader) ### adamW
        for epoch in range(1, NUM_EPOCH + 1):
            # training
            start_epoch_train = timeit.default_timer()
            loss = train(epoch= epoch)
            end_epoch_train = timeit.default_timer()
            print(
                f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {end_epoch_train - start_epoch_train: .4f}"
            )
            # checking GPU memory usage
            free_mem, used_mem, total_mem = 0, 0, 0
            if torch.cuda.is_available():
                print("DEBUG: device: {}".format(torch.cuda.get_device_name(0)))
                free_mem, total_mem = torch.cuda.mem_get_info()
                used_mem = total_mem - free_mem
                print("------------Epoch {}: GPU memory usage-----------".format(epoch))
                print("Free memory: {}".format(free_mem))
                print("Total available memory: {}".format(total_mem))
                print("Used memory: {}".format(used_mem))
                print("--------------------------------------------")
            
            train_times_l.append(end_epoch_train - start_epoch_train)
            free_mem_l.append(float((free_mem*1.0)/2**30))  # in GB
            used_mem_l.append(float((used_mem*1.0)/2**30))  # in GB
            total_mem_l.append(float((total_mem*1.0)/2**30))  # in GB

            # validation
            start_val = timeit.default_timer()
            perf_metric_val, perf_loss_val, mean_aps, auc_roc = test(val_loader, neg_sampler, split_mode="val")
            end_val = timeit.default_timer()
            print(f"\tValidation loss: {perf_loss_val: .4f}")
            print(f"\tValidation aps: {mean_aps: .4f}")
            print(f"\tValidation {metric}: {perf_metric_val: .4f}")
            print(f"\tValidation auc-roc: {auc_roc: .4f}")
            print(f"\tValidation: Elapsed time (s): {end_val - start_val: .4f}")
            val_per_l.append(perf_metric_val)
            val_times_l.append(end_val - start_val)

            train_loss_l.append(loss)
            val_loss_l.append(perf_loss_val)
            aps_l.append(mean_aps)
            auc_roc_l.append(auc_roc)
            
            wandb.log({"train_loss": loss, f"val_{metric}": perf_metric_val, "val_loss": perf_loss_val,
                       "val_auc_roc": auc_roc, "val_aps": mean_aps, "train_time_per_epoch": end_epoch_train - start_epoch_train,
                       "val_time_per_epoch": end_val - start_val,})
            
            # check for early stopping
            if early_stopper.step_check(perf_metric_val, model):
                break

        train_val_time = timeit.default_timer() - start_train_val
        print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

        # ==================================================== Test
        # first, load the best model
        early_stopper.load_checkpoint(model)

        # loading the test negative samples
        dataset.load_test_ns()

        # final testing
        start_test = timeit.default_timer()
        perf_metric_test, test_loss, test_aps, auc_roc = test(test_loader, neg_sampler, split_mode="test")
        test_time = timeit.default_timer() - start_test

        wandb.log({f"test_{metric}": perf_metric_test, "test_loss": test_loss, "test_aps": test_aps,
                   "test_auc_roc": auc_roc, "test_time": test_time, "train_val_time": train_val_time, 
                   "train_val_total_time": train_val_time + test_time,})
        print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
        print(f"\tTest: {metric}: {perf_metric_test: .4f}")
        print(f"\tTest: loss: {test_loss: .4f}")
        print(f"\tTest: aps: {test_aps: .4f}")
        print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

        # save_results({'data': DATA,
        #             'model': MODEL_NAME,
        #             'run': run_idx,
        #             'seed': SEED,
        #             'train_times': train_times_l,
        #             'free_mem': free_mem_l,
        #             'total_mem': total_mem_l,
        #             'used_mem': used_mem_l,
        #             'max_used_mem': max(used_mem_l),
        #             'val_times': val_times_l,
        #             'train_loss': train_loss_l,
        #             'aps': aps_l,
        #             f'val {metric}': val_per_l,
        #             f'test {metric}': perf_metric_test,
        #             'test_loss': test_loss,
        #             'test_aps': test_aps,
        #             'test_time': test_time,
        #             'train_val_total_time': np.sum(np.array(train_times_l)) + np.sum(np.array(val_times_l)),
        #             }, 
        # results_filename)

    # obtain the current time
    # now = datetime.now()
    # current_time = now.strftime("%D/%H_00")
    # current_time = current_time.replace("/", "_")

    # print('Plotting training & validation metric ...')

    # loss_plt_path = f'{results_path}/{MODEL_NAME}_{DATA}_{run_idx}_train_val_loss_{current_time}.pdf'
    # epoch_plt_path = f'{results_path}/{MODEL_NAME}_{DATA}_{run_idx}_epoch_time_{current_time}.pdf'
    # mrr_plt_path = f'{results_path}/{MODEL_NAME}_{DATA}_{run_idx}_mrr_{current_time}.pdf'
    # aps_plt_path = f'{results_path}/{MODEL_NAME}_{DATA}_{run_idx}_aps_{current_time}.pdf'

    # Plot train & Val loss
    # fLoss = plt.figure()
    # axLoss = fLoss.add_subplot(111)
    # plt.plot(train_loss_l, label='train_loss')
    # plt.plot(val_loss_l, label='val_loss')
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.title(f'train vs. val Loss')
    # plt.legend()
    # plt.savefig(loss_plt_path)

    # Plot train/val per epoch
    # fEpoch = plt.figure()
    # axEpoch = fEpoch.add_subplot(111)
    # plt.plot(train_times_l, label= "train_per_epoch")
    # plt.plot(val_times_l, label= "val_per_epoch")
    # plt.xlabel("epoch")
    # plt.ylabel("second")
    # plt.title(f'train/val per epoch (s), mean= {np.mean(train_times_l)} / {np.mean(val_times_l): .4f}')
    # plt.legend()
    # plt.savefig(epoch_plt_path)

    # Plot MRR
    # fMRR = plt.figure()
    # axMRR = fMRR.add_subplot(111)
    # plt.plot(val_per_l, label = 'val_mrr')
    # plt.plot(train_loss_l, label='train_loss')
    # plt.xlabel("epoch")
    # plt.ylabel("val mrr value")
    # plt.title("train_loss vs. val mrr")
    # plt.legend()
    # plt.savefig(mrr_plt_path)

    # TODO Plot average precision (Primary for most of previous research)
    # fAP = plt.figure()
    # axAP = fAP.add_subplot(111)
    # plt.plot(aps_l, label= 'aps')
    # plt.xlabel('epoch')
    # plt.ylabel('test aps')
    # plt.title('mean average precision')
    # plt.savefig(aps_plt_path)

    # TODO plot auc-roc (Secondary for most of previous research paper)

    # print(f'Saved plotting result at {results_path}')

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")


