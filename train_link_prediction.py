
'''Reference: https://github.com/yule-BUAA/DyGLib_TGB/tree/master'''


import logging
import time
import timeit
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tgb.linkproppred.evaluate import Evaluator

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.module import MergeLayer, MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_tgb_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
import wandb

if __name__ == "__main__":
    
    wandb.login()

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name = \
        get_link_prediction_tgb_data(dataset_name=args.dataset_name)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize train negative sampler
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # since the version 2 of tgbl-wiki has included all possible negative destinations for each positive edge, we set batch size to 1 to reduce the memory cost
    if args.dataset_name == "tgbl-wiki":
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=1, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=1, shuffle=False)
    else:
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        with wandb.init(project="test_onnx", config=args):
            set_random_seed(seed=run)

            args.seed = run
            args.save_model_name = f'{args.model_name}_seed{args.seed}'

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
            fh.setLevel(logging.DEBUG)
            # create console handler with a higher log level
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            # create formatter and add it to the handlers
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            # add the handlers to logger
            logger.addHandler(fh)
            logger.addHandler(ch)

            run_start_time = time.time()
            logger.info(f"********** Run {run + 1} starts. **********")

            logger.info(f'configuration is {args}')

            # create model
            if args.model_name == 'TGAT':
                dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                        dropout=args.dropout, device=args.device)
            elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                    compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
                dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                            dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                            dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
            elif args.model_name == 'CAWN':
                dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, output_dim=args.output_dim, walk_length=args.walk_length,
                                        num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
            elif args.model_name == 'TCL':
                dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                    num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
            elif args.model_name == 'GraphMixer':
                dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers,
                                            dropout=args.dropout, device=args.device)
            elif args.model_name == 'DyGFormer':
                dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, output_dim=args.output_dim,
                                            patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                            max_input_sequence_length=args.max_input_sequence_length, device=args.device)
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            link_predictor = MergeLayer(input_dim1=args.output_dim, input_dim2=args.output_dim, hidden_dim=args.output_dim, output_dim=1)
            model = nn.Sequential(dynamic_backbone, link_predictor)
            logger.info(f'model -> {model}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

            model = convert_to_gpu(model, device=args.device)

            save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
            shutil.rmtree(save_model_folder, ignore_errors=True)
            os.makedirs(save_model_folder, exist_ok=True)

            early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                        save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

            loss_func = nn.BCEWithLogitsLoss()
            evaluator = Evaluator(name=args.dataset_name)
            
            wandb.watch(model, loss_func, log="all", log_freq=10)

            for epoch in range(args.num_epochs):
                
                train_start_time = time.time()

                model.train()
                if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                    # training, only use training graph
                    model[0].set_neighbor_sampler(train_neighbor_sampler)
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reinitialize memory of memory-based models at the start of each epoch
                    model[0].memory_bank.__init_memory_bank__()

                # store train losses and metrics
                train_losses, train_metrics = [], []
                train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
                for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                    train_data_indices = train_data_indices.numpy()
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                        train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                    _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                    batch_neg_src_node_ids = batch_src_node_ids

                    # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                    # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                    if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors)
                    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # note that negative nodes do not change the memories while the positive nodes change the memories,
                        # we need to first compute the embeddings of negative nodes for memory-based models
                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=None,
                                                                            edges_are_positive=False,
                                                                            num_neighbors=args.num_neighbors)

                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            edge_ids=batch_edge_ids,
                                                                            edges_are_positive=True,
                                                                            num_neighbors=args.num_neighbors)
                    elif args.model_name in ['GraphMixer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times,
                                                                            num_neighbors=args.num_neighbors,
                                                                            time_gap=args.time_gap)
                    elif args.model_name in ['DyGFormer']:
                        # get temporal embedding of source and destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_src_node_embeddings, batch_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                            dst_node_ids=batch_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)

                        # get temporal embedding of negative source and negative destination nodes
                        # two Tensors, with shape (batch_size, output_dim)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                            dst_node_ids=batch_neg_dst_node_ids,
                                                                            node_interact_times=batch_node_interact_times)
                    else:
                        raise ValueError(f"Wrong value for model_name {args.model_name}!")
                    # get positive and negative probabilities, shape (batch_size, )
                    positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                    predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                    labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                    loss = loss_func(input=predicts, target=labels)

                    train_losses.append(loss.item())

                    train_metrics.append({'average_precision': average_precision_score(y_true=labels.cpu().numpy(), y_score=predicts.cpu().detach().numpy()),
                                        'roc_auc': roc_auc_score(y_true=labels.cpu().numpy(), y_score=predicts.cpu().detach().numpy())})

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')
                    
                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                        model[0].memory_bank.detach_memory_bank()

                train_end_time = time.time()
                
                val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                            model=model,
                                                            neighbor_sampler=full_neighbor_sampler,
                                                            evaluate_idx_data_loader=val_idx_data_loader,
                                                            evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                            evaluate_data=val_data,
                                                            eval_stage='val',
                                                            eval_metric_name=eval_metric_name,
                                                            evaluator=evaluator,
                                                            num_neighbors=args.num_neighbors,
                                                            time_gap=args.time_gap)
                val_end_time = time.time()
                
                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                    val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

                logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
                for metric_name in train_metrics[0].keys():
                    logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
                for metric_name in val_metrics[0].keys():
                    logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

                wandb.log({"train_time_per_epoch": train_end_time - train_start_time,
                            "val_time_per_epoch": val_end_time - train_end_time,
                            "train_loss": np.mean(train_losses),
                            "val_mrr": np.mean([val_metric["mrr"] for val_metric in val_metrics]),})

                # perform testing once after test_interval_epochs
                if (epoch + 1) % args.test_interval_epochs == 0:
                    test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                model=model,
                                                                neighbor_sampler=full_neighbor_sampler,
                                                                evaluate_idx_data_loader=test_idx_data_loader,
                                                                evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                                evaluate_data=test_data,
                                                                eval_stage='test',
                                                                eval_metric_name=eval_metric_name,
                                                                evaluator=evaluator,
                                                                num_neighbors=args.num_neighbors,
                                                                time_gap=args.time_gap)

                    if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                        # reload validation memory bank for testing nodes or saving models
                        # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                        model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

                    for metric_name in test_metrics[0].keys():
                        logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')

                # select the best model based on all the validate metrics
                val_metric_indicator = []
                for metric_name in val_metrics[0].keys():
                    val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
                
                early_stop = early_stopping.step(val_metric_indicator, model)

                if early_stop:
                    break
            
            train_val_time = time.time() - run_start_time
            
            # load the best model
            early_stopping.load_checkpoint(model)

            # evaluate the best model
            logger.info(f'get final performance on dataset {args.dataset_name}...')
            

            # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                            model=model,
                                                            neighbor_sampler=full_neighbor_sampler,
                                                            evaluate_idx_data_loader=val_idx_data_loader,
                                                            evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                            evaluate_data=val_data,
                                                            eval_stage='val',
                                                            eval_metric_name=eval_metric_name,
                                                            evaluator=evaluator,
                                                            num_neighbors=args.num_neighbors,
                                                            time_gap=args.time_gap)

            start_test_time = time.time()
            test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                        model=model,
                                                        neighbor_sampler=full_neighbor_sampler,
                                                        evaluate_idx_data_loader=test_idx_data_loader,
                                                        evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                        evaluate_data=test_data,
                                                        eval_stage='test',
                                                        eval_metric_name=eval_metric_name,
                                                        evaluator=evaluator,
                                                        num_neighbors=args.num_neighbors,
                                                        time_gap=args.time_gap)

            # store the evaluation metrics at the current run
            val_metric_dict, test_metric_dict = {}, {}

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                    logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                    val_metric_dict[metric_name] = average_val_metric
                    # wandb.log({f"test_{metric_name}": average_val_metric})

            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric
                # wandb.log({f"test_{metric_name}": average_test_metric})


            single_run_time = time.time() - run_start_time
            logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
            
            wandb.log({"train_val_total_time": single_run_time,
                       "test_time":time.time() - start_test_time,
                       "train_val_time": train_val_time,
                       "test_mrr": test_metric_dict['mrr']})

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                val_metric_all_runs.append(val_metric_dict)
            test_metric_all_runs.append(test_metric_dict)

            # avoid the overlap of logs
            if run < args.num_runs - 1:
                logger.removeHandler(fh)
                logger.removeHandler(ch)

            # save model result
            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                result_json = {
                    "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                    "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
                }
            else:
                result_json = {
                    "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict}
                }
            result_json = json.dumps(result_json, indent=4)

            save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
            os.makedirs(save_result_folder, exist_ok=True)
            save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

            with open(save_result_path, 'w') as file:
                file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
