
'''Reference: https://github.com/yule-BUAA/DyGLib_TGB/tree/master'''

import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn
from tgb.nodeproppred.evaluate import Evaluator

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.module import MLPClassifier
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler
from evaluate_models_utils import evaluate_model_node_classification, evaluate_parameter_free_node_classification
from utils.DataLoader import get_idx_data_loader, get_node_classification_tgb_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_node_classification_args(is_evaluation=True)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_metric_name, num_classes = \
        get_node_classification_tgb_data(dataset_name=args.dataset_name)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # we separately evaluate PersistentForecast and MovingAverage, since they do not contain any trainable parameters and have a different evaluation pipeline
    if args.model_name in ['PersistentForecast', 'MovingAverage']:
        evaluate_parameter_free_node_classification(args=args, train_data=train_data, val_data=val_data, test_data=test_data,
                                                    train_idx_data_loader=train_idx_data_loader, val_idx_data_loader=val_idx_data_loader,
                                                    test_idx_data_loader=test_idx_data_loader, eval_metric_name=eval_metric_name, num_classes=num_classes)

    else:
        val_metric_all_runs, test_metric_all_runs = [], []

        for run in range(args.num_runs):

            set_random_seed(seed=run)

            args.seed = run
            args.load_model_name = f'{args.model_name}_seed{args.seed}'
            args.save_result_name = f'eval_{args.load_model_name}'

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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
                dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                        dropout=args.dropout, device=args.device)
            elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
                src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                    compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
                dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                               time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                               dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                               dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
            elif args.model_name == 'CAWN':
                dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, output_dim=args.output_dim, walk_length=args.walk_length,
                                        num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
            elif args.model_name == 'TCL':
                dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                       time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                       num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
            elif args.model_name == 'GraphMixer':
                dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                              time_feat_dim=args.time_feat_dim, output_dim=args.output_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers,
                                              dropout=args.dropout, device=args.device)
            elif args.model_name == 'DyGFormer':
                dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                             time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, output_dim=args.output_dim,
                                             patch_size=args.patch_size, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                             max_input_sequence_length=args.max_input_sequence_length, device=args.device)
            else:
                raise ValueError(f"Wrong value for model_name {args.model_name}!")
            node_classifier = MLPClassifier(input_dim=args.output_dim, output_dim=num_classes, dropout=args.dropout)
            model = nn.Sequential(dynamic_backbone, node_classifier)
            logger.info(f'model -> {model}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                        f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

            # load the saved model
            load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
            early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                           save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
            early_stopping.load_checkpoint(model, map_location='cpu')

            model = convert_to_gpu(model, device=args.device)
            # put the node raw messages of memory-based models on device
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                for node_id, node_raw_messages in model[0].memory_bank.node_raw_messages.items():
                    new_node_raw_messages = []
                    for node_raw_message in node_raw_messages:
                        new_node_raw_messages.append((node_raw_message[0].to(args.device), node_raw_message[1]))
                    model[0].memory_bank.node_raw_messages[node_id] = new_node_raw_messages

            loss_func = nn.CrossEntropyLoss()
            evaluator = Evaluator(name=args.dataset_name)

            # evaluate the best model
            logger.info(f'get final performance on dataset {args.dataset_name}...')

            # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                val_losses, val_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                             model=model,
                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                             evaluate_idx_data_loader=val_idx_data_loader,
                                                                             evaluate_data=val_data,
                                                                             eval_stage='val',
                                                                             eval_metric_name=eval_metric_name,
                                                                             evaluator=evaluator,
                                                                             loss_func=loss_func,
                                                                             num_neighbors=args.num_neighbors,
                                                                             time_gap=args.time_gap)

            test_losses, test_metrics = evaluate_model_node_classification(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_data=test_data,
                                                                           eval_stage='test',
                                                                           eval_metric_name=eval_metric_name,
                                                                           evaluator=evaluator,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

            # store the evaluation metrics at the current run
            val_metric_dict, test_metric_dict = {}, {}

            if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
                logger.info(f'validate loss: {np.mean(val_losses):.4f}')
                for metric_name in val_metrics[0].keys():
                    average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
                    logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
                    val_metric_dict[metric_name] = average_val_metric

            logger.info(f'test loss: {np.mean(test_losses):.4f}')
            for metric_name in test_metrics[0].keys():
                average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                test_metric_dict[metric_name] = average_test_metric

            single_run_time = time.time() - run_start_time
            logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

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
            save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
            with open(save_result_path, 'w') as file:
                file.write(result_json)
            logger.info(f'save results at {save_result_path}')

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
