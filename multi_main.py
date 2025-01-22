"""
run example:
nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --pro_features_flag \
      --tandem_fea_flag --decay_flag --bounding_prob_mask_flag > chengdu_8.txt &
nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --pro_features_flag \
      --tandem_fea_flag --decay_flag --grid_flag  > chengdu_8.txt &

nohup python -u multi_main.py --city Chengdu --keep_ratio 0.0625 --pro_features_flag \
      --tandem_fea_flag --decay_flag   > chengdu_16.txt &
nohup python -u multi_main.py --city Chengdu --keep_ratio 0.125 --pro_features_flag \
      --tandem_fea_flag --decay_flag   > chengdu_8.txt &
nohup python -u multi_main.py --city Shanghai --keep_ratio 0.125 --pro_features_flag \
      --tandem_fea_flag --decay_flag   > shanghai_8.txt &
nohup python -u multi_main.py --city Porto --keep_ratio 0.125 --pro_features_flag \
      --tandem_fea_flag --decay_flag   > porto_8.txt &
version: GPS_Transformer_Grid_v2_6_v4beta
"""

import time
from tqdm import tqdm
import logging
import dgl
import sys
sys.path.append('../../')


from model import ProbRho, ProbTraffic, ProbTravelTime
import pickle
import  ast
from torch.utils.data import Dataset, DataLoader,Subset

import os
import argparse

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.mbr import MBR
from utils.graph_func import *
from utils.dataset import Dataset, collate_fn
from multi_train import evaluate, init_weights, train, test
from model import Encoder, DecoderMulti, Seq2SeqMulti
from utils.model_utils import AttrDict
import numpy as np
import json
from utils.shortest_path_func import SPSolver
import  pandas as pd
import networkx as nx
from utils.dataset import calculate_road_trans




# add

def create_data_loaders(dataset, batch_size_dict, collate_fn=lambda x: collate_fn(x), num_workers=4, pin_memory=False, drop_last=True):
    data_loaders = {}
    start_index = 0

    for key, num_trajectories in batch_size_dict.items():
        # 获取数据集中的子集
        end_index = start_index + num_trajectories
        subset_indices = list(range(start_index, end_index))
        subset = Subset(dataset, subset_indices)

        # 创建 DataLoader
        data_loader = DataLoader(
            subset,
            batch_size=64,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        data_loaders[key] = data_loader
        start_index = end_index  # 更新起始索引

    return data_loaders

def save_json_data(data, dir, file_name):
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir + file_name, 'w') as fp:
        json.dump(data, fp)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def get_rid_rnfea_dict(rn: RoadNetworkMapFull) -> torch.Tensor:
    norm_feat = torch.zeros(rn.valid_edge_cnt_one, 11)
    max_length = np.max(rn.edgeDis)
    for rid in rn.valid_edge.keys():
        norm_rid = [0 for _ in range(11)]
        norm_rid[0] = np.log10(rn.edgeDis[rid] + 1e-6) / np.log10(max_length)
        norm_rid[rn.wayType[rid] + 1] = 1
        in_degree = 0
        for eid in rn.edgeDict[rid]:  # 入度
            if eid in rn.valid_edge.keys():
                in_degree += 1
        out_degree = 0
        for eid in rn.edgeRevDict[rid]:  # 出度
            if eid in rn.valid_edge.keys():
                out_degree += 1
        norm_rid[9] = in_degree
        norm_rid[10] = out_degree
        norm_feat[rn.valid_edge_one[rid]] = torch.tensor(norm_rid)
    norm_feat[0] = torch.tensor([0 for _ in range(11)])
    return norm_feat


if __name__ == '__main__':
    total_start_time = time.time()
    parser = argparse.ArgumentParser(description='GTR-HMD')
    parser.add_argument('--city', type=str, default='Chengdu')
    parser.add_argument('--keep_ratio', type=float, default=0.125, help='keep ratio in float')
    parser.add_argument('--lambda1', type=int, default=10, help='weight for multi task rate')
    parser.add_argument('--lambda2', type=int, default=0.1, help='weight for multi task rate')
    parser.add_argument('--hid_dim', type=int, default=512, help='hidden dimension')
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--time_step', type=int, default=8, help='time steps')
    parser.add_argument('--traffic_dim', type=int, default=128, help='traffic_dim')
    # parser.add_argument('--pro_features_flag', action='store_true', help='flag of using profile features')
    # parser.add_argument('--online_features_flag', action='store_true', help='flag of using traffic features')
    # parser.add_argument('--tandem_fea_flag', action='store_true', help='flag of using tandem rid features')
    parser.add_argument('--no_attn_flag', action='store_false', help='flag of using attention')
    parser.add_argument('--load_pretrained_flag', action='store_true', help='flag of load pretrained model')
    parser.add_argument('--model_old_path', type=str, default='', help='old model path')
    # parser.add_argument('--decay_flag', action='store_true')
    parser.add_argument('--grid_flag', action='store_true')
    parser.add_argument('--transformer_layers', type=int, default=2)

    opts = parser.parse_args()

    # opts.online_features_flag = True

    device = torch.device("cuda:0")

    city = opts.city
    map_root = f"./data/roadnet/{city}/"

    if city == "Porto":
        zone_range = [41.111975, -8.667057, 41.177462, -8.585305]
        ts = 15
    elif city == "Chengdu":
        zone_range = [30.655, 104.043, 30.727, 104.129]
        ts = 12
    elif city == "Harbin":
        zone_range = [45.697920, 126.586130, 45.777090, 126.671862]
        ts = 15
    else:
        raise NotImplementedError

    if city == "Porto":
        rn = RoadNetworkMapFull(map_root, zone_range=[41.111975, -8.667057, 41.177462, -8.585305], unit_length=50)
    elif city == "Chengdu":
        rn = RoadNetworkMapFull(map_root, zone_range=[30.655, 104.043, 30.727, 104.129], unit_length=50)
    elif city == "Harbin":
        rn = RoadNetworkMapFull(map_root, zone_range=[45.697920, 126.586130, 45.777090, 126.671862], unit_length=50)
    else:
        raise NotImplementedError

    args = AttrDict()
    args_dict = {
        'device': device,
        'temperature': 5,
        'gnn_type': 'gat',
        'num_layers': 2,
        'transformer_layers': opts.transformer_layers,
        'max_depths': 3,

        # pre train
        'load_pretrained_flag': opts.load_pretrained_flag,
        'model_old_path': opts.model_old_path,

        # attention
        'attn_flag': opts.no_attn_flag,

        # constraint
        'dis_prob_mask_flag': True,
        'search_dist': 100 if opts.city != 'Porto' else 50,
        'neighbor_dist': 400,
        'beta': 15,
        'gamma': 30,

        # features
        'tandem_fea_flag': True,
        'pro_features_flag': True,
        # 'online_features_flag': opts.online_features_flag,
        'online_features_flag': False,
        'grid_flag': opts.grid_flag,
        'poi_features': False,

        # extra info module
        'rid_fea_dim': 11,  # 1[norm length] + 8[way type] + 1[in degree] + 1[out degree]
        'pro_input_dim': 25,  # 24[hour] + 1[holiday]
        'pro_output_dim': 8,
        'poi_num': 0,
        'online_dim': 0,  # poi/roadnetwork features dim

        #traffic_features
        'traffic_features': True,
        'traffic_dim': opts.traffic_dim,
        'num_u': rn.valid_edge_cnt_one,
        'dim_u': 200,
        'dict_u': rn.valid_edge_one,
        'lengths': rn.edgeDis,
        'num_s1':len(rn.wayType),
        'dim_s1': 64,
        'dict_s1':rn.wayType,
        'hidden_size1':300,
        'hidden_size2': 500,
        'hidden_size3': 600,
        'dim_rho':256,
        'traffic_dropout':0.3,
        'use_selu':False,
        'dim_c': 400,
        'n_in':1,

        'speed_features': True,
        'use_noise': True,
        'speed_dim': 512,
        'speed_hid': 512,

        'road_trans_features': False,
        'time_step':opts.time_step,
        'loc_size': 4285,
        # MBR
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],

        # input data params
        'city': opts.city,
        'keep_ratio': opts.keep_ratio,
        'grid_size': opts.grid_size,
        'time_span': ts,
        'win_size': 1000,
        'ds_type': 'uniform',
        'shuffle': True,

        # model params
        'hid_dim': opts.hid_dim,
        'id_emb_dim': opts.hid_dim,
        'dropout': 0.001,
        'id_size': rn.valid_edge_cnt_one,

        'lambda1': opts.lambda1,
        'lambda2': opts.lambda2,
        'n_epochs': opts.epochs,
        'batch_size': 64,
        'learning_rate': 0.0004,
        'tf_ratio': 0.5,
        'decay_flag': True,
        'decay_ratio': 0.9,
        'clip': 1,
        'log_step': 1,
        'verbose_flag': False
    }
    args.update(args_dict)

    if args.road_trans_features:
        G = nx.DiGraph()
        G.add_nodes_from(list(range(args.loc_size)))
        valid_edge = rn.valid_edge_one
        # build global trans_graph
        adj_path = f"./data/road_trans/road_trans_{city}.txt"
        adj_pd = pd.read_csv(adj_path)
        A, A_k = calculate_road_trans(args, valid_edge, adj_pd, G)
    else:
        A, A_k = None, None

    g = get_total_graph(rn)
    subg = get_sub_graphs(rn, max_deps=args.max_depths)

    print('Preparing data...')
    traj_root = f"./data/{city}/"

    if args.tandem_fea_flag:
        fea_flag = True
    else:
        fea_flag = False

    model_save_root = f'./model/GTR_HMD{city}/'
    if not os.path.exists(model_save_root):
        os.makedirs(model_save_root)

    if args.load_pretrained_flag:
        model_save_path = args.model_old_path
    else:
        model_save_path = model_save_root + 'GTR_HMD' + args.city + '_' + 'keep-ratio_' + str(args.keep_ratio) + '_' + time.strftime("%Y%m%d_%H%M%S") +'/'

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=model_save_path + 'log.txt',
                        filemode='a')

    mbr = MBR(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    args.grid_num = gps2grid(SPoint(args.max_lat, args.max_lng), mbr, args.grid_size)
    args.grid_num = (args.grid_num[0] + 1, args.grid_num[1] + 1)
    args.update(args_dict)
    # print(args)
    # logging.info(args_dict)

    args.g = g
    args.subg = dgl.batch(subg).to(args.device)
    args.subgs = subg
    args.rn_grid_dict = get_rn_grid(mbr, rn, opts.grid_size)

    print(args.subg)
    logging.info(args.subg)



    # load features
    norm_grid_poi_dict, norm_grid_rnfea_dict, online_features_dict = None, None, None


    if args:
        rid_features_dict = get_rid_rnfea_dict(rn).to(args.device)
    else:
        rid_features_dict = None

    

    if args.traffic_features:
        traffic_path = f'./data/traffic/{city}/{city}_traffics.pkl'
        with open(traffic_path, 'rb') as f:
            data_S = pickle.load(f)
        data_S_tensor = torch.tensor(data_S, dtype=torch.float32).to(device)
    else:
        data_S_tensor = None

    if args.speed_features:
        speed_path = f'./data/traffic/{city}/{city}_speed_onehot.npy'
        with open(speed_path, 'rb') as f:
            speeds = np.load(speed_path)
    else:
        speeds = None

    # load dataset
    train_dataset = Dataset(rn, traj_root, mbr, args, A,A_k, 'train')
    valid_dataset = Dataset(rn, traj_root, mbr, args, A, A_k, 'valid')
    test_dataset = Dataset(rn, traj_root, mbr, args, A,A_k, 'test')
    print('training dataset shape: ' + str(len(train_dataset)))
    print('validation dataset shape: ' + str(len(valid_dataset)))
    print('testing dataset shape: ' + str(len(test_dataset)))

    with open(f'./data/{city}/train/train_batch.txt', 'r') as f:
        batch_size_content = f.read()
    batch_size_dict_train = ast.literal_eval(batch_size_content)
    with open(f'./data/{city}/valid/valid_batch.txt', 'r') as f:
        batch_size_content = f.read()
    batch_size_dict_valid = ast.literal_eval(batch_size_content)
    with open(f'./data/{city}/test/test_batch.txt', 'r') as f:
        batch_size_content = f.read()
    batch_size_dict_test = ast.literal_eval(batch_size_content)

    train_data_loaders = create_data_loaders(train_dataset, batch_size_dict_train, pin_memory=False)
    valid_data_loaders = create_data_loaders(valid_dataset, batch_size_dict_valid, pin_memory=False)
    test_data_loaders = create_data_loaders(test_dataset, batch_size_dict_test, pin_memory=True)

    logging.info('Finish data preparing.')
    logging.info('training dataset shape: ' + str(len(train_dataset)))
    logging.info('validation dataset shape: ' + str(len(valid_dataset)))
    logging.info('testing dataset shape: ' + str(len(test_dataset)))

    enc = Encoder(args)
    dec = DecoderMulti(args)
    model = Seq2SeqMulti(enc, dec, device, args).to(device)
    model.apply(init_weights)  # learn how to init weights

    if args.load_pretrained_flag:
        model = torch.load(args.model_old_path + 'val-best-model.pt')

    print('model', str(model))
    logging.info('model' + str(model))

    ls_train_loss, ls_train_id_acc1, ls_train_id_recall, ls_train_id_precision, \
    ls_train_rate_loss, ls_train_id_loss, ls_train_mae, ls_train_rmse = [], [], [], [], [], [], [], []
    ls_valid_loss, ls_valid_id_acc1, ls_valid_id_recall, ls_valid_id_precision, \
    ls_valid_rate_loss, ls_valid_id_loss, ls_valid_mae, ls_valid_rmse = [], [], [], [], [], [], [], []

    dict_train_loss = {}
    dict_valid_loss = {}
    best_valid_loss = float('inf')  # compare id loss



    # get all parameters (model parameters + task dependent log variances)
    log_vars = [torch.zeros((1,), requires_grad=True, device=device)] * 2  # use for auto-tune multi-task param
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    stopping_count = 0
    for epoch in tqdm(range(args.n_epochs)):
        start_time = time.time()

        # train stage
        total_iterator = 0
        epoch_ttl_loss = 0
        epoch_id1_loss = 0
        epoch_recall_loss = 0
        epoch_precision_loss = 0
        epoch_rate_loss = 0
        epoch_train_id_loss = 0
        epoch_mae_loss = 0
        epoch_rmse_loss = 0

        for time_slot, data_iterator in train_data_loaders.items():
            total_iterator += len(data_iterator)

            log_vars, time_ttl_loss, time_id1_loss, time_recall_loss, time_precision_loss, \
            time_rate_loss, time_train_id_loss, time_mae_loss, time_rmse_loss = train(model, data_iterator, optimizer, log_vars,
                                                   rn, online_features_dict, rid_features_dict, args, time_slot, data_S_tensor,speeds, A)
            epoch_ttl_loss += time_ttl_loss
            epoch_id1_loss += time_id1_loss
            epoch_recall_loss += time_recall_loss
            epoch_precision_loss += time_precision_loss
            epoch_rate_loss += time_rate_loss
            epoch_train_id_loss += time_train_id_loss
            epoch_mae_loss += time_mae_loss
            epoch_rmse_loss += time_rmse_loss

        new_log_vars, train_loss, train_id_acc1, train_id_recall, train_id_precision,train_rate_loss, train_id_loss, train_mae, \
        train_rmse = log_vars, epoch_ttl_loss / total_iterator, epoch_id1_loss / total_iterator, epoch_recall_loss / total_iterator, \
           epoch_precision_loss / total_iterator, epoch_rate_loss / total_iterator, epoch_train_id_loss / total_iterator, \
           epoch_mae_loss / total_iterator, epoch_rmse_loss / total_iterator


        # evaluate stage
        total_iterator = 0
        epoch_id1_loss = 0
        epoch_recall_loss = 0
        epoch_precision_loss = 0
        epoch_rate_loss = 0
        epoch_valid_id_loss = 0
        epoch_mae_loss = 0
        epoch_rmse_loss = 0
        for time_slot, data_iterator in valid_data_loaders.items():
            total_iterator += len(data_iterator)
            time_id1_loss, time_recall_loss, time_precision_loss, time_rate_loss, time_id_loss, time_mae, time_rmse =\
                evaluate(model, data_iterator, rn, online_features_dict, rid_features_dict, args, time_slot, data_S_tensor, speeds, A)

            epoch_id1_loss += time_id1_loss
            epoch_recall_loss += time_recall_loss
            epoch_precision_loss += time_precision_loss
            epoch_rate_loss += time_rate_loss
            epoch_valid_id_loss += time_id_loss
            epoch_mae_loss += time_mae
            epoch_rmse_loss += time_rmse


        valid_id_acc1, valid_id_recall, valid_id_precision, valid_rate_loss, valid_id_loss, valid_mae, valid_rmse = epoch_id1_loss / total_iterator, \
            epoch_recall_loss / total_iterator, epoch_precision_loss / total_iterator, epoch_rate_loss / total_iterator, epoch_valid_id_loss / total_iterator, \
            epoch_mae_loss / total_iterator, epoch_rmse_loss / total_iterator

        ls_train_loss.append(train_loss)
        ls_train_id_acc1.append(train_id_acc1)
        ls_train_id_recall.append(train_id_recall)
        ls_train_id_precision.append(train_id_precision)
        ls_train_rate_loss.append(train_rate_loss)
        ls_train_id_loss.append(train_id_loss)
        ls_train_mae.append(train_mae)
        ls_train_rmse.append(train_rmse)

        ls_valid_id_acc1.append(valid_id_acc1)
        ls_valid_id_recall.append(valid_id_recall)
        ls_valid_id_precision.append(valid_id_precision)
        ls_valid_rate_loss.append(valid_rate_loss)
        ls_valid_id_loss.append(valid_id_loss)
        valid_loss = valid_rate_loss + valid_id_loss
        ls_valid_loss.append(valid_loss)
        ls_valid_mae.append(valid_mae)
        ls_valid_rmse.append(valid_rmse)

        dict_train_loss['train_ttl_loss'] = ls_train_loss
        dict_train_loss['train_id_acc1'] = ls_train_id_acc1
        dict_train_loss['train_id_recall'] = ls_train_id_recall
        dict_train_loss['train_id_precision'] = ls_train_id_precision
        dict_train_loss['train_rate_loss'] = ls_train_rate_loss
        dict_train_loss['train_id_loss'] = ls_train_id_loss
        dict_train_loss['train_mae'] = ls_train_mae
        dict_train_loss['train_rmse'] = ls_train_rmse

        dict_valid_loss['valid_ttl_loss'] = ls_valid_loss
        dict_valid_loss['valid_id_acc1'] = ls_valid_id_acc1
        dict_valid_loss['valid_id_recall'] = ls_valid_id_recall
        dict_valid_loss['valid_id_precision'] = ls_valid_id_precision
        dict_valid_loss['valid_rate_loss'] = ls_valid_rate_loss
        dict_valid_loss['valid_id_loss'] = ls_valid_id_loss
        dict_valid_loss['valid_mae'] = ls_valid_mae
        dict_valid_loss['valid_rmse'] = ls_valid_rmse

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, model_save_path + 'val-best-model.pt')
            stopping_count = 0
        else:
            stopping_count += 1

        if (epoch % args.log_step == 0) or (epoch == args.n_epochs - 1):
            logging.info('Epoch: ' + str(epoch + 1) + ' Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')
            logging.info('Epoch: ' + str(epoch + 1) + ' TF Ratio: ' + str(args.tf_ratio))
            weights = [torch.exp(weight) ** 0.5 for weight in new_log_vars]
            logging.info('log_vars:' + str(weights))
            logging.info('\tTrain Loss:' + str(train_loss) +
                         '\tTrain RID Acc1:' + str(train_id_acc1) +
                         '\tTrain RID Recall:' + str(train_id_recall) +
                         '\tTrain RID Precision:' + str(train_id_precision) +
                         '\tTrain Rate Loss:' + str(train_rate_loss) +
                         '\tTrain RID Loss:' + str(train_id_loss) +
                         '\tTrain MAE Loss:' + str(train_mae) +
                         '\tTrain RMSE Loss:' + str(train_rmse))
            logging.info('\tValid Loss:' + str(valid_loss) +
                         '\tValid RID Acc1:' + str(valid_id_acc1) +
                         '\tValid RID Recall:' + str(valid_id_recall) +
                         '\tValid RID Precision:' + str(valid_id_precision) +
                         '\tValid Rate Loss:' + str(valid_rate_loss) +
                         '\tValid RID Loss:' + str(valid_id_loss) +
                         '\tValid MAE Loss:' + str(valid_mae) +
                         '\tValid RMSE Loss:' + str(valid_rmse))

            torch.save(model, model_save_path + 'train-mid-model.pt')
            save_json_data(dict_train_loss, model_save_path, "train_loss.json")

            save_json_data(dict_valid_loss, model_save_path, "valid_loss.json")
        if args.decay_flag:
            args.tf_ratio = args.tf_ratio * args.decay_ratio

  # test stage
  #   model_save_path = './model/RNTrajRec/Porto/Porto/'
    model = torch.load(model_save_path + 'val-best-model.pt').to(device)
    verbose_root = f'./model/GTR-HMD/{city}/'
    output = None
    if args.verbose_flag:
        if not os.path.exists(verbose_root):
            os.makedirs(verbose_root)
        output_path = verbose_root + f'test_output.txt'
        output = open(output_path, 'w+')
    traj_path = traj_root + f'test/test_output.txt'

    sp_solver = SPSolver(rn, use_ray=False, use_lru=True)

    ls_test_id_acc, ls_test_id_recall, ls_test_id_precision, ls_test_id_f1, \
    ls_test_mae, ls_test_rmse, ls_test_rn_mae, ls_test_rn_rmse = [], [], [], [], [], [], [], []

    start_time = time.time()

    total_iterator = 0
    epoch_id1_loss = []
    epoch_recall_loss = []
    epoch_precision_loss = []
    epoch_f1_loss = []
    epoch_mae_loss = []
    epoch_rmse_loss = []
    epoch_rn_mae_loss = []
    epoch_rn_rmse_loss = []

    for time_slot, data_iterator in train_data_loaders.items():

        time_id1_loss, time_recall_loss, time_precision_loss, time_f1_loss, time_mae_loss,\
        time_rmse_loss, time_rn_mae_loss, time_rn_rmse_loss = test(model, data_iterator, rn, online_features_dict,
                                                                   rid_features_dict, args, sp_solver, time_slot,data_S_tensor, speeds, A)
        epoch_id1_loss.extend(time_id1_loss)
        epoch_recall_loss.extend(time_recall_loss)
        epoch_precision_loss.extend(time_precision_loss)
        epoch_f1_loss.extend(time_f1_loss)
        epoch_mae_loss.extend(time_mae_loss)
        epoch_rmse_loss.extend(time_rmse_loss)
        epoch_rn_mae_loss.extend(time_rn_mae_loss)
        epoch_rn_rmse_loss.extend(time_rn_rmse_loss)


    test_id_acc, test_id_recall, test_id_precision, test_id_f1, \
    test_mae, test_rmse, test_rn_mae, test_rn_rmse = np.mean(epoch_id1_loss), np.mean(epoch_recall_loss), np.mean(epoch_precision_loss), np.mean(epoch_f1_loss), \
           np.mean(epoch_mae_loss), np.mean(epoch_rmse_loss), np.mean(epoch_rn_mae_loss), \
           np.mean(epoch_rn_rmse_loss)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    logging.info('Time: ' + str(epoch_mins) + 'm' + str(epoch_secs) + 's')

    logging.info('\tTest RID Acc:' + str(test_id_acc) +
                 '\tTest RID Recall:' + str(test_id_recall) +
                 '\tTest RID Precision:' + str(test_id_precision) +
                 '\tTest RID F1 Score:' + str(test_id_f1) +
                 '\tTest MAE Loss:' + str(test_mae) +
                 '\tTest RMSE Loss:' + str(test_rmse) +
                 '\tTest RN MAE Loss:' + str(test_rn_mae) +
                 '\tTest RN RMSE Loss:' + str(test_rn_rmse))

    total_end_time = time.time()
    total_hours =  (total_end_time - total_start_time)/ 3600
    print(f'Total Time: {total_hours:.2f} hours')
    logging.info(f'Total Time: {total_hours:.2f} hours')
