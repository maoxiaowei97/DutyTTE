import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from utils.argparser import ws
import time
from tqdm import tqdm
import os

class Dataset_list(Dataset):
    def __init__(self, xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od):
        self.xs = xs
        self.segment_travel_time_mean = segment_travel_time_mean
        self.total_ts = total_ts
        self.segment_travel_time = segment_travel_time
        self.segment_num = segment_num
        self.ts_10min = ts_10min
        self.od = od

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.segment_travel_time_mean[idx], self.total_ts[idx], self.segment_travel_time[idx], self.segment_num[idx], self.ts_10min[idx],  self.od[idx]

def collate_fn_list(batch):
    # 分离 batch 中的各个元素
    xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = zip( *batch)

    # 转换为 tensors 并堆叠
    xs = torch.tensor(xs).long()
    segment_travel_time_mean = torch.tensor(segment_travel_time_mean, dtype=torch.float32)
    total_ts = torch.tensor(total_ts, dtype=torch.float32)
    segment_travel_time = torch.tensor(segment_travel_time, dtype=torch.float32)
    segment_num = torch.tensor(segment_num).long()
    ts_10min = torch.tensor(ts_10min).long()
    od = torch.tensor(od).long()

    return xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)


class CustomDataset(Dataset):
    def __init__(self, xs, nodes, segment_travel_time_mean, start_timestamp, start_ts, total_ts, segment_travel_time,
                 segment_num, ts_10min, od, start_day):
        self.xs = xs
        self.nodes = nodes
        self.start_timestamp = start_timestamp
        self.start_ts = start_ts
        self.total_ts = total_ts
        self.segment_travel_time = segment_travel_time
        self.segment_num = segment_num
        self.ts_10min = ts_10min
        self.od = od
        self.start_day = start_day
        self.segment_travel_time_mean = segment_travel_time_mean

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        sample = {
            'xs': self.xs[idx],
            'nodes': self.nodes[idx],
            'start_timestamp': self.start_timestamp[idx],
            'segment_travel_time_mean': self.segment_travel_time_mean[idx],
            'start_ts': self.start_ts[idx],
            'total_ts': self.total_ts[idx],
            'segment_travel_time': self.segment_travel_time[idx],
            'segment_num': self.segment_num[idx],
            'ts_10min': self.ts_10min[idx],
            'od': self.od[idx],
            'start_day': self.start_day[idx]
        }
        return sample


import pickle

class Trainer:
    def __init__(self, model: nn.Module, dataset, device, args):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.rho = args.rho
        self.early_stop = args.early_stop

    def generated_path_eta_uq(self, args):
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)
        optimizer = torch.optim.Adam(self.model.parameters(), args.lr)
        """
        1.shrinked travel time
        """
        shrinked_segment_travel_time_path = ws + '/processed_data/CityA_segment_travel_time_distribution_dict_shrinked.npy'
        segment_travel_time_dict = np.load(shrinked_segment_travel_time_path, allow_pickle=True).item()
        """
        2.shrinked segment index
        """
        shinked_segment_index_dict_path = ws +  '/processed_data/CityA_segment_dict_shrinked.pkl'
        with open(shinked_segment_index_dict_path, 'rb') as f:
            segment_index_dict = pickle.load(f)

        pickle_file_path = ws + '/processed_data/train_val_test_held_subset.pkl'
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                data_to_load = pickle.load(f)
        print('loaded train_val_test_held data: ', pickle_file_path)

        """
        specify the path of predicted paths
        """

        train_generated_nodes = np.load( ws + '/results/drl_generated_paths/2024-08-20 15:40:04/planned_path_drl_train.npy', allow_pickle=True).tolist()
        val_generated_nodes = np.load(ws +'/results/drl_generated_paths/2024-08-20 15:40:04/planned_path_drl_val.npy',  allow_pickle=True).tolist()
        test_generated_nodes = np.load( ws +'/results/drl_generated_paths/2024-08-20 15:40:04/planned_path_drl_test.npy', allow_pickle=True).tolist()

        def loss_fn(y_pred_mean, bias_lower, bias_upper, y_true):
            y_pred_upper =  y_pred_mean + bias_upper
            y_pred_lower = y_pred_mean - bias_lower

            rho = self.rho
            loss0 = torch.abs(y_pred_mean - y_true)
            loss1 = torch.max(y_true - y_pred_upper, torch.tensor([0.]).cuda()) * 2 / rho
            loss2 = torch.max(y_pred_lower - y_true, torch.tensor([0.]).cuda()) * 2 / rho
            loss3 = torch.abs(y_pred_upper - y_pred_lower)
            loss = loss0  + loss1 + loss2  + loss3
            return loss.mean()

        def mis(y_pred_mean, bias_lower, bias_upper, y_true):
            y_pred_upper = y_pred_mean + bias_upper
            y_pred_lower = y_pred_mean - bias_lower
            rho = self.rho
            loss0 = np.abs(y_pred_mean - y_true)
            loss1 = np.max(y_true - y_pred_upper, 0) * 2 / rho
            loss2 = np.max(y_pred_lower - y_true, 0) * 2 / rho
            loss3 = np.abs(y_pred_upper - y_pred_lower)
            loss = loss0 + loss1 + loss2 + loss3
            return loss.mean()

        def picp(y_pred_mean, bias_lower, bias_upper, y_true):
            y_pred_upper = y_pred_mean + bias_upper
            y_pred_lower = y_pred_mean - bias_lower
            picp = (((y_true < y_pred_upper.reshape(-1)) & (y_true > y_pred_lower.reshape(-1))) + 0).sum() / len(y_true)
            return picp

        def find_non_padding_route(route):
            for i in range(len(route) - 1):
                if route[i] == 0 and route[i + 1] == 0:
                    return route[:i]
            return route

        """
        generated train set
        """
        train_ground_truth_nodes = data_to_load['nodes_train']
        train_ground_truth_segments_num = data_to_load['segment_num_train']
        train_ground_truth_od = data_to_load['od_train']

        train_generated_segment_travel_time_distribution = []
        train_generated_segment_num = []
        train_generated_segment_travel_time_mean = []
        train_generated_segments = []
        train_generated_total_ts = []
        train_generated_ts_10min = []
        train_generated_od = []

        for k, ( route, real_route, real_segment_num, real_od, start_10min_ts, start_day, total_ts, ts_10min) in enumerate(
                tqdm(zip(train_generated_nodes, train_ground_truth_nodes, train_ground_truth_segments_num,
                         train_ground_truth_od, data_to_load['ts_10min_train'], data_to_load['start_day_train'],
                         data_to_load['total_ts_train'], data_to_load['ts_10min_train']))):

            path_generated_segment_travel_time_mean = []
            path_generated_segment_travel_time_distribution = []
            path_generated_segments = []
            path_generated_total_ts = []
            path_generated_ts_10min = []
            path_generated_od = []

            non_padding_nodes = find_non_padding_route(route)
            node_num = len(non_padding_nodes)

            path_generated_total_ts.append(total_ts)
            path_generated_ts_10min.append(ts_10min)
            path_generated_od.append([real_od[0], real_od[1]])
            train_generated_segment_num.append([node_num - 1])
            for r in range(len(route)):
                current_node = route[r]
                if r < len(route) - 1:
                    next_node = route[r + 1]
                else:
                    next_node = 0

                if (current_node, next_node) in segment_index_dict:
                    path_generated_segments.extend([segment_index_dict[current_node, next_node]])
                else:
                    path_generated_segments.extend([0])

                if r >= node_num - 1:
                    path_generated_segment_travel_time_distribution.append([-1] * 11)
                    path_generated_segment_travel_time_mean.extend([float(-1)])
                else:
                    if (start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node) in segment_travel_time_dict.keys():
                        path_generated_segment_travel_time_mean.extend(([float(segment_travel_time_dict[(start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node)][-1])]))
                        path_generated_segment_travel_time_distribution.extend([segment_travel_time_dict[(start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node)]])
                    else:
                        path_generated_segment_travel_time_distribution.append([20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 20])
                        path_generated_segment_travel_time_mean.extend([float(20)])
            train_generated_segment_travel_time_mean.append(path_generated_segment_travel_time_mean)
            train_generated_segment_travel_time_distribution.append(path_generated_segment_travel_time_distribution)
            train_generated_segments.append(path_generated_segments)
            train_generated_total_ts.append(path_generated_total_ts[0])
            train_generated_ts_10min.append(path_generated_ts_10min[0])
            train_generated_od.append(path_generated_od[0])


        traindataset = Dataset_list(train_generated_segments,
                                    train_generated_segment_travel_time_mean, train_generated_total_ts,
                                    train_generated_segment_travel_time_distribution,
                                    train_generated_segment_num, train_generated_ts_10min,
                                    train_generated_od)

        traindataloader = DataLoader(traindataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list, drop_last=True)

        """
        generated validation set
        """
        val_ground_truth_nodes = data_to_load['nodes_val']
        val_ground_truth_segments_num = data_to_load['segment_num_val']
        val_ground_truth_od = data_to_load['od_val']

        val_generated_segment_travel_time_distribution = []
        val_generated_segment_num = []
        val_generated_segment_travel_time_mean = []
        val_generated_segments = []
        val_generated_total_ts = []
        val_generated_ts_10min = []
        val_generated_od = []

        for k, ( route, real_route, real_segment_num, real_od, start_10min_ts, start_day, total_ts, ts_10min) in enumerate(
                tqdm(zip(val_generated_nodes, val_ground_truth_nodes, val_ground_truth_segments_num,
                         val_ground_truth_od, data_to_load['ts_10min_val'], data_to_load['start_day_val'],
                         data_to_load['total_ts_val'], data_to_load['ts_10min_val']))):

            non_padding_nodes = find_non_padding_route(route)
            node_num = len(non_padding_nodes)
            path_generated_segment_travel_time_mean = []
            path_generated_segment_travel_time_distribution = []
            path_generated_segments = []
            path_generated_total_ts = []
            path_generated_ts_10min = []
            path_generated_od = []
            path_generated_total_ts.append(total_ts)
            path_generated_ts_10min.append(ts_10min)
            path_generated_od.append([real_od[0], real_od[1]])
            val_generated_segment_num.append([node_num - 1])
            for r in range(len(route)):
                current_node = route[r]
                if r < len(route) - 1:
                    next_node = route[r + 1]
                else:
                    next_node = 0
                if (current_node, next_node) in segment_index_dict:
                    path_generated_segments.extend([segment_index_dict[current_node, next_node]])
                else:
                    path_generated_segments.extend([0])

                if r >= node_num - 1:
                    path_generated_segment_travel_time_distribution.append([-1] * 11)
                    path_generated_segment_travel_time_mean.extend([float(-1)])
                else:
                    if (start_day[0], int(start_10min_ts[0]) - 1, current_node,  next_node) in segment_travel_time_dict.keys():
                        path_generated_segment_travel_time_mean.extend(([float(segment_travel_time_dict[(
                            start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node)][-1])]))
                        path_generated_segment_travel_time_distribution.extend([segment_travel_time_dict[(
                            start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node)]])
                    else:
                        path_generated_segment_travel_time_distribution.append([20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 20])
                        path_generated_segment_travel_time_mean.extend([float(20)])
            val_generated_segment_travel_time_mean.append(path_generated_segment_travel_time_mean)
            val_generated_segment_travel_time_distribution.append(path_generated_segment_travel_time_distribution)
            val_generated_segments.append(path_generated_segments)
            val_generated_total_ts.append(path_generated_total_ts[0])
            val_generated_ts_10min.append(path_generated_ts_10min[0])
            val_generated_od.append(path_generated_od[0])

        valdataset = Dataset_list(val_generated_segments, val_generated_segment_travel_time_mean,
                                   val_generated_total_ts, val_generated_segment_travel_time_distribution,
                                   val_generated_segment_num, val_generated_ts_10min, val_generated_od)

        valdataloader = DataLoader(valdataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        """
        generated test set
        """
        test_ground_truth_nodes = data_to_load['nodes_test']
        test_ground_truth_segments_num = data_to_load['segment_num_test']
        test_ground_truth_od = data_to_load['od_test']

        test_generated_segment_travel_time_distribution = []
        test_generated_segment_num = []
        test_generated_segment_travel_time_mean = []
        test_generated_segments = []
        test_generated_total_ts = []
        test_generated_ts_10min = []
        test_generated_od = []

        for k, ( route, real_route, real_segment_num, real_od, start_10min_ts, start_day, total_ts, ts_10min) in enumerate(
                tqdm(zip(test_generated_nodes, test_ground_truth_nodes, test_ground_truth_segments_num,
                         test_ground_truth_od, data_to_load['ts_10min_test'], data_to_load['start_day_test'],
                         data_to_load['total_ts_test'], data_to_load['ts_10min_test']))):

            non_padding_nodes = find_non_padding_route(route)
            node_num = len(non_padding_nodes)
            path_generated_segment_travel_time_mean = []
            path_generated_segment_travel_time_distribution = []
            path_generated_segments = []
            path_generated_total_ts = []
            path_generated_ts_10min = []
            path_generated_od = []
            path_generated_total_ts.append(total_ts)
            path_generated_ts_10min.append(ts_10min)
            path_generated_od.append([real_od[0], real_od[1]])
            test_generated_segment_num.append([node_num - 1])
            for r in range(len(route)):
                current_node = route[r]
                if r < len(route) - 1:
                    next_node = route[r + 1]
                else:
                    next_node = 0

                if (current_node, next_node) in segment_index_dict:
                    path_generated_segments.extend([segment_index_dict[current_node, next_node]])
                else:
                    path_generated_segments.extend([0])

                if r >= node_num - 1:
                    path_generated_segment_travel_time_distribution.append([-1] * 11)
                    path_generated_segment_travel_time_mean.extend([float(-1)])
                else:
                    if (start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node) in segment_travel_time_dict.keys():
                        path_generated_segment_travel_time_mean.extend(([float(segment_travel_time_dict[(
                            start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node)][-1])]))
                        path_generated_segment_travel_time_distribution.extend([segment_travel_time_dict[(
                            start_day[0], int(start_10min_ts[0]) - 1, current_node, next_node)]])
                    else:
                        path_generated_segment_travel_time_distribution.append([20, 1, 0, 0, 0, 0, 0, 0, 0, 0, 20])
                        path_generated_segment_travel_time_mean.extend([float(20)])
            test_generated_segment_travel_time_mean.append(path_generated_segment_travel_time_mean)
            test_generated_segment_travel_time_distribution.append(path_generated_segment_travel_time_distribution)
            test_generated_segments.append(path_generated_segments)
            test_generated_total_ts.append(path_generated_total_ts[0])
            test_generated_ts_10min.append(path_generated_ts_10min[0])
            test_generated_od.append(path_generated_od[0])

        testdataset = Dataset_list(test_generated_segments,  test_generated_segment_travel_time_mean,
                                  test_generated_total_ts, test_generated_segment_travel_time_distribution,
                                   test_generated_segment_num, test_generated_ts_10min, test_generated_od)

        testdataloader = DataLoader(testdataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        train_loss = []
        early_stop = EarlyStop(mode='minimize', patience=self.early_stop)
        for epoch in range(args.n_epoch):
            if early_stop.stop_flag: break
            self.model.train()
            print('train epoch {}'.format(epoch))
            for batch in tqdm(traindataloader):
                xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                predict_mean, bias_lower, bias_upper = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                mis_loss = loss_fn(predict_mean.reshape(-1), bias_lower.reshape(-1), bias_upper.reshape(-1), total_ts.reshape(-1).float().to(self.device))
                optimizer.zero_grad()
                mis_loss.backward()
                optimizer.step()
                train_loss.append(mis_loss.item())
            print(f'training... loss of epoch: {epoch}: ' + str((sum(train_loss) / len(train_loss))))
            if epoch % 1 == 0:
                print(f'validation... of epoch {epoch}')
                predicts = []
                predicts_bias_lower = []
                predicts_bias_upper = []
                label = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(valdataloader):
                        xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                        predict_mean, bias_lower, bias_upper = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                        total_ts = pad_sequence(total_ts, batch_first=True, padding_value=0).float()

                        predicts += predict_mean.reshape(-1).tolist()
                        predicts_bias_lower += bias_lower.reshape(-1).tolist()
                        predicts_bias_upper += bias_upper.reshape(-1).tolist()
                        label += total_ts.reshape(-1).float().tolist()

                    predicts = np.array(predicts).reshape(-1)
                    label = np.array(label).reshape(-1)
                    predicts_bias_lower = np.array(predicts_bias_lower).reshape(-1)
                    predicts_bias_upper = np.array(predicts_bias_upper).reshape(-1)
                    from sklearn.metrics import mean_squared_error as mse
                    from sklearn.metrics import mean_absolute_error as mae
                    def mape_(label, predicts):
                        return (abs(predicts - label) / label).mean()

                    val_mape = mape_(label, predicts)
                    val_mse = mse(label, predicts)
                    val_mae = mae(label, predicts)

                    predicts_upper = predicts + predicts_bias_upper
                    predicts_lower = predicts - predicts_bias_lower
                    val_width = predicts_upper - predicts_lower
                    val_mis = mis(predicts, predicts_bias_lower, predicts_bias_upper, label)
                    val_picp = picp(predicts, predicts_bias_lower, predicts_bias_upper, label)

                    print('val point estimation: MAPE:%.3f\tRMSE:%.2f\tMAE:%.2f' % (val_mape * 100, np.sqrt(val_mse), val_mae))
                    print('val_width: ' + str(np.mean(val_width.reshape(-1))))
                    print('val_mis:' + str(val_mis))
                    print('val_picp: ' + str(val_picp))
                is_best_change = early_stop.append(val_mape * 100)
                if is_best_change:
                    dir_check(ws + f"/model_params/MoEUQ/{local_time}/")
                    best_model_path = ws + f"/model_params/MoEUQ/{local_time}/finished_{epoch}.pth"
                    torch.save(self.model.state_dict(), best_model_path)
                    print('val best model saved at: ', best_model_path)
                    self.model.load_state_dict(torch.load(best_model_path))
                    print('val best model loaded')

        print('testing...')
        print('load best val model at: ', best_model_path)
        self.model.load_state_dict(torch.load(best_model_path))
        print('testing, best val model loaded')
        predicts = []
        predicts_bias_lower = []
        predicts_bias_upper = []
        label = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(testdataloader):
                xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                predict_mean, bias_lower, bias_upper = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                total_ts = pad_sequence(total_ts, batch_first=True, padding_value=0).float()

                predicts += predict_mean.reshape(-1).tolist()
                predicts_bias_lower += bias_lower.reshape(-1).tolist()
                predicts_bias_upper += bias_upper.reshape(-1).tolist()
                label += total_ts.reshape(-1).float().tolist()

            predicts = np.array(predicts).reshape(-1)
            label = np.array(label).reshape(-1)
            predicts_bias_lower = np.array(predicts_bias_lower).reshape(-1)
            predicts_bias_upper = np.array(predicts_bias_upper).reshape(-1)
            from sklearn.metrics import mean_squared_error as mse
            from sklearn.metrics import mean_absolute_error as mae
            def mape_(label, predicts):
                return (abs(predicts - label) / label).mean()

            test_mape = mape_(label, predicts)
            test_mse = mse(label, predicts)
            test_mae = mae(label, predicts)

            predicts_upper = predicts + predicts_bias_upper
            predicts_lower = predicts - predicts_bias_lower
            test_width = predicts_upper - predicts_lower
            test_mis = mis(predicts, predicts_bias_lower, predicts_bias_upper, label)
            test_picp = picp(predicts, predicts_bias_lower, predicts_bias_upper, label)

            print('test point estimation: MAPE:%.3f\tRMSE:%.2f\tMAE:%.2f' % (test_mape * 100, np.sqrt(test_mse), test_mae))
            print('test_width: ' + str(np.mean(test_width.reshape(-1))))
            print('test_mis:' + str(test_mis))
            print('test_picp: ' + str(test_picp))

class EarlyStop():
    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1  # the best epoch
        self.is_best_change = False  # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        # update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        # update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize' else self.metric_lst.index(
            min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch  # update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:
            return -1
        else:
            return self.metric_lst[self.best_epoch]


def whether_stop(metric_lst=[], n=2, mode='maximize'):
    if len(metric_lst) < 1: return False  # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx, v in enumerate(metric_lst):
        if v == max_v: max_idx = idx
    return max_idx < len(metric_lst) - n