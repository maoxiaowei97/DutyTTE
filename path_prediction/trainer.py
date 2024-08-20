import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from loader.dataset import TrajFastDataset
import time
from tqdm import tqdm
import pickle
from path_prediction.policy_network import Planner
from utils.argparser import ws
def LCSSDistance(a, b):
    lena = len(a)
    lenb = len(b)
    c = [[0 for i in range(lenb + 1)] for j in range(lena + 1)]
    for i in range(lena):
        for j in range(lenb):
            if a[i] == b[j]:
                c[i + 1][j + 1] = c[i][j] + 1
            elif c[i + 1][j] > c[i][j + 1]:
                c[i + 1][j + 1] = c[i + 1][j]
            else:
                c[i + 1][j + 1] = c[i][j + 1]
    return c[lena - 1][lenb - 1]

def DTWDistance(G, s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            a_lng, a_lat = G.nodes[s1[i]]["lng"], G.nodes[s1[i]]["lat"]
            b_lng, b_lat = G.nodes[s2[j]]["lng"], G.nodes[s2[j]]["lat"]
            dist = (a_lng - b_lng) ** 2 + (a_lat - b_lat) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return DTW[len(s1) - 1, len(s2) - 1] ** 0.5
class Dataset_list(Dataset):
    def __init__(self, xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day):
        self.xs = xs
        self.nodes = nodes
        self.segment_travel_time_mean = segment_travel_time_mean
        self.start_timestamp = start_timestamp
        self.total_ts = total_ts
        self.segment_travel_time = segment_travel_time
        self.segment_num = segment_num
        self.ts_10min = ts_10min
        self.od = od
        self.start_day = start_day

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.nodes[idx], self.segment_travel_time_mean[idx], self.start_timestamp[idx],\
                self.total_ts[idx], self.segment_travel_time[idx], self.segment_num[idx], self.ts_10min[idx],\
               self.od[idx], self.start_day[idx]


def collate_fn_list(batch):
    xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day = zip(*batch)
    xs = torch.tensor(xs).long()
    nodes = torch.tensor(nodes).long()
    segment_travel_time_mean = torch.tensor(segment_travel_time_mean, dtype=torch.float32)
    start_timestamp = torch.tensor(start_timestamp, dtype=torch.float32)
    total_ts = torch.tensor(total_ts, dtype=torch.float32)
    segment_travel_time = torch.tensor(segment_travel_time, dtype=torch.float32)
    segment_num = torch.tensor(segment_num).long()
    ts_10min = torch.tensor(ts_10min).long()
    od = torch.tensor(od).long()
    start_day = torch.tensor(start_day).long()

    return  xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day


def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

class Trainer:
    def __init__(self, model: Planner, dataset: TrajFastDataset, device, args):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.early_stop = args.early_stop
        import os
        pickle_file_path = ws +  '/processed_data/train_val_test_held_subset.pkl'
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                data_to_load = pickle.load(f)
        print('loaded aranged data')

        traindataset = Dataset_list(data_to_load['train_xs'], data_to_load['nodes_train'],
                                    data_to_load['segment_travel_time_mean_train'],
                                    data_to_load['start_timestamp_train'], data_to_load['total_ts_train'],
                                    data_to_load['segment_travel_time_train'],
                                    data_to_load['segment_num_train'], data_to_load['ts_10min_train'],
                                    data_to_load['od_train'], data_to_load['start_day_train'])
        self.traindataloader = DataLoader(traindataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list)

        valdataset = Dataset_list(data_to_load['val_xs'], data_to_load['nodes_val'],
                                   data_to_load['segment_travel_time_mean_val'],
                                   data_to_load['start_timestamp_val'], data_to_load['total_ts_val'],
                                   data_to_load['segment_travel_time_val'],
                                   data_to_load['segment_num_val'], data_to_load['ts_10min_val'],
                                   data_to_load['od_val'], data_to_load['start_day_val'])
        self.valdataloader = DataLoader(valdataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list)

        helddataset = Dataset_list(data_to_load['held_xs'], data_to_load['nodes_held'],
                                  data_to_load['segment_travel_time_mean_held'],
                                  data_to_load['start_timestamp_held'], data_to_load['total_ts_held'],
                                  data_to_load['segment_travel_time_held'],
                                  data_to_load['segment_num_held'], data_to_load['ts_10min_held'],
                                  data_to_load['od_held'], data_to_load['start_day_held'])
        self.helddataloader = DataLoader(helddataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list)

        testdataset = Dataset_list(data_to_load['test_xs'], data_to_load['nodes_test'],
                                   data_to_load['segment_travel_time_mean_test'],
                                   data_to_load['start_timestamp_test'], data_to_load['total_ts_test'],
                                   data_to_load['segment_travel_time_test'],
                                   data_to_load['segment_num_test'], data_to_load['ts_10min_test'],
                                   data_to_load['od_test'], data_to_load['start_day_test'])
        self.testdataloader = DataLoader(testdataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list)

    def drl_path_prediction(self, n_epoch, lr):
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)
        early_stop = EarlyStop(mode='maximize', patience=self.early_stop)
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        print('drl path prediction start..')
        iter, train_loss_avg = 0, 0
        dir_check(ws + f"/model_params/drl_path_prediction/{local_time}/")
        self.model.train()
        try:
            for epoch in range(n_epoch):
                if early_stop.stop_flag: break
                with tqdm(self.traindataloader, total=len(self.traindataloader)) as t:
                    for i, batch in enumerate(t):
                        xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day = batch
                        loss = self.model(nodes, segment_num, od, start_day, ts_10min)
                        train_loss_avg += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        iter += 1
                # validation
                lcs_list = []
                dtw_list = []
                max_lcs = 0.
                max_dtw = 0.
                paths_planned_val = []
                for batch in tqdm(self.valdataloader):
                    xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day = batch
                    oris = od[:, 0].tolist()
                    dests = od[:, 1].tolist()
                    batch_planned_path = self.model.inference(oris, dests, start_day, ts_10min)
                    with torch.autograd.no_grad():
                        paths_planned_val.extend(batch_planned_path)
                    mean_lcs = 0.
                    mean_dtw = 0.
                    for k, (planned_path, ground, seg_num_path) in enumerate( zip(batch_planned_path, nodes, segment_num)):
                        lcs_path = LCSSDistance(planned_path, ground.tolist()[:int(seg_num_path.item())])
                        dtw_path = DTWDistance(self.model.G, planned_path,  ground.tolist()[:int(seg_num_path.item())])
                        mean_dtw += dtw_path
                        max_dtw = max(max_dtw, dtw_path)
                        mean_lcs += lcs_path
                        max_lcs = max(max_lcs, lcs_path)
                    mean_lcs /= len(nodes)
                    mean_dtw /= len(nodes)
                    lcs_list.extend([mean_lcs])
                    dtw_list.extend([mean_dtw])
                print(f"epoch: {epoch}, val mean LCS: {np.mean(lcs_list)}, val max LCS: {max_lcs}",
                      f"val mean DTW: {np.mean(dtw_list)}", f"val max DTW: {max_dtw}")
                is_best_change = early_stop.append(np.mean(lcs_list))
                if is_best_change:
                    best_valmodel_name = ws + f"/model_params/drl_path_prediction/{local_time}/finished_{epoch}.pth"
                    torch.save(self.model.state_dict(), best_valmodel_name)
                    self.model.load_state_dict(torch.load(best_valmodel_name))
                    print('best val model saved and loaded')

                    print(f"best model saved at: {best_valmodel_name}")
            # testing
            print('testing...')
            print('loaded best val model at: ', best_valmodel_name)
            self.model.load_state_dict(torch.load(best_valmodel_name))
            print('testing, best val model loaded')
            lcs_list = []
            dtw_list = []
            max_lcs = 0.
            max_dtw = 0.
            paths_planned_test = []
            for batch in tqdm(self.testdataloader):
                xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day = batch
                oris = od[:, 0].tolist()
                dests = od[:, 1].tolist()
                batch_planned_path = self.model.inference(oris, dests, start_day, ts_10min)
                with torch.autograd.no_grad():
                    paths_planned_test.extend(batch_planned_path)
                mean_lcs = 0.
                mean_dtw = 0.
                for k, (planned_path, ground, seg_num_path) in enumerate(zip(batch_planned_path, nodes, segment_num)):
                    lcs_path = LCSSDistance(planned_path, ground.tolist()[:int(seg_num_path.item())])
                    dtw_path = DTWDistance(self.model.G, planned_path, ground.tolist()[:int(seg_num_path.item())])
                    mean_dtw += dtw_path
                    max_dtw = max(max_dtw, dtw_path)
                    mean_lcs += lcs_path
                    max_lcs = max(max_lcs, lcs_path)
                mean_lcs /= len(nodes)
                mean_dtw /= len(nodes)
                lcs_list.extend([mean_lcs])
                dtw_list.extend([mean_dtw])
            print(f"test mean LCS: {np.mean(lcs_list)}, test max LCS: {max_lcs}",
                  f"test mean DTW: {np.mean(dtw_list)}", f"test max DTW: {max_dtw}")
        except KeyboardInterrupt as E:
            print("Interruptted")

    def load_model_for_inference(self):
        """
        specify the best_model_path for inference
        """

        best_model_path = ws + '/results/drl_path_prediction/drl_for_path_prediction.pth'

        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)

        self.model.load_state_dict(torch.load(best_model_path))

        print('best model loaded')
        """
        generate test set
        """
        dataloader_idx = 0
        for dataloder in [self.traindataloader, self.valdataloader, self.helddataloader, self.testdataloader]:
            if dataloader_idx == 0:
                set = 'train'
            elif dataloader_idx == 1:
                set = 'val'
            elif dataloader_idx == 2:
                set = 'held'
            elif dataloader_idx == 3:
                set = 'test'

            lcs_list = []
            dtw_list = []
            paths_planned = []
            max_lcs = 0.
            max_dtw = 0.
            self.model.eval()
            with torch.autograd.no_grad():
                for batch in tqdm(dataloder):
                    xs, nodes, segment_travel_time_mean, start_timestamp, total_ts, segment_travel_time, segment_num, ts_10min, od, start_day = batch
                    oris = od[:, 0].tolist()
                    dests = od[:, 1].tolist()
                    batch_planned_path = self.model.inference(oris, dests, start_day, ts_10min)
                    with torch.autograd.no_grad():
                        paths_planned.extend(batch_planned_path)
                    mean_lcs = 0.
                    mean_dtw = 0.
                    for k, (planned_path, ground, seg_num_path) in enumerate(zip(batch_planned_path, nodes, segment_num)):
                        lcs_path = LCSSDistance(planned_path, ground.tolist()[:int(seg_num_path.item())])
                        dtw_path = DTWDistance(self.model.G, planned_path, ground.tolist()[:int(seg_num_path.item())])
                        mean_dtw += dtw_path
                        max_dtw = max(max_dtw, dtw_path)
                        mean_lcs += lcs_path  # 一条路线的lcs
                        max_lcs = max(max_lcs, lcs_path)
                    mean_lcs /= len(nodes)
                    mean_dtw /= len(nodes)
                    lcs_list.extend([mean_lcs])
                    dtw_list.extend([mean_dtw])
            print(f"best model loaded for {set}_set")
            print(f"mean LCS: {np.mean(lcs_list)}, test max LCS: {max_lcs}",
                  f" mean DTW: {np.mean(dtw_list)}", f" max DTW: {max_dtw}")
            print('data all evaluated...')
            save_path = ws + f'/results/drl_generated_paths/{local_time}/'
            dir_check(save_path)
            planned_paths_path = ws +  f'/results/drl_generated_paths/{local_time}/' + f'planned_path_drl_{set}.npy'
            max_length = max(len(sublist) for sublist in paths_planned)
            padded_paths_planned = [sublist + [0] * (max_length - len(sublist)) for sublist in paths_planned]
            np.save(planned_paths_path, np.array(padded_paths_planned))
            print(f'{set}_set saved at: ', planned_paths_path)
            dataloader_idx += 1

class EarlyStop():

    def __init__(self, mode='maximize', patience=1):
        self.mode = mode
        self.patience = patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1
        self.is_best_change = False

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
