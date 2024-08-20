import torch
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy.stats import binom
from scipy.optimize import brentq
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import time
from tqdm import tqdm
import pickle
from utils.argparser import ws

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
    xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = zip( *batch)
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


def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))

### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.minimum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

### UCB of mean via Hoeffding-Bentkus hybridization
def HB_mu_plus(muhat, n, delta, maxiters=1000):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        try:
          return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)
        except:
          print(f"BRENTQ RUNTIME ERROR at muhat={muhat}")
          return 1.0


def get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam, device):
  losses = []
  dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
  model = model.to(device)
  for batch in dataloader:
    x, labels = batch
    sets = model.nested_sets_from_output(x.to(device),lam)
    losses = losses + rcps_loss_fn(sets, labels.to(device)).tolist()
  return torch.FloatTensor(losses)

def fraction_missed_loss(pset, label):
    predict_lower = pset[1]
    predict_upper = pset[2]
    misses = (predict_lower > label.reshape(-1)).float() + ( predict_upper < label.reshape(-1)).float()
    misses[misses > 1.0] = 1.0
    return misses


def get_rcps_loss_fn():
    return fraction_missed_loss

def calibrate_model(model, outputs, labels, config, device):
    with torch.no_grad():
        print(f"Calibrating...")
        model.eval()
        alpha = config.alpha
        delta = config.delta
        device = device
        print("Initialize lambdas")

        lambdas = torch.linspace(config.minimum_lambda, config.maximum_lambda, config.num_lambdas)

        print("Initialize loss")
        rcps_loss_fn = get_rcps_loss_fn()
        print("Put model on device")
        model = model.to(device)
        print("Initialize labels")

        print("Output dataset")
        out_dataset = TensorDataset(torch.FloatTensor(outputs), torch.FloatTensor(labels))
        dlambda = lambdas[1] - lambdas[0] # delta lambda
        model.set_lhat(lambdas[-1] + dlambda - 1e-9)
        print("Computing losses")
        calib_loss_table = torch.zeros((outputs.shape[0], lambdas.shape[0]))
        for lam in tqdm(reversed(lambdas)):
            losses = get_rcps_losses_from_outputs(model, out_dataset, rcps_loss_fn, lam - dlambda, device)
            calib_loss_table[:, np.where(lambdas == lam)[0]] = losses[:, None]
            Rhat = losses.mean()
            RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], delta)
            print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ", end='')
            if Rhat >= alpha or RhatPlus > alpha:
                model.set_lhat(lam)
                print("calibration done")
                print(f"Model's lhat set to {model.lhat}")
                print('lambda: ', lam)
                point_estimation = outputs[:, 0]
                upper_edge = lam * outputs[:, 2] + point_estimation
                lower_edge = point_estimation -  outputs[:, 1]
                calibrated_width = upper_edge - lower_edge
                calibrated_picp = picp(point_estimation.reshape(-1), outputs[:, 1].reshape(-1), (lam.item() * outputs[:, 2]).reshape(-1), labels.reshape(-1))
                print('calibrated width: ' + str(np.mean(calibrated_width.reshape(-1).numpy())))
                print('calibrated picp: ' + str(calibrated_picp))
                break
        return model, calib_loss_table


class ModelWithUncertainty(nn.Module):
    def __init__(self, UQModel, in_nested_sets_from_output_fn, params):
        super(ModelWithUncertainty, self).__init__()
        self.UQModel = UQModel
        self.register_buffer('lhat', None)
        self.in_nested_sets_from_output_fn = in_nested_sets_from_output_fn
        self.params = params

    def nested_sets_from_output(self, output, lam=None):
        prediction, lower_edge, upper_edge = self.in_nested_sets_from_output_fn(self, output, lam)

        return prediction, lower_edge, upper_edge

    def set_lhat(self, lhat):
        self.lhat = lhat

def mis_regression_nested_sets_from_output(model, output, lam=None):

    if lam == None:
        if model.lhat == None:
            raise Exception("You have to specify lambda unless your model is already calibrated.")
        lam = model.lhat

    point_estimation = output[:, 0]
    upper_edge = lam * output[:, 2] + point_estimation
    lower_edge =  point_estimation - lam * output[:, 1]

    return point_estimation, lower_edge, upper_edge

def add_uncertainty(uq_model, params):
    nested_sets_from_output_fn = mis_regression_nested_sets_from_output

    return ModelWithUncertainty(uq_model, nested_sets_from_output_fn, params)

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

class CalibrateModel:
    def __init__(self, model: nn.Module, dataset, device, args):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.rho = 0.1

    def calibrate(self, args):
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print('local time: ', local_time)
        """
        1.shrinked travel time
        """
        shrinked_segment_travel_time_path = ws + '/processed_data/CityA_segment_travel_time_distribution_dict_shrinked.npy'
        segment_travel_time_dict = np.load(shrinked_segment_travel_time_path, allow_pickle=True).item()
        """
        2.shrinked segment index
        """
        shinked_segment_index_dict_path = ws + '/processed_data/CityA_segment_dict_shrinked.pkl'
        with open(shinked_segment_index_dict_path, 'rb') as f:
            segment_index_dict = pickle.load(f)

        import os
        pickle_file_path = ws + '/processed_data/train_val_test_held_subset.pkl'
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                data_to_load = pickle.load(f)
        print('loaded train_val_test_held data: ', pickle_file_path)

        """
        specify the path of predicted paths
        """

        held_generated_nodes = np.load(ws + '/results/drl_generated_paths/planned_path_drl_held.npy',  allow_pickle=True).tolist()
        test_generated_nodes = np.load( ws + '/results/drl_generated_paths/planned_path_drl_test.npy', allow_pickle=True).tolist()

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

        """
        generated held set
        """
        held_ground_truth_nodes = data_to_load['nodes_held']
        held_ground_truth_segments_num = data_to_load['segment_num_held']
        held_ground_truth_od = data_to_load['od_held']

        held_generated_segment_travel_time_distribution = []
        held_generated_segment_num = []
        held_generated_segment_travel_time_mean = []
        held_generated_segments = []
        held_generated_total_ts = []
        held_generated_ts_10min = []
        held_generated_od = []

        for k, (route, real_route, real_segment_num, real_od, start_10min_ts, start_day, total_ts, ts_10min) in enumerate(
                tqdm(zip(held_generated_nodes, held_ground_truth_nodes, held_ground_truth_segments_num,
                         held_ground_truth_od, data_to_load['ts_10min_held'], data_to_load['start_day_held'],
                         data_to_load['total_ts_held'], data_to_load['ts_10min_held']))):
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
            held_generated_segment_num.append([node_num - 1])
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
            held_generated_segment_travel_time_mean.append(path_generated_segment_travel_time_mean)
            held_generated_segment_travel_time_distribution.append(path_generated_segment_travel_time_distribution)
            held_generated_segments.append(path_generated_segments)
            held_generated_total_ts.append(path_generated_total_ts[0])
            held_generated_ts_10min.append(path_generated_ts_10min[0])
            held_generated_od.append(path_generated_od[0])

        helddataset = Dataset_list(held_generated_segments, held_generated_segment_travel_time_mean,
                                   held_generated_total_ts, held_generated_segment_travel_time_distribution,
                                   held_generated_segment_num, held_generated_ts_10min, held_generated_od)

        helddataloader = DataLoader(helddataset, batch_size=128, shuffle=False, collate_fn=collate_fn_list,  drop_last=True)

        predicts = []
        predicts_bias_lower = []
        predicts_bias_upper = []
        label = []
        model_path = ws + ''
        try:
            self.model.load_state_dict(torch.load(model_path))
        except KeyboardInterrupt as E:
            print("Please specify the path of a well trained MoEUQ model here")
        print('uq model loaded')

        """
        calibration
        """
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(helddataloader):
                xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                predict_mean, bias_lower, bias_upper = self.model(xs, segment_travel_time, segment_num, ts_10min, od, self.device)
                total_ts = pad_sequence(total_ts, batch_first=True, padding_value=0).float()

                predicts += predict_mean.reshape(-1).tolist()
                predicts_bias_lower += bias_lower.reshape(-1).tolist()
                predicts_bias_upper += bias_upper.reshape(-1).tolist()
                label += total_ts.reshape(-1).float().tolist()

            predicts = np.array(predicts).reshape(-1)
            label = np.array(label).reshape(-1, 1)
            predicts_bias_lower = np.array(predicts_bias_lower).reshape(-1)
            predicts_bias_upper = np.array(predicts_bias_upper).reshape(-1)
            test_width = predicts_bias_upper + predicts_bias_lower
            test_picp = picp(predicts, predicts_bias_lower, predicts_bias_upper, label.reshape(-1))
            print('picp before calibration: ', test_picp)
            print('width before calibration: ', np.mean(test_width))
            outputs = np.concatenate([predicts.reshape(-1, 1), predicts_bias_lower.reshape(-1, 1), predicts_bias_upper.reshape(-1, 1)], axis=1) # 均值，下界，上界

        uq_model = add_uncertainty(self.model, args)
        model, _ = calibrate_model(uq_model, outputs, label, args, self.device)
        print(f"Model calibrated! lambda hat = {uq_model.lhat}")

        """
        test
        """
        predicts = []
        predicts_bias_lower = []
        predicts_bias_upper = []
        label = []
        print('testing')
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(testdataloader):
                xs, segment_travel_time_mean, total_ts, segment_travel_time, segment_num, ts_10min, od = batch
                predict_mean, bias_lower, bias_upper = self.model(xs, segment_travel_time, segment_num,   ts_10min, od, self.device)
                total_ts = pad_sequence(total_ts, batch_first=True, padding_value=0).float()

                predicts += predict_mean.reshape(-1).tolist()
                predicts_bias_lower += (bias_lower.reshape(-1) * uq_model.lhat).tolist()
                predicts_bias_upper += (bias_upper.reshape(-1) *  uq_model.lhat).tolist()
                label += total_ts.reshape(-1).float().tolist()

            predicts = np.array(predicts).reshape(-1)
            label = np.array(label).reshape(-1, 1)
            predicts_bias_lower = np.array(predicts_bias_lower).reshape(-1)
            predicts_bias_upper = np.array(predicts_bias_upper).reshape(-1)
            test_width = predicts_bias_upper + predicts_bias_lower
            test_picp = picp(predicts, predicts_bias_lower, predicts_bias_upper, label.reshape(-1))
            print('picp after calibration: ', test_picp)
            print('width after calibration: ', np.mean(test_width))
