import matplotlib.pyplot as plt 
import torch
plt.switch_backend("agg")
from loader_travel_time.preprocess.mm.refine_gps import get_trajectories
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
import multiprocessing
from multiprocessing import cpu_count
from tqdm import tqdm
import os
from os.path import join
import pickle
import h5py
import numpy as np
import networkx as nx
import datetime
from datetime import timedelta, timezone
def map_single(trajectory, map_con):
    path = [[each[1], each[2]] for each in trajectory]
    start_timestamp = trajectory[0][0]
    end_timestamp = trajectory[-1][0]
    total_travel_time = end_timestamp - start_timestamp # seconds

    datetime_obj = datetime.datetime.fromtimestamp(start_timestamp, timezone(timedelta(hours=0)))
    hour = datetime_obj.hour
    minute = datetime_obj.minute
    hour_slot = hour * 3 # 20分钟, 现在时间片粒度是20分钟
    minute_slot = minute // 20
    start_time_slice = hour_slot + minute_slot # 出发时间片，匹配交通状态时要-1

    matcher = DistanceMatcher(map_con,
                              max_dist=400,
                              max_dist_init=400,
                              min_prob_norm=0.2,
                              obs_noise=100,
                              obs_noise_ne=100,
                              dist_noise=100,
                              max_lattice_width=20,
                              non_emitting_states=False)
    
    states, match_length = matcher.match(path)
    if len(states) < len(path):
        return None
    # shrink
    states_shrinked = [states[0]]
    link_points = [states[0][0], states[0][1]]
    states_to_point = [[trajectory[0]]]
    for i in range(1, len(states)):
        if states[i - 1][0] != states[i][0] or states[i - 1][1] != states[i][1]: # 下一位置的前缀节点或后缀节点与上一位置相应的前缀或后缀不同
            assert states[i - 1][1] == states[i][0]
            link_points.append(states[i][1])
            states_shrinked.append(states[i])
            states_to_point.append([trajectory[i]])
        else:
            states_to_point[-1].append(trajectory[i])

    states_non_loop = []
    node_states = [a for a, b in states_shrinked] + [states_shrinked[-1][1]] # 经过所有节点
    # filter loops
    show_pos = dict()
    for a in node_states:    
        if a not in show_pos:
            show_pos[a] = len(states_non_loop)
            states_non_loop.append(a)
        else:
            for k in range(len(states_non_loop) - 1, show_pos[a], -1):
                last = states_non_loop.pop()
                show_pos.pop(last)
    if len(states_non_loop) < 5:
        return None
    return (link_points, states_shrinked, states_to_point,
            [start_timestamp for _ in range(len(states_non_loop))],
            [total_travel_time for _ in range(len(states_non_loop))],
            [start_time_slice for _ in range(len(states_non_loop))],
            states_non_loop) # states_non_loop 经过路段
# link_points: 不重复的路段编号，states_shrinked: 前缀后缀路段，states_to_point: 各个路段及其对应的gps点，states_non_loop无环路段

def map_batch(pid, trajectories, city, map_path):
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl"))
    trajectories_mapped = []
    for i in tqdm(range(len(trajectories)), ncols=80, position=pid):
        states_to_point_idx_states = map_single(trajectories[i], map_con)
        if states_to_point_idx_states:
            trajectories_mapped.append(states_to_point_idx_states)
    return trajectories_mapped

def mapmatching(date, city, raw_traj_path, map_path):
    trajectories = get_trajectories(date, raw_traj_path)
    trajectories_mapped = []
    
    n_process = min(int(cpu_count()) + 1, 20)
    trajectories_mapped_batch_mid = []
    with multiprocessing.Pool(processes=n_process) as pool:
        err = lambda err: print(err)
        batch_size = (len(trajectories) + n_process - 1) // n_process
        
        for i in range(0, len(trajectories), batch_size):
            pid = i // batch_size
            trajectory_mapped_batch = pool.apply_async(map_batch, (pid,trajectories[i: i + batch_size],city,map_path,), error_callback=err)
            trajectories_mapped_batch_mid.append(trajectory_mapped_batch)

        for each in trajectories_mapped_batch_mid:
            trajectories_mapped.extend(each.get())
        return trajectories_mapped

def get_matched_path(date, city, traj_path, map_path, raw_path):
    target_path = join(traj_path, f"traj_mapped_{city}_{date}.pkl")
    # if os.path.exists(target_path):
    #     print("loading...")
    #     return pickle.load(open(target_path, "rb"))
    trajectories_mapped = mapmatching(date, city, raw_path, map_path)
    print("writing...")
    pickle.dump(trajectories_mapped, open(target_path, "wb"))
    print("write complete!")
    return trajectories_mapped

def process_gps_and_graph(city, map_path, data_path, raw_path, traj_path):
    name = city
    map_con = InMemMap.from_pickle(join(map_path, f"map_{city}.pkl"))
    # calculate G and A
    target_g_path = join(data_path, f"{name}_G.pkl")
    G = nx.Graph()
    node_attrs = [(cid, {"lat": lat, "lng": lng}) for cid, (lat, lng) in map_con.all_nodes()]
    G.add_nodes_from(node_attrs)
    G.add_edges_from([(a, b) for a, _, b, _ in map_con.all_edges()])
    n = G.number_of_nodes()
    A = torch.zeros([n, n], dtype=torch.float64)
    for a, b in G.edges:
        A[a, b] = 1.
        A[b,a] = 1.
    pickle.dump(G, open(target_g_path, "wb"))
    torch.save(A, join(data_path, f"{name}_A.ts"))
    h5_file = join(data_path, f"{name}_h5_paths.h5")
    date = '2018'
    with h5py.File(h5_file, "w") as f:
        print("#####", date)
        f.create_group(date)
        trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)
        # shrink
        state_lengths, states, start_timestamp, start_ts, total_ts  = [], [], [], [], []

        for link_points, states_shrinked, states_to_point, start_travel_timestamp, total_travel_time, start_travel_time_slice, states_non_loop in trajectories_mapped:
            state_lengths.append(len(states_non_loop))
            states.extend(states_non_loop)
            start_timestamp.extend(start_travel_timestamp)
            start_ts.extend(start_travel_time_slice)
            total_ts.extend(total_travel_time)

        # calcluate prefix sum
        state_prefix = np.zeros(shape=len(state_lengths) + 1, dtype=np.int64)
        for k, L in enumerate(state_lengths):
            state_prefix[k + 1] = state_prefix[k] + L

        # length_info
        # pad all in one
        f[date].create_dataset("state_prefix", data=np.array(state_prefix))
        f[date].create_dataset("states", data=np.array(states))
        f[date].create_dataset("states_timestamp", data=np.array(start_timestamp))
        f[date].create_dataset("start_ts", data=np.array(start_ts))
        f[date].create_dataset("total_ts", data=np.array(total_ts))
    
    # calculate V
    target_v_path = join(data_path, f"{name}_v_paths.csv")
    vs = []
    trajectories_mapped = get_matched_path(date, city, traj_path, map_path, raw_path)
    non_loops  = [each[0] for each in trajectories_mapped]
    n_samples = len(non_loops)
    v_np = np.zeros([n_samples, n])
    for k, non_loop in enumerate(non_loops):
        v_np[k, non_loop] = 1.
        v_np[k, non_loop[0]] = 2.
    vs.append(v_np)
    # generate V
    v_data = np.concatenate(vs, axis=0)
    np.savetxt(target_v_path, v_data, delimiter=',', fmt='%d')