import numpy as np
import pickle
import os
from os.path import join
import pandas as pd
import multiprocessing
from tqdm import tqdm
import math
from datetime import datetime, timedelta, timezone
from functools import partial
from loader.preprocess.mm.utils import gcj02_to_wgs84

def convert_to_trajectory(group):
    trajectory = []
    for traj_id, time, lng, lat in group.values:
        # lng, lat = gcj02_to_wgs84(lng, lat)
        trajectory.append((int(time), lat, lng, traj_id))
    return trajectory


def convert_single(group, time_zone):
    group = group.sort_values(by=2).reset_index()
    group = group.drop(columns="index", axis=1, inplace=False)
    beg, end = group.index[0], group.index[-1]
    duration = group.at[end, 2] - group.at[beg, 2]
    # if duration <= 300 or duration > 7200:
    if duration <= 320 or duration > 3200:
        return None
    init_timestamp = int(group.at[beg, 2])
    finish_timestamp = int(group.at[end, 2])
    init_dt = datetime.fromtimestamp(init_timestamp, time_zone)
    finish_dt = datetime.fromtimestamp(finish_timestamp, time_zone)
    if init_dt.day != finish_dt.day:
        return None
    return convert_to_trajectory(group)

def get_trajectories(date, raw_traj_path):
    print(f"processing begins...")
    # data = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/data/tmp/hz_one_region_traj.h5', key='df')
    # data = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/data/tmp/hz_2b4106_traj.h5', key='df')
    # data = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/raw/20181101_20181116_chengdu_0606.h5', key = 'df')
    # data = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/raw/chengdu_0703_test.h5', key='df')
    # data = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/raw/chengdu_0705_test.h5', key='df')
    # data = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d1289_h891011.h5', key='df')
    # data = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d1289_h891011_after830.h5', key='df')

    # data = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d1289_h891011.h5',
    #                    key='df')
    # '/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d12897141516_h89101112.h5'
    # data = pd.read_hdf( '/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d12897141516_h89101112.h5', key='df')
    # data = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d12897141516_h89101112_after820.h5', key = 'df')
    data = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu/chengdu_d10131415161720_h9101112131415.h5', key = 'df')
    data = data.iloc[:, [0, 5, 3, 4]]
    # data = data.iloc[:, [0,1,2,3]]
    print("read raw trajectories complete!")
    
    traj_grouped = data.groupby(by=[1], axis=0)
    n_process = min(int(os.cpu_count()) + 1, 30)

    time_zone = timezone(timedelta(hours=0))
    partialprocessParallel = partial(convert_single, time_zone=time_zone)
    with multiprocessing.Pool(n_process) as pool:
        results = list(tqdm(pool.imap(partialprocessParallel, [group for _, group in traj_grouped]), total=len(traj_grouped), ncols=80))
    trajectories = [each for each in results if each is not None]
    return trajectories
