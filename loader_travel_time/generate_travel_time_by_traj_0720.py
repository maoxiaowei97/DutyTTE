"""
1.统计各轨迹在各路段通行时间
2.时间片粒度为20分钟
"""
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
MAKE_TRAJECTORY = True
MAKE_DICT = True
FILL_TRAVEL_TIME_DICT = True
#构建通行时间字典时不用过滤830之前的，真正用的轨迹数据需要过滤830
if MAKE_TRAJECTORY:
    # '/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/traj_mapped_chengdu_d12131920_h89101112_2018.pkl'
    # with open(f'/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/traj_mapped_chengdu_d12897141516_h89101112_2018.pkl', 'rb') as f:
    with open(f'/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/traj_mapped_chengdu_d12131920_h89101112_2018.pkl', 'rb') as f:
        traj_mapped = pickle.load(f)

    data_list = []

    # 遍历每一条轨迹
    for traj in tqdm(traj_mapped, total=len(traj_mapped)):
        # 获取起点和终点路段信息的列表
        route_segments = traj[1]

        # 获取轨迹点信息的列表
        traj_points = traj[2]

        # 遍历每一个起点和终点路段
        for (start_segment, end_segment), points in zip(route_segments, traj_points):
            # 遍历每一个轨迹点
            for point in points:
                timestamp, latitude, longitude, traj_id = point
                data_list.append([traj_id, timestamp, longitude, latitude, start_segment, end_segment])

    # 创建 DataFrame
    df = pd.DataFrame(data_list, columns=['轨迹编号', '时间戳','轨迹点经度', '轨迹点纬度', '起点路段编号', '终点路段编号'])
    df.to_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_mapped_traj_details_d12131920_h89101112_2018.h5', key='df', mode='w')
else:
    df = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_mapped_traj_details_d12131920_h89101112_2018.h5', key='df')

"""
构造轨迹为键，起终点路段通行时间为值的字典
"""
# df = df[:20000]
if MAKE_DICT:
    df['时间戳'] = pd.to_datetime(df['时间戳'], unit='s')
    df['日期'] = df['时间戳'].dt.date
    df['时间片'] = df['时间戳'].dt.floor('20T') # 向下取整20分钟，匹配时，取开始时间对应的上一个30分钟时间片就行，统计的范围没问题
    segment_travel_time_dict = {}
    for key, group in tqdm(df.groupby(['轨迹编号', '起点路段编号', '终点路段编号'])):
        # if len(group) < 2: # 在路段上至少两个轨迹点才考虑
        #     continue
        day = int(group['时间片'].dt.day.iloc[-1])
        time_segment = group['时间片'].iloc[-1] # 完成时间片
        hour = time_segment.hour
        minute = time_segment.minute
        hour_slot = hour * 3  # 20分钟
        minute_slot = minute // 20  # 20分钟
        time_slot_number = hour_slot + minute_slot
        start_segment = group['起点路段编号'].iloc[-1]
        key = (day, time_slot_number, start_segment)
        total_travel_time = (group['时间戳'].iloc[-1] - group['时间戳'].iloc[0]).total_seconds() + 3
        if total_travel_time > 20 * 60:
            continue
        if key not in segment_travel_time_dict:
            segment_travel_time_dict[key] = []
        segment_travel_time_dict[key].extend([total_travel_time])

    np.save('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_segment_travel_time_dict_d12131920_h89101112_2018_20min_by_traj.npy', segment_travel_time_dict)
    print('saved...')
else:
    segment_travel_time_dict = np.load('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_segment_travel_time_dict_d12131920_h89101112_2018_20min_by_traj.npy', allow_pickle=True).item()

if FILL_TRAVEL_TIME_DICT:
    df['时间戳'] = pd.to_datetime(df['时间戳'], unit='s')
    df['日期'] = df['时间戳'].dt.date
    df['时间片'] = df['时间戳'].dt.floor('20T')  # 向下取整20分钟，匹配时，取开始时间对应的上一个30分钟时间片就行，统计的范围没问题
    df['天'] = df['时间片'].dt.day
    df['ts'] = df['时间片'].dt.hour * 3 + df['时间片'].dt.minute // 20
    """
    1.先构建总的路段通行时间字典：[路段编号，通行时间]
    """
    road_list = list(df['起点路段编号'].unique())
    day_list = list(df['天'].unique())
    ts_list = list(df['ts'].unique())
    road_avg_travel_time_dict = {}
    for key in segment_travel_time_dict:
        road = key[2]
        if road not in road_avg_travel_time_dict:
            road_avg_travel_time_dict[road] = []
            road_avg_travel_time_dict[road].extend(segment_travel_time_dict[key])
        else:
            road_avg_travel_time_dict[road].extend(segment_travel_time_dict[key])
    print('number of roads:', len(road_list))
    print('number of roads in avg_travel_time_dict:', len(road_avg_travel_time_dict))
    print('begin fill segment travel time dict')
    for day in day_list:
        for ts in ts_list:
            for road in road_list:
                key = (day, ts, road)
                if key in segment_travel_time_dict:
                    continue
                else:
                    if road in road_avg_travel_time_dict:
                        segment_travel_time_dict[key] = [np.mean(road_avg_travel_time_dict[road])]
                    else:
                        segment_travel_time_dict[key] = [30] # 指定30s

    print('segment travel time dict filled, number of keys:', len(segment_travel_time_dict))
    np.save('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_segment_travel_time_dict_d12131920_h89101112_2018_20min_by_traj.npy', segment_travel_time_dict)
    print('saved...')



"""
OD travel time dict, 键为起点，终点路段，值为历史平均通行时间
"""







