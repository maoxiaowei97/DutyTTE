import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
MAKE_TRAJECTORY = False
if MAKE_TRAJECTORY:
    with open(f'/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/traj_mapped_chengdu_test_2018.pkl', 'rb') as f:
        traj_mapped = pickle.load(f)

    data_list = []

    # 遍历每一条轨迹
    for traj in traj_mapped:
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
    df.to_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/mapped_traj_details_test_cd.h5', key='df', mode='w')
else:
    df = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/mapped_traj_details_test_cd.h5', key='df')

"""
根据轨迹构建各个路段的通行时间字典
确定路段数量，天，时间片数量
"""
# 将时间戳转换为日期时间格式
df['时间戳'] = pd.to_datetime(df['时间戳'], unit='s')
df['日期'] = df['时间戳'].dt.date
df['时间片'] = df['时间戳'].dt.floor('15T') # 向下取整15分钟，匹配时，取开始时间对应的上一个15分钟时间片就行，统计的范围没问题

# 初始化通行时间字典
segment_travel_time_dict = {}

time_interval = 15 * 60  # 15 分钟时间片，单位为秒

for date in tqdm(df['日期'].unique(), total=len(df['日期'].unique())):
    for time_segment in tqdm(df['时间片'].unique(), total=len(df['时间片'].unique())):
        for start_segment in tqdm(df['起点路段编号'].unique(), total=len(df['起点路段编号'].unique())):
            # 筛选出在当前时间片和起点路段的记录
            mask = (df['日期'] == date) & (df['时间片'] == time_segment) & (df['起点路段编号'] == start_segment) # 挑出每个时间片，每个路段需要考虑的轨迹
            filtered_df = df[mask]
            # 如果没有记录，跳过
            if filtered_df.empty:
                continue
            # 计算通行时间
            travel_times = []
            grouped = filtered_df.groupby('轨迹编号')
            for traj_id, group in grouped:
                # 按时间排序
                group = group.sort_values(by='时间戳')

                # 如果只有一个轨迹点，通行时间记为 15s
                if len(group) == 1:
                    travel_times.append(15)
                    continue

                # 计算每个轨迹点的通行时间
                # for i in range(len(group) - 1):
                start_time = group.iloc[0]['时间戳']
                # end_time = group.iloc[i + 1]['时间戳']
                end_time = group.iloc[-1]['时间戳']
                travel_time = (end_time - start_time).total_seconds()
                travel_times.append(travel_time)

            # 提取小时和分钟
            hour = time_segment.hour
            minute = time_segment.minute

            # 计算小时部分的时间片数，每小时4个时间片
            hour_slot = hour * 4

            # 计算分钟部分的时间片数，每15分钟一个时间片
            minute_slot = minute // 15

            # 总的时间片编号
            time_slot_number = hour_slot + minute_slot
            day = date.day
            # 将结果存入字典
            key = (day, time_slot_number, start_segment)
            if key not in segment_travel_time_dict:
                segment_travel_time_dict[key] = []
            segment_travel_time_dict[key].extend(travel_times)

# 打印或返回通行时间字典
np.save('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/segment_travel_time_dict.npy', segment_travel_time_dict)
print('saved...')