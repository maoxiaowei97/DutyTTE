import pickle
import swifter
import numpy as np
import pandas as pd
from tqdm import tqdm
MAKE_TRAJECTORY = False
MAKE_DICT = True
# segment_travel_time_dict = np.load('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/cd_2018_segment_travel_time_dict.npy', allow_pickle=True).item()
# c = segment_travel_time_dict
if MAKE_TRAJECTORY:
    with open(f'/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/traj_mapped_chengdu_2018_2018.pkl', 'rb') as f:
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
    df.to_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/cd_2018_mapped_traj_details.h5', key='df', mode='w')
else:
    df = pd.read_hdf('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/cd_2018_mapped_traj_details.h5', key='df')

##test
df = df[:10000]
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
# 预处理时间片，起点路段编号的唯一值
"""
加速后的代码
"""
# 预处理时间片，起点路段编号的唯一值
unique_dates = df['日期'].unique()
unique_time_segments = df['时间片'].unique()
unique_start_segments = df['起点路段编号'].unique()

# 预先创建一个字典以加速查找
date_dict = {date: idx for idx, date in enumerate(unique_dates)}
time_segment_dict = {time_segment: idx for idx, time_segment in enumerate(unique_time_segments)}
start_segment_dict = {start_segment: idx for idx, start_segment in enumerate(unique_start_segments)}

# 创建一个新的列用于加速筛选
df['日期_idx'] = df['日期'].map(date_dict)
df['时间片_idx'] = df['时间片'].map(time_segment_dict)
df['起点路段编号_idx'] = df['起点路段编号'].map(start_segment_dict)


# 定义计算通行时间的函数
def calculate_travel_times(group):
    group = group.sort_values()  # 直接对 Series 进行排序
    if len(group) == 1:
        return [15]
    start_time = group.iloc[0]
    end_time = group.iloc[-1]
    travel_time = (end_time - start_time).total_seconds()
    return [travel_time for _ in range(len(group))]

# 假设 df 已经定义并包含所有必要的列
df['travel_times'] = df.groupby(['轨迹编号', '起点路段编号_idx'])['时间戳'].transform(lambda x: calculate_travel_times(x))

# 将结果存入字典
for date_idx in tqdm(range(len(unique_dates)), total=len(unique_dates)):
    for time_segment_idx in tqdm(range(len(unique_time_segments)), total=len(unique_time_segments)):
        for start_segment_idx in tqdm(range(len(unique_start_segments)), total=len(unique_start_segments)):
            mask = (df['日期_idx'] == date_idx) & (df['时间片_idx'] == time_segment_idx) & (df['起点路段编号_idx'] == start_segment_idx)
            filtered_df = df[mask]
            if filtered_df.empty:
                continue
            # 获取所有 travel_times 列表并展平，先求个平均吧，可能包含多条轨迹
            travel_times = filtered_df['travel_times'].unique().tolist() # 每段轨迹在该路段的通行时间

            time_segment = unique_time_segments[time_segment_idx]
            hour = time_segment.hour
            minute = time_segment.minute
            hour_slot = hour * 4
            minute_slot = minute // 15
            time_slot_number = hour_slot + minute_slot
            day = unique_dates[date_idx].day

            key = (day, time_slot_number, unique_start_segments[start_segment_idx])
            if key not in segment_travel_time_dict:
                segment_travel_time_dict[key] = []
            segment_travel_time_dict[key].extend(travel_times)
"""
下面是之前的代码，效率太低
"""

# if MAKE_DICT:
#     for date in tqdm(df['日期'].unique(), total=len(df['日期'].unique())):
#         for time_segment in tqdm(df['时间片'].unique(), total=len(df['时间片'].unique())):
#             for start_segment in tqdm(df['起点路段编号'].unique(), total=len(df['起点路段编号'].unique())):
#                 # 筛选出在当前时间片和起点路段的记录
#                 mask = (df['日期'] == date) & (df['时间片'] == time_segment) & (df['起点路段编号'] == start_segment) # 挑出每个时间片，每个路段需要考虑的轨迹
#                 filtered_df = df[mask]
#                 # 如果没有记录，跳过
#                 if filtered_df.empty:
#                     continue
#                 # 计算通行时间
#                 travel_times = []
#                 grouped = filtered_df.groupby('轨迹编号')
#                 for traj_id, group in grouped:
#                     # 按时间排序
#                     group = group.sort_values(by='时间戳')
#
#                     # 如果只有一个轨迹点，通行时间记为 15s
#                     if len(group) == 1:
#                         travel_times.append(15)
#                         continue
#
#                     start_time = group.iloc[0]['时间戳']
#                     end_time = group.iloc[-1]['时间戳']
#                     travel_time = (end_time - start_time).total_seconds()
#                     travel_times.append(travel_time)
#
#                 # 提取小时和分钟
#                 hour = time_segment.hour
#                 minute = time_segment.minute
#
#                 # 计算小时部分的时间片数，每小时4个时间片
#                 hour_slot = hour * 4
#
#                 # 计算分钟部分的时间片数，每15分钟一个时间片
#                 minute_slot = minute // 15
#
#                 # 总的时间片编号
#                 time_slot_number = hour_slot + minute_slot
#                 day = date.day
#                 # 将结果存入字典
#                 key = (day, time_slot_number, start_segment)
#                 if key not in segment_travel_time_dict:
#                     segment_travel_time_dict[key] = []
#                 segment_travel_time_dict[key].extend(travel_times)

# 打印或返回通行时间字典
np.save('/data/HuangYiheng/test/DiffRoute/processed_data/trajectories/cd_2018_segment_travel_time_dict.npy', segment_travel_time_dict)
print('saved...')