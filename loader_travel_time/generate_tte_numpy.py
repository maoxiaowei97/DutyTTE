"""
numpy 加速通行时间字典生成，只需要用这个脚本，相邻其他两个脚本不用
"""
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
    with open(f'/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/traj_mapped_chengdu_d1289_h891011_2018.pkl', 'rb') as f:
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
    df.to_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_mapped_traj_details_d1289_h891011_2018.h5', key='df', mode='w')
else:
    df = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_mapped_traj_details_d1289_h891011_2018.h5', key='df')

"""
根据轨迹构建各个路段的通行时间字典
确定路段数量，天，时间片数量
按完成轨迹卡路段通行时间
"""
# 将时间戳转换为日期时间格式
df['时间戳'] = pd.to_datetime(df['时间戳'], unit='s')
df['日期'] = df['时间戳'].dt.date
df['时间片'] = df['时间戳'].dt.floor('30T') # 向下取整30分钟，匹配时，取开始时间对应的上一个30分钟时间片就行，统计的范围没问题
# 再构建一个出发时间片在8:30之后的轨迹样本
# 初始化通行时间字典
segment_travel_time_dict = {}

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
        return [0]
    start_time = group.iloc[0]
    end_time = group.iloc[-1]
    travel_time = (end_time - start_time).total_seconds()
    return [travel_time for _ in range(len(group))]

# df = df[:30000]
# 假设 df 已经定义并包含所有必要的列
print('start calc travel times of each road')
df['travel_times'] = df.groupby(['轨迹编号', '起点路段编号_idx', '时间片_idx'])['时间戳'].transform(lambda x: calculate_travel_times(x)) # 只计算在同一个时间片内 一个路段的通行时间，如果一段轨迹跨越多个时间片，则按时间片切分成两段子轨迹
print('travel times of each road calculated')

# 假设 df 已经定义并包含所有必要的列
# 将相关列转换为 NumPy 数组
date_idx_array = df['日期_idx'].to_numpy()
time_segment_idx_array = df['时间片_idx'].to_numpy()
start_segment_idx_array = df['起点路段编号_idx'].to_numpy()
travel_times_array = df['travel_times'].to_numpy()

# 初始化一个空的字典
segment_travel_time_dict = {}

# 假设 unique_dates, unique_time_segments, unique_start_segments 已经定义
# for date_idx in tqdm(range(len(unique_dates)), total=len(unique_dates)):
for time_segment_idx in tqdm(range(len(unique_time_segments)), total=len(unique_time_segments)):
    for start_segment_idx in tqdm(range(len(unique_start_segments)), total=len(unique_start_segments)):
        # 基于 NumPy 的掩码计算
        day = unique_time_segments[time_segment_idx].day
        if day == 1:
            date_idx = 0
        elif day == 2:
            date_idx = 1
        elif day == 8:
            date_idx = 2
        elif day == 9:
            date_idx = 3
        mask = (date_idx_array == date_idx) & (time_segment_idx_array == time_segment_idx) & (start_segment_idx_array == start_segment_idx)
        filtered_indices = np.where(mask)[0]  # 获取满足条件的索引
        if len(filtered_indices) == 0: # 没有符合时间范围的轨迹
            filtered_travel_times = [40]
        else:
            # 获取所有 travel_times 列表并展平，先求个平均吧，可能包含多条轨迹
            travel_times = list(set(travel_times_array[filtered_indices])) # 每段轨迹在该路段的通行时间，先不考虑经过一个路段多次的情况，因为路线输入去掉了loop, 少于实际经过路段数
            filtered_travel_times = np.array(travel_times)
            filtered_travel_times = filtered_travel_times[filtered_travel_times != 0]
            filtered_travel_times = filtered_travel_times.tolist()
            if len(filtered_travel_times) == 0:
                filtered_travel_times = [40]

        time_segment = unique_time_segments[time_segment_idx]
        hour = time_segment.hour
        minute = time_segment.minute
        # hour_slot = hour * 4
        # minute_slot = minute // 15
        hour_slot = hour * 2 # 30分钟
        minute_slot = minute // 30 # 30分钟
        time_slot_number = hour_slot + minute_slot
        key = (day, time_slot_number, unique_start_segments[start_segment_idx])
        if key not in segment_travel_time_dict:
            segment_travel_time_dict[key] = []
        segment_travel_time_dict[key].extend(filtered_travel_times)

# 打印或返回通行时间字典
np.save('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories/cd_segment_travel_time_dict_d1289_h891011_2018_30min.npy', segment_travel_time_dict)
print('saved...')