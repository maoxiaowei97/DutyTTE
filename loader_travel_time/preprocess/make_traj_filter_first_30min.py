import pandas as pd
# df = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/20181101_20181116_chengdu_0606.h5', 'df')
# df['datetime'] = pd.to_datetime(df.iloc[:, 1], unit='s')
# df['day'] = df['datetime'].dt.day
# df['hour'] = df['datetime'].dt.hour
# df = df[((df['day'] == 1) | (df['day'] == 2) | (df['day'] == 8) | (df['day'] == 9)) & ( (df['hour'] == 10) | (df['hour'] == 11) | (df['hour'] == 8) | (df['hour'] == 9) )]
# # df = df[((df['day'] == 1) | (df['day'] == 2)) & ( (df['hour'] == 10) | (df['hour'] == 11) )]
# # df = df[:100000]
# df.to_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d1289_h891011.h5', key='df', mode='w')
loaded_data = pd.read_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d12897141516_h89101112.h5', key='df')
filtered_data = loaded_data[loaded_data['datetime'].dt.time >= pd.to_datetime('08:20:00').time()]
filtered_data.to_hdf('/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw/chengdu_d12897141516_h89101112_after820.h5', key='df', mode='w')
print('number of traj points: ', len(filtered_data))
# print('number of trajs: ', len(df.groupby(1)))

# data = pd.read_csv(open('/data/MaoXiaowei/ODTUQ/ODTETA/processed_data/real2/raw/gps_20161001', "r"), header=None, sep=',')
# data = data.drop(columns=[0], axis=1, inplace=False)
# selected_columns = df.iloc[:, [0, 6, 4, 3]]
# selected_columns.columns = [1, 2, 3, 4]
# selected_columns.to_hdf('/data/MaoXiaowei/ODTUQ/ODTETA/processed_data/real/raw/20181101_20181116_chengdu_0606.h5', key='df', mode='w')
# loaded_data = pd.read_hdf('/data/MaoXiaowei/ODTUQ/ODTETA/processed_data/real/raw/20181101_20181116_chengdu_0606.h5', key='df')
# print('length of data:', len(loaded_data))