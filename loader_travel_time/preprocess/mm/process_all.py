from loader_travel_time.preprocess.mm.fetch_rdnet import fetch_map, build_map
from loader_travel_time.preprocess.mm.mapmatching import process_gps_and_graph

def process_main():
    # data_path = "/data/HuangYiheng/test/DiffRoute/processed_data"
    # city = "hangzhou_one_courier_2b4106_pickup"
    # # bounds = [104.0, 30.64, 104.15, 30.73]
    # bounds = [119.97758000000002, 30.31055200000002, 120.08713250000001, 30.43544000000001]
    # map_path = "/data/HuangYiheng/test/DiffRoute/processed_data/map"
    # fetch_map(city, bounds, map_path)
    # map_con = build_map(city, map_path)
    #
    # raw_path = "/data/HuangYiheng/test/DiffRoute/processed_data/raw"
    # traj_path = "/data/HuangYiheng/test/DiffRoute/processed_data/trajectories"
    # process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    data_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data"
    # city = "chengdu_2018"
    # city = "chengdu_d1289_h891011_after830"
    # city = "chengdu_d12897141516_h89101112" # 用于构建通行时间字典
    # city = "chengdu_d12897141516_h89101112_after820"
    city = "chengdu_d10131415161720_h9101112131415"
    bounds = [104.0, 30.64, 104.15, 30.73]
    map_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/map"
    fetch_map(city, bounds, map_path)
    map_con = build_map(city, map_path)

    raw_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw"
    traj_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories_cd_0726"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)

if __name__ == "__main__":

    data_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data"
    # city = "chengdu_d1289_h891011_after830"
    # city = "chengdu_d12897141516_h89101112_after820" # 用于训练模型
    city = "chengdu_d10131415161720_h9101112131415"
    bounds = [104.0, 30.64, 104.15, 30.73]
    map_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/map"
    fetch_map(city, bounds, map_path)
    map_con = build_map(city, map_path)

    raw_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/raw"
    traj_path = "/data/MaoXiaowei/ODTUQ_0702/DiffRoute/processed_data/trajectories_cd_0726"
    process_gps_and_graph(city, map_path, data_path, raw_path, traj_path)
    