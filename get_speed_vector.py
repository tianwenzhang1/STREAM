# import math
# from collections import defaultdict
# import os
# import json
# import numpy as np
# import torch
# import torch.nn as nn
#
#
# def haversine_distance(lat1, lon1, lat2, lon2):
#     R = 6371  # 地球半径，单位：公里
#     phi1, phi2 = math.radians(lat1), math.radians(lat2)
#     delta_phi = math.radians(lat2 - lat1)
#     delta_lambda = math.radians(lon2 - lon1)
#     a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c
#
# def cal_cur_traj(speeds, cur_trajectory):
#     for i in range(1, len(cur_trajectory)):
#         t1, lat1, lng1, road_id1 = cur_trajectory[i - 1]
#         t2, lat2, lng2, road_id2 = cur_trajectory[i]
#
#         # 计算两点之间的距离
#         distance = haversine_distance(lat1, lng1, lat2, lng2)
#         time_diff = (t2 - t1) / 3600  # 时间差，单位为小时
#
#         if time_diff > 0:
#             speed = distance / time_diff  # 计算速度，单位为公里/小时
#             time_slot = int((t1 - timestamp_0) / (time_interval * 60))  # 确定时间槽
#
#             if road_id1 == road_id2:
#                 # 如果两个点在同一条路段，则直接将速度加到该路段的时间槽
#                 speeds[time_slot][road_id1].append(speed)
#             else:
#                 # 如果两个点在不同的路段，可以平均分配速度给两个路段
#                 speeds[time_slot][road_id1].append(speed)
#                 speeds[time_slot][road_id2].append(speed)
#
# def compute_mean_speed(speeds):
#     if len(speeds) == 0:
#         return 0
#     return np.mean(speeds)
#
# class SpeedToFeature(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SpeedToFeature, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         return self.linear(x)
#
#
#
#
# def process_speeds_mean(speeds, num_time_slots, num_road_segments):
#     # 初始化一个空数组来存储每个时间槽每个路段的平均速度
#     mean_speed_vectors = np.zeros((num_time_slots, num_road_segments, 1))
#
#     # 遍历每个时间槽和路段，计算平均速度
#     for time_slot in range(num_time_slots):
#         for road_segment in range(num_road_segments):
#             speed_list = speeds[time_slot][road_segment]
#             if len(speed_list) > 0:
#                 # 计算速度的平均值
#                 mean_speed = compute_mean_speed(speed_list)
#                 mean_speed_vectors[time_slot, road_segment, 0] = mean_speed
#             else:
#                 # 如果某个时间槽的路段没有速度数据，可以设置为0或者其他默认值
#                 mean_speed_vectors[time_slot, road_segment, 0] = 0
#
#     return mean_speed_vectors
#
#
# try:
#     # 原始代码
#     folder_path = './data/data_v'
#     output_folder_path = './traffic'
#     edge_valid = './data/rn_validedge_one_chengdu.json'
#
#     with open(edge_valid, 'r') as file:
#         edge_valid_map = json.load(file)
#
#     # timestamp_0 = 1372637715
#     timestamp_0= 1477929834
#     time_interval = 60
#
#     # 初始化速度数据为嵌套defaultdict，存储多个速度值
#     speeds = defaultdict(lambda: defaultdict(list))
#
#     num_time_slots = 72
#     num_road_segments =4285
#
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'):
#             input_filepath = os.path.join(folder_path, filename)
#             with open(input_filepath, 'r') as file:
#                 lines = file.readlines()
#                 num = 0
#                 cur_trajectory = []
#                 for line in lines:
#                     if line.startswith('-'):
#                         cal_cur_traj(speeds, cur_trajectory)
#                         cur_trajectory = []
#                         num += 1
#                         if (num % 5000 == 0):
#                             print(num)
#                         continue
#                     timestamp, lat, lng, road_id = line.strip().split()
#                     road_id = edge_valid_map.get(road_id)
#                     if road_id is not None:
#                         cur_trajectory.append((int(timestamp), float(lat), float(lng), int(road_id)))
#
#     # 使用新的 process_speeds_mean 函数
#     mean_speed_vectors = process_speeds_mean(speeds, num_time_slots, num_road_segments)
#
#     # 保存处理后的平均速度数据
#     np.save(os.path.join(output_folder_path, 'road_mean_speeds.npy'), mean_speed_vectors)
#     print("处理完成，平均速度数据已保存为 (6284, 12614, 1) 的 nPy 文件。")
#
# except Exception as e:
#     import traceback
#
#     print("错误行：", traceback.format_exc())





# one_hot值
import math
from collections import defaultdict
import os
import json
import numpy as np
import torch
import torch.nn as nn


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # 地球半径，单位：公里
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def cal_cur_traj(speeds, cur_trajectory):
    for i in range(1, len(cur_trajectory)):
        t1, lat1, lng1, road_id1 = cur_trajectory[i - 1]
        t2, lat2, lng2, road_id2 = cur_trajectory[i]

        # 计算两点之间的距离
        distance = haversine_distance(lat1, lng1, lat2, lng2)
        time_diff = (t2 - t1) / 3600  # 时间差，单位为小时

        if time_diff > 0:
            speed = distance / time_diff  # 计算速度，单位为公里/小时
            time_slot = int((t1 - timestamp_0) / (time_interval * 60))  # 确定时间槽

            if road_id1 == road_id2:
                # 如果两个点在同一条路段，则直接将速度加到该路段的时间槽
                speeds[time_slot][road_id1].append(speed)
            else:
                # 如果两个点在不同的路段，可以平均分配速度给两个路段
                speeds[time_slot][road_id1].append(speed)
                speeds[time_slot][road_id2].append(speed)

def compute_mean_speed(speeds):
    if len(speeds) == 0:
        return 0
    return np.mean(speeds)

class SpeedToFeature(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SpeedToFeature, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)



def speed_to_one_hot(speeds, bins):
    """
    将速度转换为 One-Hot 编码，速度值将根据给定的速度区间进行分类
    :param speeds: 速度列表
    :param bins: 速度区间
    :return: 对应的 One-Hot 编码
    """
    one_hot_encoding = np.zeros(len(bins) - 1, dtype=np.float32)
    for speed in speeds:
        bin_idx = np.digitize(speed, bins) - 1  # 计算速度对应的区间
        if 0 <= bin_idx < len(one_hot_encoding):
            one_hot_encoding[bin_idx] = 1  # 将对应的区间标记为 1
    return one_hot_encoding

def process_speeds_mean_to_one_hot(speeds, num_time_slots, num_road_segments, bins):
    """
    处理每个时间槽和路段的速度，计算平均速度并转换为 One-Hot 编码
    """
    # 初始化数组来存储 One-Hot 编码
    one_hot_vectors = np.zeros((num_time_slots, num_road_segments, len(bins) - 1), dtype=np.float32)

    # 遍历每个时间槽和路段，计算平均速度并进行 One-Hot 编码
    for time_slot in range(num_time_slots):
        for road_segment in range(num_road_segments):
            speed_list = speeds[time_slot][road_segment]

            if len(speed_list) > 0:
                # 计算平均速度
                mean_speed = compute_mean_speed(speed_list)

                # 转换为 One-Hot 编码
                one_hot_encoding = speed_to_one_hot([mean_speed], bins)
                one_hot_vectors[time_slot, road_segment] = one_hot_encoding
            else:
                # 如果没有速度数据，则不设置任何区间（保持为0）
                one_hot_vectors[time_slot, road_segment] = np.zeros(len(bins) - 1, dtype=np.float32)

    return one_hot_vectors

def process_speeds_mean(speeds, num_time_slots, num_road_segments):
    # 初始化一个空数组来存储每个时间槽每个路段的平均速度
    mean_speed_vectors = np.zeros((num_time_slots, num_road_segments, 1))

    # 遍历每个时间槽和路段，计算平均速度
    for time_slot in range(num_time_slots):
        for road_segment in range(num_road_segments):
            speed_list = speeds[time_slot][road_segment]
            if len(speed_list) > 0:
                # 计算速度的平均值
                mean_speed = compute_mean_speed(speed_list)
                mean_speed_vectors[time_slot, road_segment, 0] = mean_speed
            else:
                # 如果某个时间槽的路段没有速度数据，可以设置为0或者其他默认值
                mean_speed_vectors[time_slot, road_segment, 0] = 0

    return mean_speed_vectors


try:
    # 原始代码
    speed_bins = torch.tensor([0, 1, 15, 30, 50, 100])
    folder_path = './data/data_v'
    output_folder_path = './traffic'
    edge_valid = './data/rn_validedge_one_chengdu.json'

    with open(edge_valid, 'r') as file:
        edge_valid_map = json.load(file)

    # timestamp_0 = 1372637715
    timestamp_0= 1477929834
    time_interval = 60

    # 初始化速度数据为嵌套defaultdict，存储多个速度值
    speeds = defaultdict(lambda: defaultdict(list))

    num_time_slots = 72
    num_road_segments =4285

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            input_filepath = os.path.join(folder_path, filename)
            with open(input_filepath, 'r') as file:
                lines = file.readlines()
                num = 0
                cur_trajectory = []
                for line in lines:
                    if line.startswith('-'):
                        cal_cur_traj(speeds, cur_trajectory)
                        cur_trajectory = []
                        num += 1
                        if (num % 5000 == 0):
                            print(num)
                        continue
                    timestamp, lat, lng, road_id = line.strip().split()
                    road_id = edge_valid_map.get(road_id)
                    if road_id is not None:
                        cur_trajectory.append((int(timestamp), float(lat), float(lng), int(road_id)))

    # 使用新的 process_speeds_mean 函数
    # mean_speed_vectors = process_speeds_mean(speeds, num_time_slots, num_road_segments)
    one_hot_vectors = process_speeds_mean_to_one_hot(speeds, num_time_slots, num_road_segments, speed_bins)
    # 保存处理后的平均速度数据
    np.save(os.path.join(output_folder_path, 'one_hot_vectors_chengdu.npy'), one_hot_vectors)
    print("处理完成")

except Exception as e:
    import traceback

    print("错误行：", traceback.format_exc())
