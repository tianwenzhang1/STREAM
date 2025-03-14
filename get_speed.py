import math
from collections import defaultdict
import os
import json
import numpy as np
import torch
import torch.nn as nn
from utils.map import RoadNetworkMapFull


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
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

        distance = haversine_distance(lat1, lng1, lat2, lng2)
        time_diff = (t2 - t1) / 3600

        if time_diff > 0:
            speed = distance / time_diff
            time_slot = int((t1 - timestamp_0) / (time_interval * 60))

            if road_id1 == road_id2:
                speeds[time_slot][road_id1].append(speed)
            else:
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
    one_hot_encoding = np.zeros(len(bins) - 1, dtype=np.float32)
    for speed in speeds:
        bin_idx = np.digitize(speed, bins) - 1
        if 0 <= bin_idx < len(one_hot_encoding):
            one_hot_encoding[bin_idx] = 1
    return one_hot_encoding

def process_speeds_mean_to_one_hot(speeds, num_time_slots, num_road_segments, bins):
    one_hot_vectors = np.zeros((num_time_slots, num_road_segments, len(bins) - 1), dtype=np.float32)

    for time_slot in range(num_time_slots):
        for road_segment in range(num_road_segments):
            speed_list = speeds[time_slot][road_segment]

            if len(speed_list) > 0:
                mean_speed = compute_mean_speed(speed_list)

                # convert One-Hot coding
                one_hot_encoding = speed_to_one_hot([mean_speed], bins)
                one_hot_vectors[time_slot, road_segment] = one_hot_encoding
            else:
                one_hot_vectors[time_slot, road_segment] = np.zeros(len(bins) - 1, dtype=np.float32)

    return one_hot_vectors

def process_speeds_mean(speeds, num_time_slots, num_road_segments):
    mean_speed_vectors = np.zeros((num_time_slots, num_road_segments, 1))

    for time_slot in range(num_time_slots):
        for road_segment in range(num_road_segments):
            speed_list = speeds[time_slot][road_segment]
            if len(speed_list) > 0:
                mean_speed = compute_mean_speed(speed_list)
                mean_speed_vectors[time_slot, road_segment, 0] = mean_speed
            else:
                mean_speed_vectors[time_slot, road_segment, 0] = 0

    return mean_speed_vectors


try:
    city = 'Chengdu'
    map_root = f"./data/roadnet/{city}/"
    if city == "Porto":
        rn = RoadNetworkMapFull(map_root, zone_range=[41.111975, -8.667057, 41.177462, -8.585305], unit_length=50)
    elif city == "chengdu":
        rn = RoadNetworkMapFull(map_root, zone_range=[30.655, 104.043, 30.727, 104.129], unit_length=50)
    elif city == "harbin":
        rn = RoadNetworkMapFull(map_root, zone_range=[45.697920, 126.586130, 45.777090, 126.671862], unit_length=50)
    else:
        raise NotImplementedError

    speed_bins = torch.tensor([0, 1, 15, 30, 50, 100])
    folder_path = f'./data/{city}dataset'
    output_folder_path = f'./data/{city}/traffic'

    edge_valid = rn.valid_edge_one

    # timestamp_0 = 1372637715
    timestamp_0= 1477929834
    time_interval = 60

    #init
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
                        continue
                    timestamp, lat, lng, road_id = line.strip().split()
                    road_id = edge_valid.get(road_id)
                    if road_id is not None:
                        cur_trajectory.append((int(timestamp), float(lat), float(lng), int(road_id)))

    one_hot_vectors = process_speeds_mean_to_one_hot(speeds, num_time_slots, num_road_segments, speed_bins)
    # save
    np.save(os.path.join(output_folder_path, 'one_hot_vectors_chengdu.npy'), one_hot_vectors)

except Exception as e:
    import traceback

    print("error：", traceback.format_exc())
