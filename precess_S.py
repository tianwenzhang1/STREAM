import pandas as pd
import os
import datetime
import pickle
import numpy as np
import geopy.distance


timestamp_0 = 1478016098
time_interval = 60


def height2lat(height):
    return height / 110.574


def width2lng(width):
    return width / 111.320 / 0.99974


def in_boundary(lat, lng, b):
    return b['min_lng'] < lng < b['max_lng'] and b['min_lat'] < lat < b['max_lat']

def cal_speed(lat_prev, lng_prev, t_prev, lat, lng, timestamp):
    coords_1 = (lat_prev, lng_prev)
    coords_2 = (lat, lng)

    # print(coords_1, coords_2)

    dist = geopy.distance.distance(coords_1, coords_2).km


    t = (timestamp - t_prev) / (60.0*60.0)

    speed = dist / t

    return speed

def rawTrj2csv():
    # preprocess raw trajs; do filtering;
    # step 1: select the grid cells with the highest trajectory density (e.g., top-200 popular cells) as S or D.
    print('starting traffic2pkl')
    fname = data_dir + '/' + data_name + '_traffics.pkl'
    traffics = np.zeros((96, 130, 130))  #author 6284
    t_indx_n, lat_n, lng_n = np.shape(traffics)
    traffics_dict = {}


    grid_freq = dict()
    lat_size, lng_size = height2lat(grid_height), width2lng(grid_width)
    high = (boundary['max_lat'] - boundary['min_lat']) / lat_size
    wide = (boundary['max_lng'] - boundary['min_lng']) / lng_size
    path = "chengdu_traj"
    files = os.listdir(path)
    for file in files:
        if '.txt' in file:
            print('step 1 with %s' %file)
            f = open(os.path.join(path, file), "r")
            lines = f.readlines()
            t_prev = 0
            lat_prev, lng_prev = 0, 0
            for line in lines:
                if line.startswith('#'):
                    t_prev = 0
                    lat_prev, lng_prev = 0, 0
                    continue
                splits = line.strip().split(' ')
                lng, lat = splits[2], splits[1]
                grid_i = int((float(lat) - boundary['min_lat']) / lat_size)
                grid_j = int((float(lng) - boundary['min_lng']) / lng_size)
                # Grid size: (130, 130)
                if grid_i >= 130 or grid_i < 0 or grid_j >= 130 or grid_j < 0:
                    continue

                timestamp = float(splits[0])
                t_ind = int((timestamp - timestamp_0) / (time_interval*60))
                if t_prev != 0:
                    if float(lat) != lat_prev and float(lng)!= lng_prev and timestamp != t_prev:
                        if t_ind not in traffics_dict.keys():
                            traffics_dict[t_ind] = {}
                        if (grid_i,grid_j) not in traffics_dict[t_ind].keys():
                            traffics_dict[t_ind][(grid_i,grid_j)] = []
                        if lat_prev >= -90 and lat_prev <= 90 and float(lat) >= -90 and float(lat) <= 90:
                            speed = cal_speed(lat_prev, lng_prev, t_prev, float(lat), float(lng), timestamp)
                            traffics_dict[t_ind][(grid_i, grid_j)].append(speed)


                t_prev = timestamp
                lat_prev = float(lat)
                lng_prev = float(lng)

                if (grid_i, grid_j) not in grid_freq.keys():
                    grid_freq[(grid_i, grid_j)] = 0
                grid_freq[(grid_i, grid_j)] = grid_freq[(grid_i, grid_j)] + 1

    for t_ind in traffics_dict.keys():
        for cell in traffics_dict[t_ind].keys():
            if cell[0]<lat_n and cell[1]<lng_n:
                ave_speed = np.mean(traffics_dict[t_ind][cell])
                traffics[t_ind][cell[0]][cell[1]] = ave_speed

    # save traffics to pkl
    print('saving traffics to pkl')
    with open(fname, 'wb') as handle:
        pickle.dump(traffics, handle, protocol=pickle.HIGHEST_PROTOCOL)





if __name__ == '__main__':
    data_dir = './traffic'
    data_name = "chengdu"
    grid_height, grid_width = 0.062, 0.074
    boundary = {'min_lat': 30.655, 'max_lat': 30.727, 'min_lng': 104.043, 'max_lng': 104.129}
    # boundary = {'min_lat': 41.101975, 'max_lat': 41.187462, 'min_lng': -8.677057, 'max_lng': -8.575305}
    rawTrj2csv()
    # main()