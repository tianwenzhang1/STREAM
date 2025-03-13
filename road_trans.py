import os
from collections import deque
from scipy.sparse import lil_matrix

folder_path = './data/train_valid'
max_road_id = 4712
k_steps = 1

transition_matrix = lil_matrix((max_road_id + 1, max_road_id + 1), dtype=int)

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r') as file:
            recent_roads = deque(maxlen=k_steps)

            for line in file:
                if line.startswith('-'):
                    recent_roads.clear()
                    continue

                parts = line.strip().split()
                if len(parts) < 4:
                    continue

                timestamp, lat, lon, road_id = parts
                road_id = int(road_id)

                for prev_road in recent_roads:
                    transition_matrix[prev_road, road_id] += 1
                recent_roads.append(road_id)

output_file_path = './data/road_trans_Chengdu.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write('src,dst,weight\n')
    for i in range(max_road_id + 1):
        for j in range(max_road_id + 1):
            if transition_matrix[i, j] > 0:
                output_file.write(f"{i},{j},{transition_matrix[i, j]}\n")

print(f"successful")
