import os
import random
import numpy as np


def read_trajectory_file(file_path):
    trajectories = []
    start = {}
    with open(file_path, 'r') as file:
        current_trajectory = []
        for line in file:
            line = line.strip()
            if line.startswith('-'):
                if current_trajectory:
                    parts = current_trajectory[0].strip().split()
                    start[len(trajectories)] = parts[0]
                    trajectories.append(current_trajectory)
                    current_trajectory = []
            else:
                current_trajectory.append(line)
    return trajectories, start


def count_num(sorted_start):
    time_num = {i: 0 for i in range(72)}
    for timestamp in sorted_start.values():
        remainder = int((int(timestamp) - 1477929834) / 3600)
        time_num[remainder] += 1
    return time_num


# test dataset
def downsample_traj(pt_list, ds_type, keep_ratio):
    assert ds_type in ['uniform', 'random'], 'only `uniform` or `random` is supported'
    old_pt_list = pt_list.copy()
    start_pt = old_pt_list[0]
    end_pt = old_pt_list[-1]

    if ds_type == 'uniform':
        if (len(old_pt_list) - 1) % int(1 / keep_ratio) == 0:
            new_pt_list = old_pt_list[::int(1 / keep_ratio)]
        else:
            new_pt_list = old_pt_list[::int(1 / keep_ratio)] + [end_pt]
    elif ds_type == 'random':
        num_points_to_sample = int((len(old_pt_list) - 2) * keep_ratio)
        if num_points_to_sample > 0:
            sampled_inds = sorted(random.sample(range(1, len(old_pt_list) - 1), num_points_to_sample))
            new_pt_list = [start_pt] + list(np.array(old_pt_list)[sampled_inds]) + [end_pt]
        else:
            new_pt_list = [start_pt, end_pt]

    return new_pt_list

def sample_test_data(input_file, output_file, ds_type='uniform', keep_ratio=0.125):
    with open(input_file, 'r') as f_in:
        trajectories = []
        current_trajectory = []
        for line in f_in:
            if line.startswith('-'):
                if current_trajectory:
                    trajectories.append(current_trajectory)
                    current_trajectory = []
            else:
                current_trajectory.append(line.strip().split())


    sampled_trajectories = []
    for traj in trajectories:
        sampled_traj = downsample_traj(traj, ds_type, keep_ratio)
        sampled_trajectories.append(sampled_traj)

    counter = 1
    with open(output_file, 'w') as f_out:
        for traj in sampled_trajectories:
            for point in traj:
                f_out.write(' '.join(point) + '\n')
            f_out.write(f'-{counter}\n')
            counter += 1

    print(f"Sampled test data written to {output_file}")



def main():
    input_file = './raw_linear/chengdu_input.txt'
    output_file = './chengdu_output/chengdu_output.txt'
    output_folder = './dataset'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_trajectories, input_start = read_trajectory_file(input_file)
    output_trajectories, output_start = read_trajectory_file(output_file)

    # Ensure input and output lengths match
    assert len(input_trajectories) == len(output_trajectories), "Input and output trajectory lengths do not match"

    sorted_start = dict(sorted(input_start.items(), key=lambda item: item[1]))
    time_num = count_num(sorted_start)
    sorted_input_trajectories = [input_trajectories[index] for index in sorted_start.keys()]
    sorted_output_trajectories = [output_trajectories[index] for index in sorted_start.keys()]

    # Initialize storage for trajectories by time slot
    trajectories_by_time_slot = []
    for time_slot, num_trajectories in time_num.items():
        if num_trajectories >= 64:
            input_slot_trajectories = sorted_input_trajectories[:num_trajectories]
            output_slot_trajectories = sorted_output_trajectories[:num_trajectories]
            trajectories_by_time_slot.append((time_slot, input_slot_trajectories, output_slot_trajectories))
            sorted_input_trajectories = sorted_input_trajectories[num_trajectories:]
            sorted_output_trajectories = sorted_output_trajectories[num_trajectories:]

    # Allocate time slots into train, validation, and test sets
    total_trajectories = sum(len(slot[1]) for slot in trajectories_by_time_slot)
    train_limit = int(0.7 * total_trajectories)
    valid_limit = int(0.1 * total_trajectories)

    train_input, train_output, valid_input, valid_output, test_input, test_output = [], [], [], [], [], []
    train_data_num, valid_data_num, test_data_num = {}, {}, {}
    count_train, count_valid = 0, 0

    for time_slot, input_in_slot, output_in_slot in trajectories_by_time_slot:
        slot_size = len(input_in_slot)

        if count_train + slot_size <= train_limit:
            train_input.append(input_in_slot)
            train_output.append(output_in_slot)
            train_data_num[time_slot] = slot_size
            count_train += slot_size
        elif count_valid + slot_size <= valid_limit:
            valid_input.append(input_in_slot)
            valid_output.append(output_in_slot)
            valid_data_num[time_slot] = slot_size
            count_valid += slot_size
        else:
            test_input.append(input_in_slot)
            test_output.append(output_in_slot)
            test_data_num[time_slot] = slot_size

    # Write to txt files
    def write_trajectories(output_file_path, trajectories):
        with open(output_file_path, 'w') as output_file:
            counter = 1
            for trajectory_list in trajectories:
                for trajectory in trajectory_list:
                    for point in trajectory:
                        output_file.write(f"{point}\n")
                    output_file.write(f'-{counter}\n')
                    counter += 1

    write_trajectories(os.path.join(output_folder, "train_input.txt"), train_input)
    write_trajectories(os.path.join(output_folder, "train_output.txt"), train_output)
    write_trajectories(os.path.join(output_folder, "valid_input.txt"), valid_input)
    write_trajectories(os.path.join(output_folder, "valid_output.txt"), valid_output)
    write_trajectories(os.path.join(output_folder, "test_input.txt"), test_input)
    write_trajectories(os.path.join(output_folder, "test_output.txt"), test_output)

    # Write trajectory count
    with open(os.path.join(output_folder, "train_batch.txt"), 'w') as file:
        file.write(str(train_data_num))
    with open(os.path.join(output_folder, "valid_batch.txt"), 'w') as file:
        file.write(str(valid_data_num))
    with open(os.path.join(output_folder, "test_batch.txt"), 'w') as file:
        file.write(str(test_data_num))


if __name__ == '__main__':
    main()
    sample_test_data('./dataset/test_input.txt', './dataset/test_input_8.txt')







