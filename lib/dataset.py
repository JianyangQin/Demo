import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    def __init__(self, city, interval_1, interval_2, seq_len, pred_len, mode, device):

        if city not in ['nyc_mb', 'nyc_mt', 'nyc_bt']:
            raise ValueError('Unknown Dataset!')

        if interval_1 not in [5, 15, 30, 45, 60] and interval_2 not in [5, 15, 30, 45, 60]:
            raise ValueError('Time Interval must be one of {15, 30, 45, 60}(min)!')

        print('************************************************************')
        print('City:', city, '; ')
        print('Traffic mode 1 interval:', str(interval_1) + ' min')
        print('Traffic mode 2 interval:', str(interval_2) + ' min')

        if city == 'nyc_mb':
            mode1_data = np.load('data/Metro+Bus/Metro_NYC.npy')[:, :, np.newaxis]
            mode2_data = np.load('data/Metro+Bus/Bus_NYC.npy')[:, :, np.newaxis]
            time_data = np.load('data/Metro+Bus/NYC_mb_date.npy')
            weather_data = np.load('data/Metro+Bus/weather_nyc_mb.npy')
        elif city == 'nyc_mt':
            mode1_data = np.load('data/Metro+Taxi/Metro_NYC.npy')[:, :, np.newaxis]
            mode2_data = np.load('data/Metro+Taxi/Taxi_NYC.npy')[:, :, np.newaxis]
            time_data = np.load('data/Metro+Taxi/NYC_mt_date.npy')
            weather_data = np.load('data/Metro+Taxi/weather_nyc_mt.npy')
        elif city == 'nyc_bt':
            mode1_data = np.load('data/Bus+Taxi/Bus_NYC.npy')[:, :, np.newaxis]
            mode2_data = np.load('data/Bus+Taxi/Taxi_NYC.npy')[:, :, np.newaxis]
            time_data = np.load('data/Bus+Taxi/NYC_bt_date.npy')
            weather_data = np.load('data/Bus+Taxi/weather_nyc_bt.npy')
        else:
            mode1_data = None
            mode2_data = None

        data_mode1 = granularity_transform(mode1_data, source=interval_1, target=interval_1)
        data_mode2 = granularity_transform(mode2_data, source=interval_2, target=interval_2)

        print(data_mode1.shape)
        print(data_mode2.shape)

        self.X_1, self.Y_1, self.X_2, self.Y_2, self.time_X, self.weather_X, self.time_Y, self.weather_Y = \
            make_dataset(
                graph_feats_1 = data_mode1,
                graph_feats_2 = data_mode2,
                time_data = time_data,
                weather_data = weather_data,
                interval_1 = 60,
                interval_2 = 60,
                seq_len = seq_len,
                pred_len = pred_len,
                mode = mode,
                normalize=True,
                device = device
            )

        assert self.X_1.shape[0] == self.Y_1.shape[0], 'Data Error!'
        assert self.X_2.shape[0] == self.Y_2.shape[0], 'Data Error!'
        print('X_1:', self.X_1.shape, 'Y_1:', self.Y_1.shape)
        print('X_1:', self.X_1.dtype, 'Y_1:', self.Y_1.dtype)
        print('X_2:', self.X_2.shape, 'Y_2:', self.Y_2.shape)
        print('time_X:', self.time_X.shape, 'weather_X:', self.weather_X.shape)
        print('time_Y:', self.time_Y.shape, 'weather_Y:', self.weather_Y.shape)
        print(mode + ' dataset created !')
        print('************************************************************')

    def __getitem__(self, index):
        return self.X_1[index], self.Y_1[index], self.X_2[index], self.Y_2[index], \
               self.time_X[index], self.weather_X[index]

    def __len__(self):
        return len(self.Y_1)


def get_partial_data(data, start_month, end_month, time_per_days):
    month_index = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "10": 8, "11": 9, "12": 10, "1": 11}
    day_per_months = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]
    time_per_months = list(np.asarray(day_per_months) * time_per_days)
    split_index = [0]
    for i in range(len(time_per_months)):
        split_index.append(split_index[i] + time_per_months[i])

    start_month_index, end_month_index = month_index[start_month], month_index[end_month]
    return data[split_index[start_month_index]:split_index[end_month_index+1]]


def granularity_transform(data_seq, source, target):
    """
    :param data_seq: (num_timestamps, num_nodes, num_feats)
    :param source: source granularity of data_seq (min)
    :param target: target granularity (min)
    :return:
    """
    if source == target:
        return data_seq
    else:
        origin_len = len(data_seq)
        merge_len = int(target / source)
        coarsened_size = int(origin_len / merge_len)
        coarsened_seq = []
        for t in range(coarsened_size):
            coarsened_seq.append(sum(data_seq[t * merge_len: (t + 1) * merge_len]))
        return np.array(coarsened_seq)


def get_samples(data_1, data_2, seq_len, pred_len, interval_1, interval_2):
    samples_1 = []
    samples_2 = []
    if interval_1 >= interval_2:
        seq_step = seq_len * 60 // interval_1
        pred_step = int(pred_len * 60 // interval_1)
        times = interval_1 // interval_2
        len_data = data_1.shape[0]

        for idx in range(len_data):
            if seq_step <= idx and idx + pred_step <= len_data:
                sample_x_1 = data_1[idx - seq_step: idx, :, :]
                sample_y_1 = data_1[idx: idx + pred_step, :, :]
                sample_x_2 = data_2[(idx - seq_step) * times: idx * times, :, :]
                sample_y_2 = data_2[idx * times: (idx + pred_step) * times, :, :]
                samples_1.append((sample_x_1, sample_y_1))
                samples_2.append((sample_x_2, sample_y_2))

    samples_1 = [np.stack(i, axis=0) for i in zip(*samples_1)]
    samples_2 = [np.stack(i, axis=0) for i in zip(*samples_2)]

    return samples_1, samples_2

def get_samples_environment(data, seq_len, pred_len, interval_1):
    samples_xs = []
    samples_ys = []

    if np.isnan(data).any():
        print("env:True!")
    seq_step = seq_len * 60 // interval_1
    pred_step = int(pred_len * 60 // interval_1)
    len_data = data.shape[0]
    if data.ndim < 2:
        data = data[:,np.newaxis]

    for idx in range(len_data):
        if seq_step <= idx and idx + pred_step <= len_data:
            sample_x = data[idx - seq_step: idx, :]
            sample_y = data[idx: idx + pred_step, :]
            samples_xs.append((sample_x))
            samples_ys.append((sample_y))

    samples_xs = np.array(samples_xs)
    samples_ys = np.array(samples_ys)
    return samples_xs, samples_ys

def get_samples_date(data, seq_len, pred_len, interval_1):
    samples_xs = []
    samples_ys = []

    if np.isnan(data).any():
        print("date:True!")
    data = np.asarray(data, dtype=float)

    seq_step = seq_len * 60 // interval_1
    pred_step = int(pred_len * 60 // interval_1)
    len_data = data.shape[0]
    if data.ndim < 2:
        data = data[:,np.newaxis]

    for idx in range(len_data):
        if seq_step <= idx and idx + pred_step <= len_data:
            sample_x = data[idx - seq_step: idx, :]
            sample_y = data[idx: idx + pred_step, :]
            samples_xs.append((sample_x))
            samples_ys.append((sample_y))

    samples_xs = np.array(samples_xs)
    samples_ys = np.array(samples_ys)
    return samples_xs, samples_ys


def split_multi(samples_x1, samples_y1, samples_x2, samples_y2, mode, device, normalize=True):
    sp_x1_1 = round(samples_x1.shape[0] * 0.7)
    sp_x1_2 = round(samples_x1.shape[0] * 0.8)

    sp_x2_1 = round(samples_x1.shape[0] * 0.7)
    sp_x2_2 = round(samples_x1.shape[0] * 0.8)


    train_x1, val_x1, test_x1 = samples_x1[:sp_x1_1], samples_x1[sp_x1_1:sp_x1_2], samples_x1[sp_x1_2:]
    train_y1, val_y1, test_y1 = samples_y1[:sp_x1_1], samples_y1[sp_x1_1:sp_x1_2], samples_y1[sp_x1_2:]

    train_x2, val_x2, test_x2 = samples_x2[:sp_x2_1], samples_x2[sp_x2_1:sp_x2_2], samples_x2[sp_x2_2:]
    train_y2, val_y2, test_y2 = samples_y2[:sp_x2_1], samples_y2[sp_x2_1:sp_x2_2], samples_y2[sp_x2_2:]

    train_x = np.concatenate([train_x1, train_x2], axis=2)

    if normalize:
        mean = train_x.mean(axis=(0, 1, 2, 3), keepdims=True)
        std = train_x.std(axis=(0, 1, 2, 3), keepdims=True)
        print("mean:{}".format(mean))
        print("std:{}".format(std))

        def z_score(x):
            return (x - mean) / std

        train_x1 = z_score(train_x1)
        val_x1 = z_score(val_x1)
        test_x1 = z_score(test_x1)

        train_x2 = z_score(train_x2)
        val_x2 = z_score(val_x2)
        test_x2= z_score(test_x2)

    if mode == 'train':
        return torch.from_numpy(train_x1).float().to(device), torch.from_numpy(train_y1).float().to(device), \
               torch.from_numpy(train_x2).float().to(device), torch.from_numpy(train_y2).float().to(device)
    elif mode == 'val':
        return torch.from_numpy(val_x1).float().to(device), torch.from_numpy(val_y1).float().to(device), \
               torch.from_numpy(val_x2).float().to(device), torch.from_numpy(val_y2).float().to(device)
    elif mode == 'test':
        return torch.from_numpy(test_x1).float().to(device), torch.from_numpy(test_y1).float().to(device), \
               torch.from_numpy(test_x2).float().to(device), torch.from_numpy(test_y2).float().to(device)
    else:
        raise ValueError('Invalid Type of Dataset!')


def split(samples_x, samples_y, mode, device, normalize=True):
    sp1 = round(samples_x.shape[0] * 0.7)
    sp2 = round(samples_x.shape[0] * 0.8)

    train_x, val_x, test_x = samples_x[:sp1], samples_x[sp1:sp2], samples_x[sp2:]
    train_y, val_y, test_y = samples_y[:sp1], samples_y[sp1:sp2], samples_y[sp2:]

    
    if normalize:
        mean = train_x.mean(axis=(0, 1, 2, 3), keepdims=True)
        std = train_x.std(axis=(0, 1, 2, 3), keepdims=True)
        print("mean:{}".format(mean))
        print("std:{}".format(std))

        def z_score(x):
            return (x - mean) / std
        
        train_x = z_score(train_x)
        val_x = z_score(val_x)
        test_x = z_score(test_x)

    if mode == 'train':
        return torch.from_numpy(train_x).float().to(device), torch.from_numpy(train_y).float().to(device)
    elif mode == 'val':
        return torch.from_numpy(val_x).float().to(device), torch.from_numpy(val_y).float().to(device)
    elif mode == 'test':
        return torch.from_numpy(test_x).float().to(device), torch.from_numpy(test_y).float().to(device)
    else:
        raise ValueError('Invalid Type of Dataset!')
    
def split_environment(samples_x, mode, device, normalize=True):
    # sp1 = round(samples_x.shape[0] * 0.6)
    sp1 = round(samples_x.shape[0] * 0.7)
    sp2 = round(samples_x.shape[0] * 0.8)

    train_x, val_x, test_x = samples_x[:sp1], samples_x[sp1:sp2], samples_x[sp2:]

    print(train_x.shape)

    if mode == 'train':
        return torch.from_numpy(train_x).float().to(device)
    elif mode == 'val':
        return torch.from_numpy(val_x).float().to(device)
    elif mode == 'test':
        return torch.from_numpy(test_x).float().to(device)
    else:
        raise ValueError('Invalid Type of Dataset!')


def make_dataset(graph_feats_1, graph_feats_2, time_data, weather_data, interval_1, interval_2, seq_len, pred_len, mode, device, normalize=True):

    samples_1, samples_2 = get_samples(graph_feats_1, graph_feats_2, seq_len, pred_len, interval_1, interval_2)
    samples_date_x, samples_date_y = get_samples_date(time_data, seq_len, pred_len, interval_1)
    samples_weather_x, samples_weather_y = get_samples_environment(weather_data, seq_len, pred_len, interval_1)
    samples_x_1 = samples_1[0]
    samples_y_1 = samples_1[1]
    samples_x_2 = samples_2[0]
    samples_y_2 = samples_2[1]
    data_samples_x1, data_samples_y1 = split(samples_x_1, samples_y_1, mode, device, normalize)
    data_samples_x2, data_samples_y2 = split(samples_x_2, samples_y_2, mode, device, normalize)

    data_samples_date_x = split_environment(samples_date_x, mode, device, normalize)
    data_samples_weather_x = split_environment(samples_weather_x, mode, device, normalize)

    data_samples_date_y = split_environment(samples_date_y, mode, device, normalize)
    data_samples_weather_y = split_environment(samples_weather_y, mode, device, normalize)
    

    return data_samples_x1, data_samples_y1, data_samples_x2, data_samples_y2, \
           data_samples_date_x, data_samples_weather_x, data_samples_date_y, data_samples_weather_y
     

