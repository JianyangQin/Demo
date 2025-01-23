import os
import numpy as np

def time_add(data, week_start, interval=5, weekday_only=False, day_start=0, hour_of_day=24):
    # day and week
    if data.ndim == 1:
        data = data[:,np.newaxis]
        
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    hour_data = np.zeros_like(data)
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    # index_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    for index in range(data.shape[0]):
        hour_data[index:index + 1, :] = index % time_slot
        if (index) % time_slot == 0:
            day_init = day_start
        day_init = day_init + 1
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init

    return day_data, week_data, hour_data


def load_st_dataset(dataset):
    if dataset == 'NYC_BIKE':
        data_path = '/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/NYC/Ride-sharing+Bike-sharing/Bike_NYC.npy'
        data = np.load(data_path)  # DROP & PICK
        week_start = 5
        weekday_only = False
        interval = 30
        
        day_data, week_data, hour_data = time_add(data[..., 0], week_start, interval, weekday_only)
    elif dataset == 'NYC_TAXI':
        data_path = '/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/NYC/Ride-sharing+Bike-sharing/Taxi_NYC.npy'
        data = np.load(data_path)  # DROP & PICK
        week_start = 5
        weekday_only = False
        interval = 30
        
        day_data, week_data, hour_data = time_add(data[..., 0], week_start, interval, weekday_only)
    elif dataset == 'NYC_METRO':
        data_path = os.path.join('/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/NYC/Metro+Bus/Metro_NYC.npy')
        data = np.load(data_path)  # DROP & PICK
        week_start = 2
        weekday_only = False
        interval = 60
        
        day_data, week_data, hour_data = time_add(data[..., 0], week_start, interval, weekday_only)
    elif dataset == 'NYC_BUS':
        data_path = os.path.join('/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/NYC/Metro+Bus/Bus_NYC.npy')
        data = np.load(data_path)  # DROP & PICK
        week_start = 2
        weekday_only = False
        interval = 60
        
        day_data, week_data, hour_data = time_add(data[..., 0], week_start, interval, weekday_only)
    elif dataset == 'DC_BIKE':
        data_path = os.path.join('/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/DC/Ride-sharing+Bike-sharing/DC_Taxi.npy')
        data = np.load(data_path)  # DROP & PICK
        week_start = 1
        weekday_only = False
        interval = 60
        
        day_data, week_data, hour_data = time_add(data[..., 0], week_start, interval, weekday_only)
    elif dataset == 'DC_TAXI':
        data_path = os.path.join('/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/DC/Ride-sharing+Bike-sharing/DC_Bike.npy')
        data = np.load(data_path)  # DROP & PICK
        week_start = 1
        weekday_only = False
        interval = 60
        
        day_data, week_data, hour_data = time_add(data[..., 0], week_start, interval, weekday_only)
    else:
        raise ValueError
    day_data = np.expand_dims(day_data, axis=-1).astype(int)
    week_data = np.expand_dims(week_data, axis=-1).astype(int)
    hour_data = np.expand_dims(hour_data, axis=-1).astype(int)
    data_date = np.concatenate([week_data, hour_data], axis=-1)
        
    return data_date

data1 = load_st_dataset("DC_TAXI")[:,0,:]
data2 = load_st_dataset("DC_BIKE")
print(data1.shape)
print(data1)
np.save("/Users/curley/Downloads/new_model/USTCM/data/UrbanCityDataset/NYC/Metro+Bus/DC_tb_date.npy",data1)
