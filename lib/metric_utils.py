import torch
import os
from datetime import datetime



import math

import numpy as np

import random



def dynamic_weight_average(loss_t_old, loss_t_new ):
    """

    :param loss_t_old: 每个task上一轮的loss列表，并且为标量
    :param loss_t_new:
    :return:
    """
    T = 2
    # 第1和2轮，w初设化为1，lambda也对应为1
    if loss_t_old is None or loss_t_new is None:
        return [1 , 1]

    assert len(loss_t_old) == len(loss_t_new)
    task_n = len(loss_t_old)

    w = [l_old / l_new for l_old, l_new in zip(loss_t_old, loss_t_new)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    return [task_n * l / lamb_sum for l in lamb]

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def masked_mse_loss(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def masked_mean_absolute_error(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def masked_mean_absolute_percentage_error(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs((true - pred) / true))


def masked_root_mean_squared_error(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((true - pred) ** 2) ** 0.5


def metrics(pred, true, mae_mask=None, mape_mask=0.0, rmse_mask=None):
    mae = masked_mean_absolute_error(pred, true, mae_mask).item()
    mape = masked_mean_absolute_percentage_error(pred, true, mape_mask).item() * 100
    rmse = masked_root_mean_squared_error(pred, true, rmse_mask).item()
    return round(mae, 4), round(mape, 4), round(rmse, 4)

def get_log_dir(path):
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    current_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(current_dir, 'experiments', path, current_time)
    return log_dir 
