import os
import random
import cv2
import h5py
import time
import numpy as np

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def to_np(x):
    return x.cpu().detach().numpy()


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']



def load_data_hdf5(data_path, dict_name, max_length=-1):
    ''' load data '''
    data_f = h5py.File(data_path, 'r')

    ''' frame count '''
    data_fc = data_f['frame_count'][0]
    data_fc = data_fc if max_length == -1 else min(data_fc, max_length)

    ''' time stamp '''
    data_ts = np.array(data_f['ts'][:data_fc])

    ''' raw data '''
    data = np.array(data_f[dict_name][:data_fc]).astype(np.float)

    return data, data_fc, data_ts


def filter_artifact_in_touch(touch_raw, mask, thresh, t_win, s_neighbor):
    # filter artifact in touch
    touch = touch_raw.copy()
    touch_fc = touch_raw.shape[0]

    for idx_frame in range(touch_fc):
        idx_artifact = np.sum(
            touch_raw[max(0, idx_frame - t_win) : min(touch_fc, idx_frame + t_win)] > thresh, 0)
        x, y = np.where(idx_artifact > 0)

        for i in range(len(x)):
            if mask[x[i], y[i]] == 0:
                continue

            v, cnt = 0., 0.
            for dx in range(-s_neighbor, s_neighbor + 1):
                for dy in range(-s_neighbor, s_neighbor + 1):
                    xx, yy = x[i] + dx, y[i] + dy
                    if 0 <= xx < touch.shape[1] and 0 <= yy < touch.shape[2] and mask[xx, yy] == 1:
                        if touch_raw[idx_frame, xx, yy] <= thresh:
                            v += touch_raw[idx_frame, xx, yy]
                            cnt += 1

            # if no valid sensors are found, expand the search area
            if cnt == 0:
                v, cnt = 0., 0.
                for dx in range(-s_neighbor - 1, s_neighbor + 2):
                    for dy in range(-s_neighbor - 1, s_neighbor + 2):
                        xx, yy = x[i] + dx, y[i] + dy
                        if 0 <= xx < touch.shape[1] and 0 <= yy < touch.shape[2] and mask[xx, yy] == 1:
                            if touch_raw[idx_frame, xx, yy] <= thresh:
                                v += touch_raw[idx_frame, xx, yy]
                                cnt += 1

            touch[idx_frame, x[i], y[i]] = v / cnt

    return touch


def clip_base_response(touch, base_clip_percentile):
    # clip base response
    touch_base = np.percentile(touch, base_clip_percentile, axis=0)
    touch = np.clip(touch - touch_base, 0, np.max(touch - touch_base))
    return touch


def filter_artifact_in_scale(scale_raw, thresh):
    scale_seq = scale_raw.copy()
    for i in range(len(scale_raw)):
        if scale_raw[i] < thresh:
            if i == 0:
                for j in range(i, len(scale_raw) - 1):
                    if scale_raw[j] >= thresh:
                        scale_seq[i] = scale_raw[j]
                        break
            elif i == len(scale_raw) - 1:
                for j in range(i, 0, -1):
                    if scale_raw[j] >= thresh:
                        scale_seq[i] = scale_raw[j]
                        break
            else:
                for j in range(i, 0, -1):
                    if scale_raw[j] >= thresh:
                        scale_seq[i] = scale_raw[j]
                        break
                for j in range(i, len(scale_raw) - 1):
                    if scale_raw[j] >= thresh:
                        scale_seq[i] = (scale_seq[i] + scale_raw[j]) / 2.
                        break

    return scale_seq


def synchronize_touch_and_scale(touch_seq, scale_seq, touch_ts, scale_ts, offset):
    touch_idx = []
    scale_ret = []
    idx_touch = 0
    idx_scale = 0
    while touch_ts[idx_touch] > scale_ts[idx_scale]:
        idx_scale += 1

    for i in range(idx_scale, len(scale_seq)):
        flag = False
        while touch_ts[idx_touch] < scale_ts[i]:
            idx_touch += 1
            if idx_touch >= min(len(touch_seq), len(touch_seq) - offset):
                flag = True
                break
        if flag:
            break

        touch_idx.append(idx_touch + offset)
        scale_ret.append(scale_seq[i])

    touch_idx = np.stack(touch_idx)
    scale_ret = np.stack(scale_ret)

    # return touch_idx and scale_data
    return touch_idx, scale_ret


def synchronize_touch_and_touch(touch_0_seq, touch_1_seq, touch_0_ts, touch_1_ts, offset):
    touch_0_idx = []
    touch_1_idx = []
    idx_0 = 0
    idx_1 = 0
    while touch_0_ts[idx_0] > touch_1_ts[idx_1]:
        idx_1 += 1

    for i in range(idx_1, len(touch_1_seq)):
        flag = False
        while touch_0_ts[idx_0] < touch_1_ts[i]:
            idx_0 += 1
            if idx_0 >= min(len(touch_0_seq), len(touch_0_seq) - offset):
                flag = True
                break
        if flag:
            break

        touch_0_idx.append(idx_0 + offset)
        touch_1_idx.append(i)

    touch_0_idx = np.stack(touch_0_idx)
    touch_1_idx = np.stack(touch_1_idx)

    return touch_0_idx, touch_1_idx

