import os
import random
import cv2
import h5py
import time
import numpy as np
from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from knit_calib.utils.utils import load_data_hdf5, filter_artifact_in_touch, filter_artifact_in_scale
from knit_calib.utils.utils import clip_base_response, synchronize_touch_and_scale


def make_dataset_for_sock_calibration_withscale_touch_only(args, mask, touch_path, debug=0):

    # load data
    touch_raw, touch_fc, touch_ts = load_data_hdf5(touch_path, 'pressure')

    # do not filter the touch signal on sock as saturated signal is meaningful - body weight
    # only clip the base response
    touch_seq = clip_base_response(touch_raw, base_clip_percentile=5)

    if debug:
        plt.figure()

        idx = np.array([70, 98, 155])
        length = 1000

        plt.subplot(111)
        plt.plot(np.sum(touch_seq[:length], (1, 2)), 'b-')
        plt.plot(idx, np.sum(touch_seq[idx], (1, 2)), 'ro')

        plt.show()

        plt.close()

    # prepare training data according to observation window
    touch_raw_rec = []
    touch_rec = []
    touch_ts_rec = []

    if 'left' in args.knit_name:
        touch_mean = 7.9329
        touch_std = 29.3390

    elif 'right' in args.knit_name:
        touch_mean = 6.1001
        touch_std = 16.5276

    for i in range(touch_seq.shape[0]):
        touch_st_idx = i - args.obs_window // 2
        touch_ed_idx = i + args.obs_window // 2 + 1
        if touch_st_idx < 0 or touch_ed_idx > touch_seq.shape[0]:
            continue

        touch_raw_cur = touch_raw[i]
        touch_cur = (touch_seq[touch_st_idx:touch_ed_idx] - touch_mean) / touch_std
        touch_ts_cur = touch_ts[i]

        touch_raw_rec.append(touch_raw_cur)
        touch_rec.append(touch_cur)
        touch_ts_rec.append(touch_ts_cur)

    touch_raw_rec = np.stack(touch_raw_rec)
    touch_rec = np.stack(touch_rec)
    touch_ts_rec = np.stack(touch_ts_rec)
    print('touch_raw_rec', touch_raw_rec.shape,
          '%.4f %.4f %.4f %.4f' % (touch_raw_rec.mean(), touch_raw_rec.std(), touch_raw_rec.min(), touch_raw_rec.max()))
    print('touch_rec', touch_rec.shape,
          '%.4f %.4f %.4f %.4f' % (touch_rec.mean(), touch_rec.std(), touch_rec.min(), touch_rec.max()))
    print()

    return touch_raw_rec, touch_rec, touch_ts_rec



def make_dataset_for_sock_calibration_withscale(args, phase, data_path_prefix, mask, debug=0):
    print("Loading data from %s ..." % data_path_prefix)

    # load data
    scale_path = os.path.join(data_path_prefix, 'None.hdf5')
    touch_path = os.path.join(data_path_prefix, 'touch1.hdf5')

    touch_raw, touch_fc, touch_ts = load_data_hdf5(touch_path, 'pressure')
    scale_raw, scale_fc, scale_ts = load_data_hdf5(scale_path, 'scale')

    # do not filter the touch signal on sock as saturated signal is meaningful - body weight
    # only clip the base response
    touch_seq = clip_base_response(touch_raw, base_clip_percentile=5)

    # filter artifact in scale
    scale_seq = filter_artifact_in_scale(scale_raw, thresh=2.5e5)

    # synchronize touch and scale
    if 'left' in args.knit_name:
        offset = -1
    elif 'right' in args.knit_name:
        offset = 5
    touch_idx_per_round, scale_per_round = synchronize_touch_and_scale(
        touch_seq, scale_seq, touch_ts, scale_ts, offset=offset)

    if debug:
        plt.figure()

        idx = np.array([794, 1376, 2483, 2710])
        length = 3000

        plt.subplot(311)
        plt.plot(np.sum(touch_raw[touch_idx_per_round[:length]], (1, 2)), 'r-')
        plt.plot(idx, np.sum(touch_raw[touch_idx_per_round[idx]], (1, 2)), 'ro')

        plt.subplot(312)
        plt.plot(np.sum(touch_seq[touch_idx_per_round[:length]], (1, 2)), 'b-')
        plt.plot(idx, np.sum(touch_seq[touch_idx_per_round[idx]], (1, 2)), 'ro')

        plt.subplot(313)
        plt.plot(scale_per_round[:length], 'g-')
        plt.plot(idx, scale_per_round[idx], 'ro')

        plt.show()

        plt.close()

    # prepare training data according to observation window
    touch_raw_rec = []
    touch_rec = []
    scale_rec = []

    if 'left' in args.knit_name:
        touch_mean = 7.9329
        touch_std = 29.3390
        scale_mean = 593005.2296
        scale_std = 286031.0056
        scale_min = -0.9579

    elif 'right' in args.knit_name:
        touch_mean = 6.1001
        touch_std = 16.5276
        scale_mean = 685580.2034
        scale_std = 300010.6740
        scale_min = -1.3115


    if phase == 'train':
        st_idx, ed_idx = 65, int(len(scale_per_round) * args.train_valid_ratio)
    elif phase == 'valid':
        st_idx, ed_idx = int(len(scale_per_round) * args.train_valid_ratio), len(scale_per_round)
    elif phase == 'full':
        st_idx, ed_idx = 0, len(scale_per_round)

    for i in range(st_idx, ed_idx):
        touch_st_idx = touch_idx_per_round[i] - args.obs_window // 2
        touch_ed_idx = touch_idx_per_round[i] + args.obs_window // 2 + 1
        if touch_st_idx < 0 or touch_ed_idx > touch_raw.shape[0]:
            continue

        touch_raw_cur = touch_raw[touch_idx_per_round[i]]
        touch_cur = (touch_seq[touch_st_idx:touch_ed_idx] - touch_mean) / touch_std
        scale_cur = (scale_per_round[i] - scale_mean) / scale_std
        scale_cur = scale_cur - scale_min
        # scale_cur = (scale_per_round[i + args.obs_window // 2] - scale_min) / (scale_max - scale_min)

        touch_raw_rec.append(touch_raw_cur)
        touch_rec.append(touch_cur)
        scale_rec.append(scale_cur)

    touch_raw_rec = np.stack(touch_raw_rec)
    touch_rec = np.stack(touch_rec)
    scale_rec = np.stack(scale_rec)

    return touch_raw_rec, touch_rec, scale_rec



class KnittedGloveDataset(Dataset):

    def __init__(self, args, mask, phase):

        print("Initialize dataset %s, phase: %s ..." % (args.knit_name, phase))

        self.args = args
        self.phase = phase

        if 'left' in args.knit_name:
            data_list = ['rec_2019-12-18_15-18-17']
        elif 'right' in args.knit_name:
            data_list = ['rec_2019-12-26_10-15-13']

        self.touch_raw_rec, self.touch_rec, self.scale_rec = [], [], []

        for i in range(len(data_list)):
            data_path_prefix = os.path.join(args.data_path_prefix, data_list[i])

            touch_raw_rec, touch_rec, scale_rec = \
                    make_dataset_for_sock_calibration_withscale(
                        args, phase, data_path_prefix=data_path_prefix, mask=mask, debug=args.debug)

            self.touch_raw_rec.append(touch_raw_rec)
            self.touch_rec.append(touch_rec)
            self.scale_rec.append(scale_rec)

        self.touch_raw_rec = np.concatenate(self.touch_raw_rec, 0)
        self.touch_rec = np.concatenate(self.touch_rec, 0)
        self.scale_rec = np.concatenate(self.scale_rec, 0)

        print('touch_raw_rec', self.touch_raw_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.touch_raw_rec.mean(), self.touch_raw_rec.std(),
                                       self.touch_raw_rec.min(), self.touch_raw_rec.max()))
        print('touch_rec', self.touch_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.touch_rec.mean(), self.touch_rec.std(),
                                       self.touch_rec.min(), self.touch_rec.max()))
        print('scale_rec', self.scale_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.scale_rec.mean(), self.scale_rec.std(),
                                       self.scale_rec.min(), self.scale_rec.max()))

    def __len__(self):
        return self.scale_rec.shape[0]

    def __getitem__(self, idx):

        args = self.args

        touch_raw_data = torch.FloatTensor(self.touch_raw_rec[idx])
        touch_data = torch.FloatTensor(self.touch_rec[idx])
        scale_data = self.scale_rec[idx].astype(np.float32)

        return touch_raw_data, touch_data, scale_data

