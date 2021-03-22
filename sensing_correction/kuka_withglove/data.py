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
from knit_calib.utils.utils import clip_base_response, synchronize_touch_and_touch



def make_dataset_for_kuka_calibration_withglove_touch_only(args, mask, touch_path, debug=0):

    # load data
    touch_raw, touch_fc, touch_ts = load_data_hdf5(touch_path, 'pressure')

    # filter artifact in touch
    '''
    touch = filter_artifact_in_touch(touch_raw, mask, thresh=1000., t_win=5, s_neighbor=1)
    '''

    # clip base response
    touch_seq = clip_base_response(touch_raw, base_clip_percentile=50)

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

    touch_mean = 4.4101
    touch_std = 21.4518

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



def make_dataset_for_kuka_calibration_withglove(args, phase, data_path_prefix, mask_glove, mask_kuka, debug=0, max_length=-1):

    print("Loading data from %s ..." % data_path_prefix)

    # load data
    touch_glove_path = os.path.join(data_path_prefix, 'touch_1.hdf5')
    touch_kuka_path = os.path.join(data_path_prefix, 'touch_0.hdf5')

    touch_glove_raw, touch_glove_fc, touch_glove_ts = load_data_hdf5(touch_glove_path, 'pressure', max_length=max_length)
    touch_kuka_raw, touch_kuka_fc, touch_kuka_ts = load_data_hdf5(touch_kuka_path, 'pressure')


    # filter artifact in touch_glove
    touch_glove = filter_artifact_in_touch(
        touch_glove_raw, mask_glove, thresh=1000., t_win=5, s_neighbor=1)

    # clip base response for touch_glove
    touch_glove_seq = clip_base_response(touch_glove, base_clip_percentile=50)


    # filter artifact in touch_kuka
    '''
    touch_kuka = filter_artifact_in_touch(
        touch_kuka_raw, mask_kuka, thresh=1000., t_win=5, s_neighbor=1)
    '''

    # clip base response in touch_kuka
    touch_kuka_seq = clip_base_response(touch_kuka_raw, base_clip_percentile=50)


    # synchronize touch_glove and touch_kuka
    touch_glove_idx_per_round, touch_kuka_idx_per_round = synchronize_touch_and_touch(
        touch_glove_seq, touch_kuka_seq, touch_glove_ts, touch_kuka_ts, offset=-1)

    if debug:
        plt.figure()

        idx = np.array([140, 162, 280, 375])
        length = 400

        plt.subplot(411)
        plt.plot(np.sum(touch_glove_raw[touch_glove_idx_per_round[:length]], (1, 2)), 'r-')
        plt.plot(idx, np.sum(touch_glove_raw[touch_glove_idx_per_round[idx]], (1, 2)), 'ro')

        plt.subplot(412)
        plt.plot(np.sum(touch_glove_seq[touch_glove_idx_per_round[:length]], (1, 2)), 'b-')
        plt.plot(idx, np.sum(touch_glove_seq[touch_glove_idx_per_round[idx]], (1, 2)), 'ro')

        plt.subplot(413)
        plt.plot(np.sum(touch_kuka_raw[touch_kuka_idx_per_round[:length]], (1, 2)), 'r-')
        plt.plot(idx, np.sum(touch_kuka_raw[touch_kuka_idx_per_round[idx]], (1, 2)), 'ro')

        plt.subplot(414)
        plt.plot(np.sum(touch_kuka_seq[touch_kuka_idx_per_round[:length]], (1, 2)), 'b-')
        plt.plot(idx, np.sum(touch_kuka_seq[touch_kuka_idx_per_round[idx]], (1, 2)), 'ro')

        plt.show()

        plt.close()

    # prepare training data according to observation window
    touch_glove_raw_rec = []
    touch_glove_rec = []
    touch_kuka_raw_rec = []
    touch_kuka_rec = []

    touch_glove_mean = 2.3655
    touch_glove_std = 14.1341

    touch_kuka_mean = 4.4101
    touch_kuka_std = 21.4518

    if phase == 'train':
        st_idx, ed_idx = 65, int(len(touch_glove_idx_per_round) * args.train_valid_ratio)
    else:
        st_idx = int(len(touch_glove_idx_per_round) * args.train_valid_ratio)
        ed_idx = len(touch_glove_idx_per_round)

    for i in range(st_idx, ed_idx):
        touch_glove_st_idx = touch_glove_idx_per_round[i] - args.obs_window // 2
        touch_glove_ed_idx = touch_glove_idx_per_round[i] + args.obs_window // 2 + 1
        if touch_glove_st_idx < 0 or touch_glove_ed_idx > touch_glove_raw.shape[0]:
            continue

        touch_kuka_st_idx = touch_kuka_idx_per_round[i] - args.obs_window // 2
        touch_kuka_ed_idx = touch_kuka_idx_per_round[i] + args.obs_window // 2 + 1
        if touch_kuka_st_idx < 0 or touch_kuka_ed_idx > touch_kuka_raw.shape[0]:
            continue

        touch_glove_raw_cur = touch_glove_raw[touch_glove_idx_per_round[i]]
        touch_kuka_raw_cur = touch_kuka_raw[touch_kuka_idx_per_round[i]]
        touch_glove_cur = (touch_glove_seq[touch_glove_st_idx:touch_glove_ed_idx] - touch_glove_mean) / touch_glove_std
        touch_kuka_cur = (touch_kuka_seq[touch_kuka_st_idx:touch_kuka_ed_idx] - touch_kuka_mean) / touch_kuka_std

        touch_glove_raw_rec.append(touch_glove_raw_cur)
        touch_kuka_raw_rec.append(touch_kuka_raw_cur)
        touch_glove_rec.append(touch_glove_cur)
        touch_kuka_rec.append(touch_kuka_cur)

    touch_glove_raw_rec = np.stack(touch_glove_raw_rec)
    touch_glove_rec = np.stack(touch_glove_rec)
    touch_kuka_raw_rec = np.stack(touch_kuka_raw_rec)
    touch_kuka_rec = np.stack(touch_kuka_rec)

    return touch_glove_raw_rec, touch_glove_rec, touch_kuka_raw_rec, touch_kuka_rec



class KnittedVestDataset(Dataset):

    def __init__(self, args, mask_glove, mask_kuka, phase):

        print("Initialize dataset %s, phase: %s ..." % (args.knit_name, phase))

        self.args = args
        self.phase = phase

        data_list = ['rec_2019-12-12_17-03-25', 'rec_2019-12-12_17-17-46']
        max_length_list = [8000, -1]

        self.touch_glove_raw_rec, self.touch_glove_rec = [], []
        self.touch_kuka_raw_rec, self.touch_kuka_rec = [], []

        for i in range(len(data_list)):
            data_path_prefix = os.path.join(args.data_path_prefix, data_list[i])

            touch_glove_raw_rec, touch_glove_rec, touch_kuka_raw_rec, touch_kuka_rec = \
                    make_dataset_for_kuka_calibration_withglove(
                        args, phase, data_path_prefix=data_path_prefix, debug=args.debug,
                        mask_kuka=mask_kuka, mask_glove=mask_glove, max_length=max_length_list[i])

            self.touch_glove_raw_rec.append(touch_glove_raw_rec)
            self.touch_glove_rec.append(touch_glove_rec)
            self.touch_kuka_raw_rec.append(touch_kuka_raw_rec)
            self.touch_kuka_rec.append(touch_kuka_rec)

        self.touch_glove_raw_rec = np.concatenate(self.touch_glove_raw_rec, 0)
        self.touch_glove_rec = np.concatenate(self.touch_glove_rec, 0)
        self.touch_kuka_raw_rec = np.concatenate(self.touch_kuka_raw_rec, 0)
        self.touch_kuka_rec = np.concatenate(self.touch_kuka_rec, 0)

        print('touch_glove_raw_rec', self.touch_glove_raw_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.touch_glove_raw_rec.mean(), self.touch_glove_raw_rec.std(),
                                       self.touch_glove_raw_rec.min(), self.touch_glove_raw_rec.max()))
        print('touch_glove_rec', self.touch_glove_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.touch_glove_rec.mean(), self.touch_glove_rec.std(),
                                       self.touch_glove_rec.min(), self.touch_glove_rec.max()))

        print('touch_kuka_raw_rec', self.touch_kuka_raw_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.touch_kuka_raw_rec.mean(), self.touch_kuka_raw_rec.std(),
                                       self.touch_kuka_raw_rec.min(), self.touch_kuka_raw_rec.max()))
        print('touch_kuka_rec', self.touch_kuka_rec.shape,
              '%.4f %.4f %.4f %.4f' % (self.touch_kuka_rec.mean(), self.touch_kuka_rec.std(),
                                       self.touch_kuka_rec.min(), self.touch_kuka_rec.max()))

    def __len__(self):
        return self.touch_glove_rec.shape[0]

    def __getitem__(self, idx):

        args = self.args

        touch_glove_raw = torch.FloatTensor(self.touch_glove_raw_rec[idx])
        touch_glove = torch.FloatTensor(self.touch_glove_rec[idx])
        touch_kuka_raw = torch.FloatTensor(self.touch_kuka_raw_rec[idx])
        touch_kuka = torch.FloatTensor(self.touch_kuka_rec[idx])

        return touch_glove_raw, touch_glove, touch_kuka_raw, touch_kuka

