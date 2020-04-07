import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from knit_calib.utils.utils import synchronize_touch_and_touch


class ActionDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase

        data = []
        label = []

        for idx_obj in range(args.n_obj):
            obj_name = args.object_list[idx_obj]
            n_round = args.n_rounds[idx_obj]

            touch_0, touch_1 = [], []
            for idx_round in range(n_round):
                file_path_0 = os.path.join(
                    args.data_path, '%s' % obj_name,
                    '%s_%d' % (obj_name, idx_round + 1), 'touch_0.hdf5')
                f = h5py.File(file_path_0, 'r')
                fc = f['frame_count'][0]
                touch_0_ts = f['ts'][:fc]
                touch_0_seq = np.array(f['pressure'][:fc]).astype(np.float32)

                file_path_1 = os.path.join(
                    args.data_path, '%s' % obj_name,
                    '%s_%d' % (obj_name, idx_round + 1), 'touch_1.hdf5')
                f = h5py.File(file_path_1, 'r')
                fc = f['frame_count'][0]
                touch_1_ts = f['ts'][:fc]
                touch_1_seq = np.array(f['pressure'][:fc]).astype(np.float32)

                # subsample the touch sequence
                if args.subsample > 1:
                    subsample = args.subsample
                    for x in range(0, touch_0_seq.shape[1], subsample):
                        for y in range(0, touch_0_seq.shape[2], subsample):
                            v = np.mean(touch_0_seq[:, x:x+subsample, y:y+subsample], (1, 2))
                            touch_0_seq[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)
                    for x in range(0, touch_1_seq.shape[1], args.subsample):
                        for y in range(0, touch_1_seq.shape[2], args.subsample):
                            v = np.mean(touch_1_seq[:, x:x+subsample, y:y+subsample], (1, 2))
                            touch_1_seq[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

                touch_0_idx, touch_1_idx = synchronize_touch_and_touch(
                    touch_0_seq, touch_1_seq, touch_0_ts, touch_1_ts, offset=0)

                for idx_data in range(0, touch_0_idx.shape[0], args.skip):
                    idx_0 = touch_0_idx[idx_data]
                    idx_1 = touch_1_idx[idx_data]
                    if idx_0 + args.input_window_size > touch_0_seq.shape[0]:
                        break
                    if idx_1 + args.input_window_size > touch_1_seq.shape[0]:
                        break
                    touch_0.append(touch_0_seq[idx_0:idx_0 + args.input_window_size])
                    touch_1.append(touch_1_seq[idx_1:idx_1 + args.input_window_size])

            touch_0 = np.stack(touch_0)
            touch_1 = np.stack(touch_1)

            n_reserve = args.n_valid_per_obj + args.n_test_per_obj
            n_train = touch_0.shape[0] - n_reserve

            if phase == 'train':
                print(obj_name, touch_0.shape, touch_1.shape)
                idx = np.random.choice(n_train, args.n_train_per_obj)
                touch_0 = touch_0[idx]
                touch_1 = touch_1[idx]
            elif phase == 'valid':
                touch_0 = touch_0[n_train:n_train + args.n_valid_per_obj]
                touch_1 = touch_1[n_train:n_train + args.n_valid_per_obj]
            elif phase == 'test':
                touch_0 = touch_0[-args.n_test_per_obj:]
                touch_1 = touch_1[-args.n_test_per_obj:]

            for idx_data in range(touch_0.shape[0]):
                touch = np.swapaxes(np.stack([touch_0[idx_data], touch_1[idx_data]]), 0, 1)
                data.append(touch.reshape(args.input_window_size * 2, touch.shape[2], touch.shape[3]))
                label.append(idx_obj)

        data = np.array(data)
        label = np.array(label)

        print(phase, 'data:', data.shape, 'label:', label.shape)
        print(np.min(data), np.max(data), np.mean(data), np.std(data))

        min_ = 89.0
        max_ = 1011.0
        mean_ = 536.4649528748312
        std_ = 46.81434484100088

        # data = ((data - min_) / (max_ - min_) - 0.5) * 2.
        data = (data - mean_) / std_

        self.data = torch.FloatTensor(data)
        self.label = torch.LongTensor(label)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        data = self.data[idx]

        # augment the training set
        if self.phase == 'train':
            noise = (np.random.randn(data.shape[0], data.shape[1], data.shape[2]) - 0.5) * 0.5
            data = data + torch.FloatTensor(noise)

        return data, self.label[idx]

