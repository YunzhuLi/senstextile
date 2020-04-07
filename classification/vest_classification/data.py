import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class VestDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase

        data = []
        label = []

        for idx_obj in range(args.n_obj):
            obj_name = args.object_list[idx_obj]

            file_path_prefix = os.path.join(args.data_path, '%s' % obj_name)
            if obj_name == 'sofa_straight':
                touch_seq = []
                for i in range(2):
                    file_path = file_path_prefix + '%d.hdf5' % (i + 1)
                    f = h5py.File(file_path, 'r')
                    fc = f['frame_count'][0]
                    touch_seq.append(np.array(f['pressure'][:fc]).astype(np.float32))
                touch_seq = np.concatenate(touch_seq, 0)
            else:
                file_path = file_path_prefix + '.hdf5'
                f = h5py.File(file_path, 'r')
                fc = f['frame_count'][0]
                touch_seq = np.array(f['pressure'][:fc]).astype(np.float32)

            # subsample the touch sequence
            if args.subsample > 1:
                subsample = args.subsample
                for x in range(0, touch_seq.shape[1], subsample):
                    for y in range(0, touch_seq.shape[2], subsample):
                        v = np.mean(touch_seq[:, x:x+subsample, y:y+subsample], (1, 2))
                        touch_seq[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

            touch = []
            for idx_data in range(0, touch_seq.shape[0] - args.input_window_size + 1, args.skip):
                touch.append(touch_seq[idx_data:idx_data + args.input_window_size])

            touch = np.stack(touch)
            n_reserve = args.n_valid_per_obj + args.n_test_per_obj
            n_train = touch.shape[0] - n_reserve

            if phase == 'train':
                print(obj_name, touch.shape)
                idx = np.random.choice(n_train, args.n_train_per_obj)
                touch = touch[idx]
            elif phase == 'valid':
                touch = touch[n_train:n_train + args.n_valid_per_obj]
            elif phase == 'test':
                touch = touch[-args.n_test_per_obj:]

            data.append(touch)
            label.append([idx_obj] * touch.shape[0])

        data = np.concatenate(data, 0)
        label = np.concatenate(label, 0)

        print(phase, 'data:', data.shape, 'label:', label.shape)
        print(np.min(data), np.max(data), np.mean(data), np.std(data))

        min_ = 26.0
        max_ = 1023.0
        mean_ = 561.9890334818384
        std_ = 105.46822622271125

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

