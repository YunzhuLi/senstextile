import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class GloveDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase

        data = []
        label = []

        for idx_obj in range(args.n_obj):
            obj_name = args.object_list[idx_obj]

            file_path_prefix = os.path.join(args.data_path, '%s' % obj_name)

            touch = []
            for idx_round in range(args.n_round):
                file_path = os.path.join(file_path_prefix, '%s_%d.hdf5'% (obj_name, idx_round + 1))
                f = h5py.File(file_path, 'r')
                fc = f['frame_count'][0]
                touch_seq = np.array(f['pressure'][:fc - 100]).astype(np.float32)

                # subsample the touch sequence
                if args.subsample > 1:
                    subsample = args.subsample
                    for x in range(0, touch_seq.shape[1], subsample):
                        for y in range(0, touch_seq.shape[2], subsample):
                            v = np.mean(touch_seq[:, x:x+subsample, y:y+subsample], (1, 2))
                            touch_seq[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)

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

        min_ = 48.0
        max_ = 1023.0
        mean_ = 525.35846
        std_ = 14.988879

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

