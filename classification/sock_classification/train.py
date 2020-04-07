import os
import cv2
import copy
import h5py
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from config import gen_args
from models import CNNet
from data import ActionDataset
from utils import plot_confusion_matrix, accuracy

args = gen_args()
print(args)

use_gpu = torch.cuda.is_available()


'''
dataset
'''
datasets = {}
dataloaders = {}
for phase in ['train', 'valid', 'test']:
    datasets[phase] = ActionDataset(args, phase=phase)
    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)


'''
model
'''
model = CNNet(args)

if use_gpu:
    model = model.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)


best_top1 = {}
best_top3 = {}
for phase in ['valid', 'test']:
    best_top1[phase] = 0.
    best_top3[phase] = 0.

for epoch in range(args.n_epoch):  # loop over the dataset multiple times

    phases = ['train', 'valid', 'test']
    for phase in phases:

        model.train(phase == 'train')

        running_loss = 0.0
        running_top1, running_top3 = 0, 0
        pred_rec, true_rec = [], []

        bar = ProgressBar(max_value=len(dataloaders[phase]))

        for i, data in bar(enumerate(dataloaders[phase], 0)):
            inputs, labels = data

            if use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()


            if phase == 'test':
                outputs = model_best(inputs)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print statistics
            running_loss += loss.item()

            # record the prediction
            _, predicted = torch.max(outputs.data, 1)
            pred_rec.append(predicted.data.cpu().numpy().astype(np.int))
            true_rec.append(labels.data.cpu().numpy().astype(np.int))

            # record the topk accuracy
            top1, top3 = accuracy(outputs, labels, topk=(1, 3))
            running_top1 += top1
            running_top3 += top3

            if i > 0 and i % args.log_per_iter == 0:    # print every 2000 mini-batches
                print()
                print('[%d/%d] loss: %.4f (%.4f), acc: top1 %.4f (%.4f) top3 %.4f (%.4f)' % (
                    epoch, i,
                    loss.item(), running_loss / (i + 1),
                    top1, running_top1 / (i + 1),
                    top3, running_top3 / (i + 1)))

        running_loss = running_loss / len(dataloaders[phase])
        top1_cur = running_top1 / len(dataloaders[phase])
        top3_cur = running_top3 / len(dataloaders[phase])


        if phase in ['valid', 'test']:
            # scheduler.step(running_loss)

            if top1_cur >= best_top1[phase]:
                best_top1[phase] = top1_cur
                best_top3[phase] = top3_cur

                if phase == 'valid':
                    model_best = copy.deepcopy(model)

            print('[%d, %s] loss: %.4f, acc: top1 %.4f top3 %.4f, best_acc: top1 %.4f top3 %.4f' % (
                epoch, phase, running_loss, top1_cur, top3_cur, best_top1[phase], best_top3[phase]))

            pred_rec = np.concatenate(pred_rec)
            true_rec = np.concatenate(true_rec)

            plot_confusion_matrix(true_rec, pred_rec, args.object_list)

            plt.savefig(os.path.join(args.rec_path, '%s_%d.pdf' % (phase, epoch)))
            plt.close()

        else:
            print('[%d, %s] loss: %.4f, acc: top1 %.4f top3 %.4f' % (
                epoch, phase, running_loss, top1_cur, top3_cur))

    print()

