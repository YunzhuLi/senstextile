# import open3d before torch to avoid conflicts
from knit_calib.visualization.visualizer import visualizer

import os
import random
import cv2
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
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LinearRegression

from data import KnittedGloveDataset
from config import gen_args

from knit_calib.utils.utils import to_np, set_seed, get_lr
from knit_calib.models.models import CNNCali


args = gen_args()
print(args)

use_gpu = torch.cuda.is_available()
set_seed(42)


'''
visualizer
'''
side = 'left' if 'left' in args.knit_name else 'right'
vis = visualizer('sock', side)



'''
model
'''
model = CNNCali(args, vis.mask)

if args.eval == 0:
    if args.resume == 1:
        model_name = 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter)
        model_path = os.path.join(args.ckp_path, model_name)
        print("Loading network from %s" % model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Randomly initialize the network")

if args.eval == 1:
    if args.epoch < 0:
        model_name = 'net_best.pth'
    else:
        model_name = 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter)

    model_path = os.path.join(args.ckp_path, model_name)
    print("Loading network from %s" % model_path)
    model.load_state_dict(torch.load(model_path))

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.imshow(model.position_bias.detach().numpy()[0, 0])
    plt.colorbar()
    plt.savefig(os.path.join(args.vis_path, 'position_bias.png'))
    plt.close()

if use_gpu:
    model = model.cuda()


'''
dataloader
'''
datasets = {}
dataloaders = {}

phases = ['train', 'valid'] if args.eval == 0 else ['valid']

for phase in phases:
    datasets[phase] = KnittedGloveDataset(args, vis.mask, phase=phase)
    dataloaders[phase] = DataLoader(
        datasets[phase],
        batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)


'''
criterion & optimizer
'''
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

# reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)


'''
training and evaluation
'''

best_valid = 1e10
store_video_idx = 20

n_epoch = args.n_epoch if args.eval == 0 else 1

for epoch in range(n_epoch):

    for phase in phases:

        model.train(phase == 'train')


        # set up recorder
        if args.eval == 1:
            if args.vis == 1:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_path = os.path.join(args.vis_path, '%d.avi' % epoch)
                print('Saving video to', video_path)
                out = cv2.VideoWriter(video_path, fourcc, 10., (1800, 600))

            touch_raw_rec, touch_rec, touch_cal_rec, scale_rec = \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros(len(datasets[phase]))


        # start training
        running_sample = 0.
        running_loss = 0.
        running_loss_scale = 0.
        running_loss_recon = 0.

        bar = ProgressBar(max_value=len(dataloaders[phase]))

        min_value, max_value = 1e10, -1e10
        for i, data in bar(enumerate(dataloaders[phase], 0)):

            touch_raw, touch, scale = data

            if use_gpu:
                touch_raw, touch, scale = touch_raw.cuda(), touch.cuda(), scale.cuda()

            with torch.set_grad_enabled(phase == 'train'):
                x_cal, x_recon = model(touch)

            min_value = min(min_value, x_recon.min().item())
            max_value = max(max_value, x_recon.max().item())

            scale_pred = torch.mean(x_cal, (1, 2, 3))

            loss_scale = F.mse_loss(scale_pred, scale)
            loss_recon = F.l1_loss(x_recon[:, 0], touch[:, args.obs_window//2])
            loss = loss_scale + loss_recon * args.lam_recon

            if phase == 'train':
                # optimzie the neural network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = scale.size(0)

            if args.eval == 1:
                # record the results
                touch_raw_cur = to_np(touch_raw)
                touch_cur = to_np(touch[:, args.obs_window // 2])
                result = to_np(x_recon[:, 0])
                scale_cur = to_np(scale)

                st = int(running_sample)
                touch_raw_rec[st:st + B] = touch_raw_cur
                touch_rec[st:st + B] = touch_cur
                touch_cal_rec[st:st + B] = result
                scale_rec[st:st + B] = scale_cur

                if args.vis == 1:
                    for idx_frame in range(B):
                        touch_raw_render = vis.render(
                            touch_raw_cur[idx_frame], lim_low=-2, lim_high=27, text='raw')
                        touch_render = vis.render(
                            touch_cur[idx_frame], lim_low=-2, lim_high=27, text='handcrafted')
                        result_render = vis.render(
                            result[idx_frame], lim_low=0., lim_high=0.4, text='selfsupervised')

                        frame = vis.merge_frames(
                            [touch_raw_render, touch_render, result_render], nx=1, ny=3)

                        out.write(frame)

            running_sample += B
            running_loss += loss.item() * B
            running_loss_scale += loss_scale.item() * B
            running_loss_recon += loss_recon.item() * B

            if i % args.log_per_iter == 0:
                print('[%d/%d][%d/%d] LR: %.6f, loss: %.4f (%.4f), scale: %.4f (%.4f), recon: %.4f (%.4f)' % (
                    epoch, args.n_epoch, i, len(dataloaders[phase]), get_lr(optimizer),
                    loss.item(), running_loss / running_sample,
                    loss_scale.item(), running_loss_scale / running_sample,
                    loss_recon.item(), running_loss_recon / running_sample))

            if i > 0 and i % args.ckp_per_iter == 0 and phase == 'train':
                model_path = '%s/net_epoch_%d_iter_%d.pth' % (args.ckp_path, epoch, i)
                torch.save(model.state_dict(), model_path)


        loss_cur = running_loss / running_sample
        loss_cur_scale = running_loss_scale / running_sample
        loss_cur_recon = running_loss_recon / running_sample
        print('[%d/%d %s] loss: %.4f, scale: %.4f, recon: %.4f, best_valid_loss: %.4f' % (
            epoch, args.n_epoch, phase, loss_cur, loss_cur_scale, loss_cur_recon, best_valid))
        print('[%d/%d %s] min_value: %.4f, max_value: %.4f, gBias: %.4f, gScale: %.4f' % (
            epoch, args.n_epoch, phase, min_value, max_value, model.global_bias, model.global_scale))

        if phase == 'valid' and not args.eval:
            print()
            scheduler.step(loss_cur)
            if loss_cur < best_valid:
                best_valid = loss_cur
                torch.save(model.state_dict(), '%s/net_best.pth' % (args.ckp_path))

        if args.eval == 1:
            if args.vis == 1:
                out.release()

            n_frames = 800

            from sklearn.metrics import roc_curve
            from sklearn.metrics import precision_recall_curve

            print('touch_raw_rec', touch_raw_rec.shape)
            print('touch_rec', touch_rec.shape)
            print('touch_cal_rec', touch_cal_rec.shape)
            print('scale_rec', scale_rec.shape)

            touch_raw_sum = np.sum(touch_raw_rec, (1, 2))
            touch_sum = np.sum(touch_rec, (1, 2))
            touch_cal_sum = np.sum(touch_cal_rec, (1, 2))

            precision, recall, thresh = [], [], []

            ### bar plot before/after calibration
            def autolabel(rects):
                """Attach a text label above each bar in *rects*, displaying its height."""
                for rect in rects:
                    height = rect.get_height()
                    height = np.round(height, 3)
                    ax.annotate('{}'.format(height),
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=20)

            cor_raw = np.corrcoef(scale_rec, touch_raw_sum)[0, 1]
            cor_hand = np.corrcoef(scale_rec, touch_sum)[0, 1]
            cor_cal = np.corrcoef(scale_rec, touch_cal_sum)[0, 1]

            plt.rcParams["figure.figsize"] = (4.2, 6)
            # plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(dpi=200)

            x = np.arange(3)
            width = 0.6
            rects1 = ax.bar(x[0], cor_raw, width, color='royalblue', label='Raw')
            rects2 = ax.bar(x[1], cor_hand, width, color='limegreen', label='Manual')
            rects3 = ax.bar(x[2], cor_cal, width, color='orangered', label='Self-supervised')
            plt.xlim(-0.5, 2.5)
            if 'left' in args.knit_name:
                plt.ylim(0.85, 1.01)
            elif 'right' in args.knit_name:
                plt.ylim(0.70, 1.01)
            plt.xticks([], [])
            ax.tick_params(labelsize=15)
            plt.legend(loc='upper left', fontsize=20)

            autolabel(rects1)
            autolabel(rects2)
            autolabel(rects3)

            plt.tight_layout(pad=0.5)
            plt.savefig(os.path.join(args.vis_path, 'touch_correlation_bar.png'))
            plt.savefig(os.path.join(args.vis_path, 'touch_correlation_bar.pdf'))

            ### scatter plot before/after calibration
            y = scale_rec
            X_raw = touch_raw_sum.reshape(-1, 1)
            X_hand = touch_sum.reshape(-1, 1)
            X_cal = touch_cal_sum.reshape(-1, 1)
            y_hat_raw = LinearRegression().fit(X_raw, y).predict(X_raw)
            y_hat_hand = LinearRegression().fit(X_hand, y).predict(X_hand)
            y_hat_cal = LinearRegression().fit(X_cal, y).predict(X_cal)

            plt.rcParams["figure.figsize"] = (6, 6)
            # plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(dpi=200)
            plt.scatter(y[::2], y_hat_raw[::2], s=30, c='royalblue', alpha=0.5, edgecolors='none',
                        label='Raw signal')
            plt.scatter(y[::2], y_hat_cal[::2], s=30, c='orangered', alpha=0.5, edgecolors='none',
                        label='Self-supervised calibration')
            min_y = np.min(y)
            max_y = np.max(y)
            plt.xlim(min_y, max_y)
            plt.ylim(min_y, max_y)
            plt.plot([min_y, max_y], [min_y, max_y], 'r-')
            plt.xticks([], [])
            plt.yticks([], [])
            ax.legend(fontsize=20)
            plt.tight_layout(pad=0.5)
            plt.savefig(os.path.join(args.vis_path, 'touch_raw_vs_cal.png'))
            plt.savefig(os.path.join(args.vis_path, 'touch_raw_vs_cal.pdf'))
            plt.close()


            # touch_raw
            if len(touch_raw_sum) > 0:
                plt.rcParams["figure.figsize"] = (6, 6)
                plt.figure()
                plt.plot(scale_rec, touch_raw_sum, 'bo', markersize=2)
                plt.savefig(os.path.join(args.vis_path, 'touch_correlation_raw.png'))
                plt.close()

                print("touch_raw corrcoef", np.corrcoef(scale_rec, touch_raw_sum)[0, 1])

                plt.rcParams["figure.figsize"] = (24, 6)
                plt.figure()
                plt.subplot(211)
                plt.plot(touch_raw_sum[:n_frames], 'r-')
                plt.subplot(212)
                plt.plot(scale_rec[:n_frames], 'b-')
                plt.savefig(os.path.join(args.vis_path, 'touch_overtime_raw.png'))
                plt.close()

                cor_raw_rec = []
                for x in range(32):
                    for y in range(32):
                        if vis.mask[x, y] == 0:
                            continue

                        max_cor = 0.
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                xx, yy = x + dx, y + dy
                                if xx < 0 or xx >= 32 or yy < 0 or yy >= 32:
                                    continue
                                if vis.mask[xx, yy] == 0:
                                    continue

                                cor = np.corrcoef(touch_raw_rec[:, x, y], touch_raw_rec[:, xx, yy])[0, 1]
                                max_cor = max(cor, max_cor)

                        cor_raw_rec.append(max_cor)

                p, r, t = precision_recall_curve(np.ones(len(cor_raw_rec)), cor_raw_rec)
                precision.append(p)
                recall.append(r)
                thresh.append(t)

            # touch
            if len(touch_sum) > 0:
                plt.rcParams["figure.figsize"] = (6, 6)
                plt.figure()
                plt.plot(scale_rec, touch_sum, 'bo', markersize=2)
                plt.savefig(os.path.join(args.vis_path, 'touch_correlation_hand.png'))
                plt.close()

                print("touch_hand corrcoef", np.corrcoef(scale_rec, touch_sum)[0, 1])

                plt.rcParams["figure.figsize"] = (24, 6)
                plt.figure()
                plt.subplot(211)
                plt.plot(touch_sum[:n_frames], 'r-')
                plt.subplot(212)
                plt.plot(scale_rec[:n_frames], 'b-')
                plt.savefig(os.path.join(args.vis_path, 'touch_overtime_hand.png'))
                plt.close()

                cor_hand_rec = []
                for x in range(32):
                    for y in range(32):
                        if vis.mask[x, y] == 0:
                            continue

                        max_cor = 0.
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                xx, yy = x + dx, y + dy
                                if xx < 0 or xx >= 32 or yy < 0 or yy >= 32:
                                    continue
                                if vis.mask[xx, yy] == 0:
                                    continue

                                cor = np.corrcoef(touch_rec[:, x, y], touch_rec[:, xx, yy])[0, 1]
                                max_cor = max(cor, max_cor)

                        cor_hand_rec.append(max_cor)

                p, r, t = precision_recall_curve(np.ones(len(cor_hand_rec)), cor_hand_rec)
                precision.append(p)
                recall.append(r)
                thresh.append(t)

            # touch_cal
            if len(touch_cal_sum) > 0:
                plt.rcParams["figure.figsize"] = (6, 6)
                plt.figure()
                plt.plot(scale_rec, touch_cal_sum, 'bo', markersize=2)
                plt.savefig(os.path.join(args.vis_path, 'touch_correlation_cal.png'))
                plt.close()

                print("touch_cal corrcoef", np.corrcoef(scale_rec, touch_cal_sum)[0, 1])

                plt.rcParams["figure.figsize"] = (24, 6)
                plt.figure()
                plt.subplot(211)
                plt.plot(touch_cal_sum[:n_frames], 'r-')
                plt.subplot(212)
                plt.plot(scale_rec[:n_frames], 'b-')
                plt.savefig(os.path.join(args.vis_path, 'touch_overtime_cal.png'))
                plt.close()

                cor_cal_rec = []
                for x in range(32):
                    for y in range(32):
                        if vis.mask[x, y] == 0:
                            continue

                        max_cor = 0.
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                if dx == 0 and dy == 0:
                                    continue
                                xx, yy = x + dx, y + dy
                                if xx < 0 or xx >= 32 or yy < 0 or yy >= 32:
                                    continue
                                if vis.mask[xx, yy] == 0:
                                    continue

                                if (touch_cal_rec[:, xx, yy] == 0).all() or (touch_cal_rec[:, x, y] == 0).all():
                                    continue

                                cor = np.corrcoef(touch_cal_rec[:, x, y], touch_cal_rec[:, xx, yy])[0, 1]
                                max_cor = max(cor, max_cor)

                        cor_cal_rec.append(max_cor)

                p, r, t = precision_recall_curve(np.ones(len(cor_cal_rec)), cor_cal_rec)
                precision.append(p)
                recall.append(r)
                thresh.append(t)

            plt.rcParams["figure.figsize"] = (6, 6)
            plt.figure()

            plt.plot(thresh[0], recall[0][1:], color='r', label='raw')
            plt.plot(thresh[1], recall[1][1:], color='g', label='hand')
            plt.plot(thresh[2], recall[2][1:], color='b', label='cal')
            plt.legend()
            plt.xlabel('Correlation')
            plt.ylabel('Recall')

            plt.savefig(os.path.join(args.vis_path, 'precision_recall.png'))
            plt.close()



