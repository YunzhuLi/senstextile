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

from config import gen_args
from data import KnittedVestDataset

from knit_calib.utils.utils import to_np, set_seed, get_lr
from knit_calib.models.models import CNNCali


args = gen_args()
print(args)

use_gpu = torch.cuda.is_available()
set_seed(42)


'''
visualizer
'''
if args.knit_name in ['vest_calibration']:
    vis_glove = visualizer('glove', 'right')
    vis_vest = visualizer('vest', 'back')


'''
model
'''
model_glove = CNNCali(args, vis_glove.mask, scale_factor=1.0, position_bias=True)
model_vest = CNNCali(args, vis_vest.mask)

# load pretrained glove network
model_name = 'net_glove_pretrained.pth'
model_path = os.path.join(args.ckp_path, model_name)
print("Loading pretrained glove network from %s" % model_path)
model_glove.load_state_dict(torch.load(model_path))


if args.eval == 0:
    if args.resume == 1:
        # load vest network
        model_name = 'net_vest_epoch_%d_iter_%d.pth' % (args.epoch, args.iter)
        model_path = os.path.join(args.ckp_path, model_name)
        print("Loading vest network from %s" % model_path)
        model_vest.load_state_dict(torch.load(model_path))
    else:
        print("Randomly initialize the vest network")


if args.eval == 1:
    if args.epoch < 0:
        model_vest_name = 'net_vest_best.pth'
    else:
        model_vest_name = 'net_vest_epoch_%d_iter_%d.pth' % (args.epoch, args.iter)

    model_path = os.path.join(args.ckp_path, model_vest_name)
    print("Loading vest network from %s" % model_path)
    model_vest.load_state_dict(torch.load(model_path))

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.imshow(model_vest.position_bias.detach().numpy()[0, 0])
    plt.colorbar()
    plt.savefig(os.path.join(args.vis_path, 'position_bias_vest.png'))
    plt.close()


if use_gpu:
    model_glove = model_glove.cuda()
    model_vest = model_vest.cuda()


'''
dataloader
'''
datasets = {}
dataloaders = {}

phases = ['train', 'valid'] if args.eval == 0 else ['valid']

for phase in phases:
    datasets[phase] = KnittedVestDataset(
        args, vis_glove.mask, vis_vest.mask, phase=phase)
    dataloaders[phase] = DataLoader(
        datasets[phase],
        batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)


'''
criterion & optimizer
'''
criterion = nn.MSELoss()
params = model_vest.parameters()
optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))

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

        model_glove.train(phase == 'train')
        model_vest.train(phase == 'train')


        # set up recorder
        if args.eval == 1:
            if args.vis == 1:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_path = os.path.join(args.vis_path, '%d.avi' % epoch)
                print('Saving video to', video_path)
                out = cv2.VideoWriter(video_path, fourcc, 10., (1200, 1200))

            touch_glove_raw_rec, touch_glove_rec, touch_glove_cal_rec = \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros((len(datasets[phase]), 32, 32))

            touch_vest_raw_rec, touch_vest_rec, touch_vest_cal_rec = \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros((len(datasets[phase]), 32, 32)), \
                    np.zeros((len(datasets[phase]), 32, 32))



        # start training
        running_sample = 0.
        running_loss = 0.
        running_loss_scale = 0.
        running_loss_recon = 0.

        bar = ProgressBar(max_value=len(dataloaders[phase]))

        min_value, max_value = 1e10, -1e10
        for i, data in bar(enumerate(dataloaders[phase], 0)):

            if args.knit_name in ['vest_calibration']:
                if use_gpu:
                    data = [d.cuda() for d in data]

                touch_glove_raw, touch_glove, touch_vest_raw, touch_vest = data

                with torch.set_grad_enabled(phase == 'train'):
                    glove_cal, glove_recon = model_glove(touch_glove)
                    vest_cal, vest_recon = model_vest(touch_vest)

                min_value = min(min_value, vest_recon.min().item())
                max_value = max(max_value, vest_recon.max().item())

                glove_scale = torch.mean(glove_cal, (1, 2, 3))
                vest_scale = torch.mean(vest_cal, (1, 2, 3))

                loss_scale = F.mse_loss(glove_scale, vest_scale)
                loss_recon = F.l1_loss(vest_recon[:, 0], touch_vest[:, args.obs_window//2])
                loss = loss_scale + loss_recon * args.lam_recon


            if phase == 'train':
                # optimzie the neural network
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            B = touch_vest.size(0)

            if args.eval == 1:
                # record the results
                touch_glove_raw_cur = to_np(touch_glove_raw)
                touch_glove_cur = to_np(touch_glove[:, args.obs_window // 2])
                touch_glove_cal_cur = to_np(glove_recon[:, 0])

                touch_vest_raw_cur = to_np(touch_vest_raw)
                touch_vest_cur = to_np(touch_vest[:, args.obs_window // 2])
                touch_vest_cal_cur = to_np(vest_recon[:, 0])

                st = int(running_sample)
                touch_glove_raw_rec[st:st + B] = touch_glove_raw_cur
                touch_glove_rec[st:st + B] = touch_glove_cur
                touch_glove_cal_rec[st:st + B] = touch_glove_cal_cur

                touch_vest_raw_rec[st:st + B] = touch_vest_raw_cur
                touch_vest_rec[st:st + B] = touch_vest_cur
                touch_vest_cal_rec[st:st + B] = touch_vest_cal_cur

                if args.vis == 1:
                    for idx_frame in range(B):
                        touch_glove_raw_render = vis_glove.render(
                            touch_glove_raw_cur[idx_frame], lim_low=500, lim_high=800, text='raw')
                        touch_glove_render = vis_glove.render(
                            touch_glove_cur[idx_frame], lim_low=-2, lim_high=27, text='selfsupervised')

                        touch_vest_raw_render = vis_vest.render(
                            touch_vest_raw_cur[idx_frame], lim_low=500, lim_high=800, text='raw')
                        touch_vest_render = vis_vest.render(
                            touch_vest_cur[idx_frame], lim_low=-2, lim_high=27, text='selfsupervised')

                        frame = vis_glove.merge_frames(
                            [touch_glove_raw_render, touch_glove_render,
                             touch_vest_raw_render, touch_vest_render], nx=2, ny=2)

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
                model_path = '%s/net_vest_epoch_%d_iter_%d.pth' % (args.ckp_path, epoch, i)
                torch.save(model_vest.state_dict(), model_path)



        loss_cur = running_loss / running_sample
        loss_cur_scale = running_loss_scale / running_sample
        loss_cur_recon = running_loss_recon / running_sample
        print('[%d/%d %s] loss: %.4f, scale: %.4f, recon: %.4f, best_valid_loss: %.4f' % (
            epoch, args.n_epoch, phase, loss_cur, loss_cur_scale, loss_cur_recon, best_valid))
        print('[%d/%d %s] min_value: %.4f, max_value: %.4f, gBias: %.4f, gScale: %.4f' % (
            epoch, args.n_epoch, phase, min_value, max_value, model_vest.global_bias, model_vest.global_scale))

        if phase == 'valid' and not args.eval:
            print()
            scheduler.step(loss_cur)
            if loss_cur < best_valid:
                best_valid = loss_cur
                torch.save(model_vest.state_dict(), '%s/net_vest_best.pth' % (args.ckp_path))

        if args.eval == 1:
            if args.vis == 1:
                out.release()

            n_frames = 800

            print('touch_glove_raw_rec', touch_glove_raw_rec.shape)
            print('touch_glove_rec', touch_glove_rec.shape)
            print('touch_glove_cal_rec', touch_glove_cal_rec.shape)

            print('touch_vest_raw_rec', touch_vest_raw_rec.shape)
            print('touch_vest_rec', touch_vest_rec.shape)
            print('touch_vest_cal_rec', touch_vest_cal_rec.shape)

            touch_glove_cal_sum = np.sum(touch_glove_cal_rec, (1, 2))

            touch_vest_raw_sum = np.sum(touch_vest_raw_rec, (1, 2))
            touch_vest_sum = np.sum(touch_vest_rec, (1, 2))
            touch_vest_cal_sum = np.sum(touch_vest_cal_rec, (1, 2))

            # remove outliers
            mask_inliers = np.logical_not(touch_glove_cal_sum > 380)
            touch_glove_cal_sum = touch_glove_cal_sum[mask_inliers]
            touch_vest_raw_sum = touch_vest_raw_sum[mask_inliers]
            touch_vest_sum = touch_vest_sum[mask_inliers]
            touch_vest_cal_sum = touch_vest_cal_sum[mask_inliers]


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

            cor_raw = np.corrcoef(touch_glove_cal_sum, touch_vest_raw_sum)[0, 1]
            cor_hand = np.corrcoef(touch_glove_cal_sum, touch_vest_sum)[0, 1]
            cor_cal = np.corrcoef(touch_glove_cal_sum, touch_vest_cal_sum)[0, 1]

            plt.rcParams["figure.figsize"] = (4.2, 6)
            # plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(dpi=200)

            x = np.arange(3)
            width = 0.6
            rects1 = ax.bar(x[0], cor_raw, width, color='royalblue', label='Raw')
            rects2 = ax.bar(x[1], cor_hand, width, color='limegreen', label='Manual')
            rects3 = ax.bar(x[2], cor_cal, width, color='orangered', label='Self-supervised')
            plt.xlim(-0.5, 2.5)
            plt.ylim(0.2, 1.01)
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
            y = touch_glove_cal_sum
            X_raw = touch_vest_raw_sum.reshape(-1, 1)
            X_hand = touch_vest_sum.reshape(-1, 1)
            X_cal = touch_vest_cal_sum.reshape(-1, 1)
            y_hat_raw = LinearRegression().fit(X_raw, y).predict(X_raw)
            y_hat_hand = LinearRegression().fit(X_hand, y).predict(X_hand)
            y_hat_cal = LinearRegression().fit(X_cal, y).predict(X_cal)

            plt.rcParams["figure.figsize"] = (6, 6)
            # plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(dpi=200)
            plt.scatter(y[::5], y_hat_raw[::5], s=30, c='royalblue', alpha=0.5, edgecolors='none',
                        label='Raw signal')
            plt.scatter(y[::5], y_hat_cal[::5], s=30, c='orangered', alpha=0.5, edgecolors='none',
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
            if len(touch_vest_raw_sum) > 0:
                plt.rcParams["figure.figsize"] = (6, 6)
                plt.figure()
                plt.plot(touch_glove_cal_sum, touch_vest_raw_sum, 'bo', markersize=2)
                plt.savefig(os.path.join(args.vis_path, 'touch_correlation_raw.png'))
                plt.close()

                print("touch_raw corrcoef", np.corrcoef(touch_glove_cal_sum, touch_vest_raw_sum)[0, 1])

                plt.rcParams["figure.figsize"] = (24, 6)
                plt.figure()
                plt.subplot(211)
                plt.plot(touch_glove_cal_sum[:n_frames], 'r-')
                plt.subplot(212)
                plt.plot(touch_vest_raw_sum[:n_frames], 'b-')
                plt.savefig(os.path.join(args.vis_path, 'touch_overtime_raw.png'))
                plt.close()

            # touch
            if len(touch_vest_sum) > 0:
                plt.rcParams["figure.figsize"] = (6, 6)
                plt.figure()
                plt.plot(touch_glove_cal_sum, touch_vest_sum, 'bo', markersize=2)
                plt.savefig(os.path.join(args.vis_path, 'touch_correlation_hand.png'))
                plt.close()

                print("touch_hand corrcoef", np.corrcoef(touch_glove_cal_sum, touch_vest_sum)[0, 1])

                plt.rcParams["figure.figsize"] = (24, 6)
                plt.figure()
                plt.subplot(211)
                plt.plot(touch_glove_cal_sum[:n_frames], 'r-')
                plt.subplot(212)
                plt.plot(touch_vest_sum[:n_frames], 'b-')
                plt.savefig(os.path.join(args.vis_path, 'touch_overtime_hand.png'))
                plt.close()

            # touch_cal
            if len(touch_vest_cal_sum) > 0:
                plt.rcParams["figure.figsize"] = (6, 6)
                plt.figure()
                plt.plot(touch_glove_cal_sum, touch_vest_cal_sum, 'bo', markersize=2)
                plt.savefig(os.path.join(args.vis_path, 'touch_correlation_cal.png'))
                plt.close()

                print("touch_cal corrcoef", np.corrcoef(touch_glove_cal_sum, touch_vest_cal_sum)[0, 1])

                plt.rcParams["figure.figsize"] = (24, 6)
                plt.figure()
                plt.subplot(211)
                plt.plot(touch_glove_cal_sum[:n_frames], 'r-')
                plt.subplot(212)
                plt.plot(touch_vest_cal_sum[:n_frames], 'b-')
                plt.savefig(os.path.join(args.vis_path, 'touch_overtime_cal.png'))
                plt.close()


