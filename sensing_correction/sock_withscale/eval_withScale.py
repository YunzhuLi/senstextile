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

from config import gen_args
from data import make_dataset_for_sock_calibration_withscale

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
processing eval list
'''
eval_list = open(args.eval_list, 'r').readlines()


for eval_path in eval_list:
    eval_path = eval_path.strip()
    eval_hdf5_path = os.path.join(args.data_path, eval_path)
    print('evaluating %s ...' % eval_hdf5_path)

    '''
    data
    '''
    data_path_prefix = eval_hdf5_path[:eval_hdf5_path.rfind('/')]
    touch_raw_rec, touch_rec, scale_rec = make_dataset_for_sock_calibration_withscale(
        args, 'full', data_path_prefix, vis.mask, debug=args.debug)

    touch_raw_rec = torch.FloatTensor(touch_raw_rec)
    touch_rec = torch.FloatTensor(touch_rec)
    scale_rec = torch.FloatTensor(scale_rec)

    print('touch_raw_rec size', touch_raw_rec.size())
    print('touch_rec size', touch_rec.size())
    print('scale_rec size', scale_rec.size())

    if use_gpu:
        touch_raw_rec = touch_raw_rec.cuda()
        touch_rec = touch_rec.cuda()
        scale_rec = scale_rec.cuda()


    '''
    calibrate
    '''
    batch_size = 64

    result = []
    scale_pred_raw = []
    scale_pred = []
    for i in range(0, touch_rec.shape[0], batch_size):
        touch = touch_rec[i:i+batch_size]
        with torch.set_grad_enabled(False):
            x_cal, x_recon = model(touch)

        result.append(x_recon.data.cpu().numpy())
        scale_pred_raw.append(torch.mean(touch_raw_rec[i:i+batch_size], (1, 2)).data.cpu().numpy())
        scale_pred.append(torch.mean(x_cal, (1, 2, 3)).data.cpu().numpy())

    result = np.concatenate(result, 0)
    scale_pred_raw = np.concatenate(scale_pred_raw, 0)
    scale_pred = np.concatenate(scale_pred, 0)
    print('result size', result.shape)
    print('scale_pred_raw size', scale_pred_raw.shape)
    print('scale_pred size', scale_pred.shape)

    touch_raw_rec = touch_raw_rec.data.cpu().numpy()
    touch_rec = touch_rec.data.cpu().numpy()
    scale_rec = scale_rec.data.cpu().numpy()

    '''
    visualize scale and scale_pred
    '''
    if args.vis_scale == 1:
        plt.rcParams["figure.figsize"] = (8, 2)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_path = os.path.join(args.vis_path, '%s_scale.avi' % eval_path.replace('/', '.'))
        image_path = os.path.join(args.vis_path, '%s_scale' % eval_path.replace('/', '.'))
        print('Saving video to', video_path)
        print('Saving image to', image_path)
        out = cv2.VideoWriter(video_path, fourcc, 10., (800, 200))
        os.system('mkdir -p ' + image_path)

        scale_rec = scale_rec - np.percentile(scale_rec, 5)
        scale_pred_raw = scale_pred_raw - np.percentile(scale_pred_raw, 5)
        scale_pred = scale_pred - np.percentile(scale_pred, 5)

        scale_rec = np.clip(scale_rec, a_min=0., a_max=None)
        scale_pred_raw = np.clip(scale_pred_raw, a_min=0., a_max=None)
        scale_pred = np.clip(scale_pred, a_min=0., a_max=None)

        percentile = 75
        scale_rec = scale_rec / np.percentile(scale_rec, percentile)
        scale_pred_raw = scale_pred_raw / np.percentile(scale_pred_raw, percentile)
        scale_pred = scale_pred / np.percentile(scale_pred, percentile)

        # set the previous N frames as 0
        scale_rec[:200 + 100] = 0.
        scale_pred[:200 + 100] = 0.

        print('scale_rec', scale_rec.shape, np.min(scale_rec), np.max(scale_rec))
        print('scale_pred_raw', scale_pred_raw.shape, np.min(scale_pred_raw), np.max(scale_pred_raw))
        print('scale_pred', scale_pred.shape, np.min(scale_pred), np.max(scale_pred))

        length = 100

        bar = ProgressBar(max_value=len(scale_rec) - length)
        for idx in bar(range(len(scale_rec) - length)):
            fig = plt.figure()
            ax = plt.subplot(111)
            plt.plot(np.arange(idx, idx + length), scale_rec[idx:idx + length], c='r')
            # plt.plot(np.arange(idx, idx + length), scale_pred_raw[idx:idx + length], c='b')
            plt.plot(np.arange(idx, idx + length), scale_pred[idx:idx + length], c='b')
            ax.set_xlim([idx, idx + length + 20])
            ax.set_ylim([-0.05, 4.])

            plt.xticks([], [])
            plt.yticks([], [])

            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

            cv2.imwrite(os.path.join(image_path, 'scale_%d.png' % idx), img)

            # plt.show()
            plt.gcf().clear()
            plt.close()

            out.write(img)

        out.release()


    '''
    store results
    '''
    if args.store == 1:
        store_path = os.path.join(args.vis_path, eval_path)
        os.system('mkdir -p %s' % store_path[:store_path.rfind('/')])

        hf = h5py.File(store_path, 'w')
        hf.create_dataset('pressure', data=result[:, 0])
        hf.create_dataset('frame_count', data=np.array([result.shape[0]]))
        hf.close()


    '''
    visualize
    '''
    if args.vis == 1:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_path = os.path.join(args.vis_path, '%s.avi' % eval_path.replace('/', '.'))
        image_path = os.path.join(args.vis_path, '%s' % eval_path.replace('/', '.'))
        print('Saving video to', video_path)
        print('Saving image to', image_path)
        out = cv2.VideoWriter(video_path, fourcc, 14., (1200, 600))
        os.system('mkdir -p ' + image_path)

        print('min: %.4f, max: %.4f' % (np.min(result), np.max(result)))
        lim_low = np.min(result)
        lim_high = np.max(result) * 2. / 3.


        bar = ProgressBar(max_value=touch_rec.shape[0])
        for i in bar(range(touch_rec.shape[0])):
            touch_raw_cur = touch_raw_rec[i]
            touch_cur = touch_rec[i, args.obs_window // 2]
            touch_cal = result[i, 0]

            fancy = True
            touch_raw_render = vis.render(
                touch_raw_cur, lim_low=np.min(np.median(touch_raw_rec, 0)) + 10., lim_high=800, text='raw', fancy=fancy)
            '''
            touch_render = vis.render(
                touch_cur, lim_low=0., lim_high=24, text='raw', fancy=fancy)
            '''
            result_render = vis.render(
                touch_cal, lim_low=lim_low, lim_high=lim_high, text='selfsupervised', fancy=fancy)

            cv2.imwrite(os.path.join(image_path, 'raw_%d.png' % i), touch_raw_render)
            cv2.imwrite(os.path.join(image_path, 'cal_%d.png' % i), result_render)

            frame = vis.merge_frames(
                [touch_raw_render, result_render], nx=1, ny=2)

            out.write(frame)

        out.release()

