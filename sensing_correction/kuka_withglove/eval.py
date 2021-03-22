# import open3d before torch to avoid conflicts
from visualizer import visualizer

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
from data import make_dataset_for_kuka_calibration_withglove_touch_only

from utils import to_np, set_seed, get_lr
from models import CNNCali


args = gen_args()
print(args)

use_gpu = torch.cuda.is_available()
set_seed(42)


'''
visualizer
'''
if args.knit_name in ['kuka_calibration']:
    vis = visualizer('kuka')


'''
model
'''
model = CNNCali(args, vis.mask)

if args.eval == 0:
    if args.resume == 1:
        model_name = 'net_kuka_epoch_%d_iter_%d.pth' % (args.epoch, args.iter)
        model_path = os.path.join(args.ckp_path, model_name)
        print("Loading network from %s" % model_path)
        model.load_state_dict(torch.load(model_path))
    else:
        print("Randomly initialize the network")

if args.eval == 1:
    if args.epoch < 0:
        model_name = 'net_kuka_best.pth'
    else:
        model_name = 'net_kuka_epoch_%d_iter_%d.pth' % (args.epoch, args.iter)

    model_path = os.path.join(args.ckp_path, model_name)
    print("Loading network from %s" % model_path)
    model.load_state_dict(torch.load(model_path))

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.imshow(model.position_bias.detach().numpy()[0, 0])
    plt.colorbar()
    plt.savefig(os.path.join(args.vis_path, 'position_bias_kuka.png'))
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
    touch_raw_rec, touch_rec, touch_ts_rec = make_dataset_for_kuka_calibration_withglove_touch_only(
        args, vis.mask, touch_path=eval_hdf5_path, debug=args.debug)
    touch_raw_rec, touch_rec = torch.FloatTensor(touch_raw_rec), torch.FloatTensor(touch_rec)

    # print('touch_raw_rec size', touch_raw_rec.size())
    # print('touch_rec size', touch_rec.size())

    if use_gpu:
        touch_raw_rec = touch_raw_rec.cuda()
        touch_rec = touch_rec.cuda()


    '''
    calibrate
    '''
    min_value, max_value = 1e10, -1e10
    batch_size = 64

    result = []
    for i in range(0, touch_rec.shape[0], batch_size):
        touch = touch_rec[i:i+batch_size]
        with torch.set_grad_enabled(False):
            x_cal, x_recon = model(touch)

        result.append(x_recon.data.cpu().numpy())

    result = np.concatenate(result, 0)
    # print('result size', result.shape)

    touch_raw_rec = touch_raw_rec.data.cpu().numpy()
    touch_rec = touch_rec.data.cpu().numpy()

    # print('pressure_raw', touch_raw_rec.shape)
    # print('pressure', result.shape)
    # print('ts', touch_ts_rec.shape)

    if args.store == 1:
        store_path = os.path.join(args.vis_path, eval_path)
        os.system('mkdir -p %s' % store_path[:store_path.rfind('/')])

        hf = h5py.File(store_path, 'w')
        hf.create_dataset('pressure_raw', data=touch_raw_rec)
        hf.create_dataset('pressure', data=result[:, 0])
        hf.create_dataset('frame_count', data=np.array([result.shape[0]]))
        hf.create_dataset('ts', data=touch_ts_rec)
        hf.close()


    '''
    visualize
    '''
    if args.vis == 1:
        fancy = True

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_path = os.path.join(args.vis_path, '%s.avi' % eval_path.replace('/', '.'))
        image_path = os.path.join(args.vis_path, '%s' % eval_path.replace('/', '.'))
        print('Saving video to', video_path)
        print('Saving image to', image_path)
        video_shape = (1280, 480) if fancy else (1200, 600)
        out = cv2.VideoWriter(video_path, fourcc, 14., video_shape)
        os.system('mkdir -p ' + image_path)

        print('min: %.4f, max: %.4f' % (np.min(result), np.max(result)))
        lim_low = np.min(result)
        lim_high = np.max(result) * 1. / 2.


        bar = ProgressBar(max_value=touch_rec.shape[0])
        for i in bar(range(touch_rec.shape[0])):
            touch_raw_cur = touch_raw_rec[i]
            touch_cur = touch_rec[i, args.obs_window // 2]
            touch_cal = result[i, 0]

            touch_raw_render = vis.render(
                touch_raw_cur, lim_low=np.min(np.median(touch_raw_rec, 0)) + 10, lim_high=800, fancy=fancy)
            '''
            touch_render = vis.render(
                touch_cur, lim_low=0., lim_high=24, text='raw', fancy=fancy)
            '''
            result_render = vis.render(
                touch_cal, lim_low=lim_low, lim_high=lim_high, fancy=fancy)

            cv2.imwrite(os.path.join(image_path, 'raw_%d.png' % i), touch_raw_render)
            cv2.imwrite(os.path.join(image_path, 'cal_%d.png' % i), result_render)

            frame = vis.merge_frames(
                [touch_raw_render, result_render], nx=1, ny=2)

            out.write(frame)

        out.release()

