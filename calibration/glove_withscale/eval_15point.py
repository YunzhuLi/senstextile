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
from data import make_dataset_for_glove_calibration_withscale

from knit_calib.utils.utils import to_np, set_seed, get_lr
from knit_calib.models.models import CNNCali


args = gen_args()
print(args)

use_gpu = torch.cuda.is_available()
set_seed(42)


'''
visualizer
'''
if args.knit_name in ['glove_calibration_randompressing_withvision']:
    vis = visualizer(sensor_type='glove', side='right')


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
n_point = 15

cor_raw = []
cor_han = []
cor_cal = []

for idx_point in range(1, n_point + 1):
    data_path_prefix = os.path.join(args.data_path, 'glove_val_15point/%d' % idx_point)

    '''
    data
    '''
    # touch_raw_rec: B x 32 x 32
    # touch_rec: B x obs_window x 32 x 32
    # scale_rec: B
    touch_raw_rec, touch_rec, scale_rec = make_dataset_for_glove_calibration_withscale(
        args, 'full', data_path_prefix, vis.mask, debug=args.debug)

    print('touch_raw_rec', touch_raw_rec.shape, '%.4f %.4f %.4f %.4f' % (
        touch_raw_rec.mean(), touch_raw_rec.std(),
        touch_raw_rec.min(), touch_raw_rec.max()))
    print('touch_rec', touch_rec.shape, '%.4f %.4f %.4f %.4f' % (
        touch_rec.mean(), touch_rec.std(),
        touch_rec.min(), touch_rec.max()))
    print('scale_rec', scale_rec.shape, '%.4f %.4f %.4f %.4f' % (
        scale_rec.mean(), scale_rec.std(),
        scale_rec.min(), scale_rec.max()))

    touch_raw_rec = torch.FloatTensor(touch_raw_rec)
    touch_rec = torch.FloatTensor(touch_rec)
    scale_rec = torch.FloatTensor(scale_rec)

    if use_gpu:
        touch_raw_rec = touch_raw_rec.cuda()
        touch_rec = touch_rec.cuda()
        scale_rec = scale_rec.cuda()

    # print('touch_raw_rec', touch_raw_rec.size())
    # print('touch_rec', touch_raw_rec.size())
    # print('scale_rec', touch_raw_rec.size())


    '''
    calibrate
    '''
    batch_size = 64

    result = []
    for i in range(0, touch_rec.shape[0], batch_size):
        touch = touch_rec[i:i+batch_size]
        with torch.set_grad_enabled(False):
            x_cal, x_recon = model(touch)

        result.append(x_recon.data.cpu().numpy())

    result = np.concatenate(result, 0)[:, 0]
    # print('result size', result.shape)

    touch_raw_rec = touch_raw_rec.data.cpu().numpy()
    touch_rec = touch_rec.data.cpu().numpy()[:, args.obs_window//2]
    scale_rec = scale_rec.data.cpu().numpy()

    '''
    compare with scale
    '''
    cor_raw_cur = np.corrcoef(scale_rec, np.sum(touch_raw_rec, (1, 2)))[0, 1]
    cor_han_cur = np.corrcoef(scale_rec, np.sum(touch_rec, (1, 2)))[0, 1]
    cor_cal_cur = np.corrcoef(scale_rec, np.sum(result, (1, 2)))[0, 1]
    print('%d raw corrcoef' % idx_point, cor_raw_cur)
    print('%d han corrcoef' % idx_point, cor_han_cur)
    print('%d cal corrcoef' % idx_point, cor_cal_cur)

    cor_raw.append(cor_raw_cur)
    cor_han.append(cor_han_cur)
    cor_cal.append(cor_cal_cur)

    '''
    plt.rcParams["figure.figsize"] = (6, 6)
    plt.figure()
    plt.plot(scale_rec, np.sum(touch_raw_rec, (1, 2)), 'bo', markersize=2)
    plt.savefig(os.path.join(args.vis_path, 'touch_%d_correlation_raw.png' % idx_point))
    plt.close()

    plt.rcParams["figure.figsize"] = (6, 6)
    plt.figure()
    plt.plot(scale_rec, np.sum(result, (1, 2)), 'bo', markersize=2)
    plt.savefig(os.path.join(args.vis_path, 'touch_%d_correlation_cal.png' % idx_point))
    plt.close()
    '''


    '''
    store
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


print('raw', np.mean(cor_raw))
print('han', np.mean(cor_han))
print('cal', np.mean(cor_cal))




### bar plot before/after calibration
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        height = np.round(height, 2)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=20)



### plot the correlation for each pointyy

plt.rcParams["figure.figsize"] = (12, 6)
# plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(dpi=200)


for i in range(len(cor_raw)):
    x = np.arange(3) + i * 4
    width = 1.0
    rects1 = ax.bar(x[0], cor_raw[i], width, color='royalblue', label='Raw' if i == 0 else None)
    rects2 = ax.bar(x[1], cor_han[i], width, color='limegreen', label='Manual' if i == 0 else None)
    rects3 = ax.bar(x[2], cor_cal[i], width, color='orangered', label='Self-supervised' if i == 0 else None)

    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)

plt.xlim(-1, 59)
plt.ylim(0.3, 1.01)
plt.xticks(np.arange(15) * 4 + 1, np.arange(15) + 1)
ax.tick_params(labelsize=15)
plt.legend(loc='upper right', fontsize=20)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(args.vis_path, 'touch_correlation_bar_15points.png'))
plt.savefig(os.path.join(args.vis_path, 'touch_correlation_bar_15points.pdf'))
plt.close()



### plot the overall correlation

plt.rcParams["figure.figsize"] = (3, 6)
# plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(dpi=200)


x = np.arange(3)
width = 0.8
rects1 = ax.bar(x[0], np.mean(cor_raw), width, color='royalblue', label='Raw')
rects2 = ax.bar(x[1], np.mean(cor_han), width, color='limegreen', label='Manual')
rects3 = ax.bar(x[2], np.mean(cor_cal), width, color='orangered', label='Self-supervised')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.xlim(-0.5, 2.5)
plt.ylim(0.3, 1.01)
plt.xticks([], [])
ax.tick_params(labelsize=15)
# plt.legend(loc='upper right', fontsize=20)

plt.tight_layout(pad=0.5)
plt.savefig(os.path.join(args.vis_path, 'touch_correlation_bar_15points_overall.png'))
plt.savefig(os.path.join(args.vis_path, 'touch_correlation_bar_15points_overall.pdf'))
plt.close()

