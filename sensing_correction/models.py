import os
import random
import cv2
import h5py
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader



class CNNCali(nn.Module):

    def __init__(self, args, mask, scale_factor=None, position_bias=None):
        super(CNNCali, self).__init__()

        self.args = args
        self.mask = torch.FloatTensor(mask).cuda()[None, None, :, :]

        if scale_factor is None:
            self.scale_factor = args.scale_factor
        else:
            self.scale_factor = scale_factor

        if position_bias is None:
            self.pbias = args.position_bias
        else:
            self.pbias = position_bias

        nf_hidden = args.nf_hidden
        nf_in = args.obs_window
        nf_out = 1

        self.position_bias = torch.nn.Parameter(
            data=torch.zeros(1, 1, args.height, args.width), requires_grad=True)
        self.global_bias = torch.nn.Parameter(
            data=torch.zeros(1), requires_grad=True)
        self.global_scale = torch.nn.Parameter(
            data=torch.ones(1), requires_grad=True)

        # identity
        self.conv_0 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf_in, nf_hidden, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU())

        # identity
        self.conv_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf_hidden, nf_hidden, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU())

        # identity
        self.conv_cal = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf_hidden, nf_out, 3, 1),
            nn.ReLU())

        if args.superres > 1:
            self.conv_superres_0 = nn.Sequential(
                nn.Upsample(scale_factor=args.superres, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0),
                nn.ReLU())

    def forward(self, x):
        scale_factor = self.scale_factor

        # x0: 32 x 32
        if self.pbias == 1:
            x0 = self.conv_0(x + self.position_bias)
        else:
            x0 = self.conv_0(x)

        # x0_scale
        x0_scale = F.interpolate(x0, scale_factor=scale_factor, mode='bilinear')

        # x1_scale
        x1_scale = self.conv_1(x0_scale) + x0_scale

        # x1: 32 x 32
        x1 = F.interpolate(x1_scale, scale_factor=1./scale_factor, mode='bilinear')

        # x_cal: 32 x 32
        x_cal = self.conv_cal(x1) + x0

        if self.args.superres > 1:
            x_cal = self.conv_superres_0(x_cal)

        # rescale for calculating reconstruction loss
        x_recon = (x_cal + self.global_bias) * self.global_scale

        # apply sensor mask
        x_cal = x_cal * self.mask
        x_recon = x_recon * self.mask
        return x_cal, x_recon

