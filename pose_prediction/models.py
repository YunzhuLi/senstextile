import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import io, os
from torch.utils.data import Dataset, DataLoader
import random
import torch
from torch.autograd import Variable
import math

############################ Model #############################

class sock2mocap_conv2d(nn.Module):
    def __init__(self, symmetric=True,window_size):
        super(sock2mocap_conv2d, self).__init__()
    
        if args.window == 0:
            self.conv1=nn.Conv2d(1,16,kernel_size=(3,3))
        else:
            self.conv1=nn.Conv2d(2*window_size,16,kernel_size=(3,3))
        self.relu=nn.ReLU()
        self.maxpool1=nn.MaxPool2d(kernel_size=3)
        self.conv2=nn.Conv2d(16,8,kernel_size=(3,3))
        self.maxpool2=nn.MaxPool2d(kernel_size=3)
        self.symmetric = symmetric
        if not symmetric:
            if args.window == 0:
                self.conv1_r=nn.Conv2d(1,16,kernel_size=(3,3))
            else:
                self.conv1_r=nn.Conv2d(window_size,16,kernel_size=(3,3))
            self.conv2_r = nn.Conv2d(16,8,kernel_size=(3,3))
            
        self.linear1 = nn.Linear(64,100)
        self.linear_extra = nn.Linear(100, 100)
        self.linear_extra2 = nn.Linear(100, 100)
        self.linear = nn.Linear(100, 69)
    
    def forward(self, input_l,input_r):
        out_l=self.maxpool1(self.relu(self.conv1(input_l)))
        out_l=self.maxpool2(self.relu(self.conv2(out_l)))
        if self.symmetric:
            out_r=self.maxpool1(self.relu(self.conv1(input_r)))
            out_r=self.maxpool2(self.relu(self.conv2(out_r)))
        else:
            out_r=self.maxpool1(self.relu(self.conv1_r(input_r)))
            out_r=self.maxpool2(self.relu(self.conv2_r(out_r)))

        out_l=out_l.reshape(out_l.shape[0],-1)
        out_r=out_r.reshape(out_r.shape[0],-1)
        
        i0 = torch.cat((out_l,out_r),axis=1)
        i1 = self.relu(self.linear1(i0))
        i2 = self.relu(self.linear_extra(i1))
        i3 = self.relu(self.linear_extra2(i2))
        output = self.linear(i3)
        return output
