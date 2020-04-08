import torch.nn as nn
from torch.autograd import Variable
from sklearn.cluster import KMeans
import torch.optim as optim
import numpy as np
import io, os
from torch.utils.data import Dataset, DataLoader
import pickle
import torch
from torch.autograd import Variable
import h5py
from torchvision import datasets, models, transforms
import math
import shutil

def window_select(data,timestep,window):
    if window ==0:
        return data[timestep : timestep + 1, :, :]
    max_len = data.shape[0]
    l = max(0,timestep-window)
    u = min(max_len,timestep+window)
    if l == 0:
        return (data[:2*window,:,:])
    elif u == max_len:
        return (data[-2*window:,:,:])
    else:
        return(data[l:u,:,:])

class sample_data(Dataset):
    def __init__(self, data_in,subsample,window): 
        self.data_in = data_in
        self.subsample = subsample
        self.window = window

    def __len__(self):
        return self.data_in[0].shape[0]

    def __getitem__(self, idx):
        left = window_select(self.data_in[0],idx,self.window)
        right = window_select(self.data_in[1],idx,self.window)
        if self.subsample > 1:
            subsample = self.subsample
            for x in range(0, left.shape[1], subsample):
                for y in range(0, left.shape[2], subsample):
                    v = np.mean(left[:, x:x+subsample, y:y+subsample], (1, 2))
                    left[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)
            for x in range(0, right.shape[1], subsample):
                for y in range(0, right.shape[2], subsample):
                    v = np.mean(right[:, x:x+subsample, y:y+subsample], (1, 2))
                    right[:, x:x+subsample, y:y+subsample] = v.reshape(-1, 1, 1)
        mocap = self.data_in[2][idx,:]
        return (left,right,mocap)
