import torch.nn as nn
from torch.autograd import Variable
from sklearn.cluster import KMeans
import torch.optim as optim
import numpy as np
import io, os
from torch.utils.data import Dataset, DataLoader
import pickle
from IPython import embed
from tensorboardX import SummaryWriter
import argparse
import random
import os.path
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
from torchvision import datasets, models, transforms
import math
import shutil
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from models import sock2mocap_conv2d

############################# Arguments #########################
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, required = True, help='Name of experiment')
parser.add_argument('--lr', type=float, default=1e-3, help='Name of experiment')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--weightdecay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--window', type=int, default=30, help='window around the time step')
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to')
parser.add_argument('--home_folder', type= str, default='/location/of/dir/', help='Where are you training your model')
parser.add_argument('--test', type=bool, default=False, help='Set true if testing time')
parser.add_argument('--checkpt', type=str, default='model_in_paper/checkpoint.pth.tar', help='checkpointnumber')
parser.add_argument('--type', type= str, default='FINAL', help='Name of dataset')
parser.add_argument('--pred', type=str, default='preds.p', help='save predictions as')
parser.add_argument('--upper_body', action='store_true') 
parser.add_argument('--testor', action='store_true') 
parser.add_argument('--all', action='store_true') 
parser.add_argument('--loss', type=str, default='mse', help='loss type')
parser.add_argument('--subsample', type=int, default=1, help='Subsample size')


args = parser.parse_args()


############################Summary writer + Checkpointer ###################
if not os.path.exists('./ckpts'):
    os.makedirs('./ckpts')

if not os.path.exists(os.path.join('./ckpts', args.exp)):
    os.makedirs(os.path.join('./ckpts', args.exp))

if not os.path.exists('./tbs'):
    os.makedirs('./tbs')
    
train_writer = SummaryWriter(os.path.join('./tbs', args.exp , 'train'))
val_writer = SummaryWriter(os.path.join('./tbs', args.exp, 'val'))

def save_checkpoint(state,epoch, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    shutil.copyfile(filename, args.home_folder+'ckpts/'+args.exp+ '/'+ str(epoch)+'.pth.tar')

#############################Data Loading########################
'''            
data_in is of the type [left_foot,right_foot, xsens]
shape of each sock =[frames,32,32] This is the reading from the sensor of the sock
shape of data from xsens = [frames,22,3] This is the joint angle data
The data is sampled at the same rate
'''
train_path = 'dataset/train.p' 
val_path = 'dataset/val.p'
test_path = 'dataset/test.p'

train_files = pickle.load( open( train_path, "rb" ) )
val_files = pickle.load( open( val_path, "rb" ) )
test_files = pickle.load( open( test_path, "rb" ) )
    

if not(args.all):
    test_files[2] = test_files[2][:,3:]
    train_files[2] = train_files[2][:,3:]
    val_files[2] = val_files[2][:,3:]

train_files[2][np.isnan(train_files[2])] = 0

train_dataset = sample_data(train_files)
val_dataset = sample_data(val_files)
test_dataset = sample_data(test_files)

print(len(train_dataset), len(val_dataset), len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=not(args.test), num_workers=4)

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=4)

test_dataloader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=4)


############################ Training Code #####################

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    seq = sock2mocap_conv2d(symmetric=False,args.window)
    seq.cuda()
    seq = nn.DataParallel(seq)
    optimizer = optim.Adam(seq.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    
    chkptno = 0
    if args.loss == 'mse':
        print ("MSE loss")
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        print ("L1 loss")
        criterion = nn.L1Loss()
    clusters = None
    
    if args.test:
        predictions = []
        inputs_ = []
        checkpoint = torch.load(args.home_folder+'ckpts/'+args.checkpt)
        seq.load_state_dict(checkpoint['state_dict'])
        seq.eval()
        for i_batch, sample_batched in enumerate(test_dataloader):
            print(i_batch)
            left = sample_batched[0].type(torch.cuda.FloatTensor) 
            right = sample_batched[1].type(torch.cuda.FloatTensor)
            xsens = sample_batched[2].type(torch.cuda.FloatTensor)
            out = seq(left,right)      
            predictions.append(out.cpu().data.numpy().reshape(xsens.shape) )
            inputs_.append(xsens.cpu().data.numpy().reshape(xsens.shape) )
        pickle.dump(predictions  , open( "results/preds.p", "wb" ) )
        pickle.dump(inputs_  , open( "results/inputs.p", "wb" ) )
    else:
        for epoch in range(args.epoch):
            print('STEP: ', epoch)
            seq.train()
            avg_train_loss = []
            for i_batch, sample_batched in enumerate(train_dataloader):
                optimizer.zero_grad()
                left = sample_batched[0].type(torch.cuda.FloatTensor) 
                right = sample_batched[1].type(torch.cuda.FloatTensor)
                xsens = sample_batched[2].type(torch.cuda.FloatTensor)
                out = seq(left,right)        
                loss = criterion(out,xsens.reshape(xsens.shape[0],-1))
                loss.backward()
                optimizer.step()
                avg_train_loss.append(loss.data.item())
                optimizer.zero_grad()
                print("Loss: ", loss)
                
                if i_batch % 100 ==0 and i_batch!=0:
                    print(chkptno)
                    print("Now running on val set")
                    train_writer.add_scalar('data/loss', np.mean(avg_train_loss) , chkptno)

                    save_checkpoint({
                        'epoch': chkptno,
                        'state_dict': seq.state_dict(),
                    },chkptno)


                    seq.eval()
                    avg_val_loss = []
                    for i_batch, sample_batched in enumerate(val_dataloader):
                        left = sample_batched[0].type(torch.cuda.FloatTensor)
                        right = sample_batched[1].type(torch.cuda.FloatTensor)
                        out = seq(left,right)
                        xsens = sample_batched[2].type(torch.cuda.FloatTensor)
                        loss = criterion(out,xsens.reshape(xsens.shape[0],-1))
                        avg_val_loss.append(loss.data.item())
                    val_loss = np.mean(avg_val_loss)
                    print('val loss:', val_loss)
                    val_writer.add_scalar('data/loss', val_loss, chkptno)
                    chkptno = chkptno+1
            avg_train_loss = np.mean(avg_train_loss)        
            print('train loss:', avg_train_loss)
