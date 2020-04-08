import h5py
import numpy as np
import io, os
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle
from fractions import gcd
import math
import random
import transformations
import rotation_matrix

# Function to align the frames using the timestep count
def findNearestFrame(srcTimes, targetTimes):
    orderSrc = np.argsort(srcTimes)
    orderTarget = np.argsort(targetTimes)
    srcSorted = srcTimes[orderSrc]
    targetSorted = targetTimes[orderTarget]
    # find nearest frame for each resampled frame
    nearestFrame = np.zeros([len(srcTimes)], int)
    mi = 0
    for i in range(len(srcSorted)):
        while True:
            if mi == len(targetSorted) - 1:
                break
            if targetSorted[mi + 1] - srcSorted[i] > srcSorted[i] - targetSorted[mi]:
                break
            mi += 1
        nearestFrame[i] = mi

    nearestFrame = orderTarget[nearestFrame]
    nearestFrame = nearestFrame[np.argsort(orderSrc)]
    return nearestFrame
    
# Use this function to time align and recover joint angle representations for the dataset
file_typ = ['test','val','train']
for typ in file_typ:
    dataloc = 'data/'+typ+'/'
    data = [os.path.join(dataloc, o) for o in os.listdir(dataloc)]

    left_total = np.zeros((1,32,32))
    right_total = np.zeros((1,32,32))
    xsens_total = np.zeros((1,23,4))

    # Load the data into variables and subsample it
    for loc in data:
        print(loc)
        left_foot = h5py.File(str(loc)+'/touch_left.hdf5', 'r')
        right_foot = h5py.File(str(loc)+'/touch_right.hdf5', 'r')
        xsens = h5py.File(str(loc)+'/xsens.hdf5', 'r')

        fc_l = left_foot['frame_count'][0]
        fc_r = right_foot['frame_count'][0]
        fc_xsens = xsens['frame_count'][0]

        ts_l = left_foot['ts'][:fc_l].astype(np.float128)[:]
        ts_r = right_foot['ts'][:fc_r].astype(np.float128)[:]
        ts_xsens = xsens['ts'][:fc_xsens].astype(np.float128)[:]

        fc_min = fc_r

        # Crop the data using the minimum frame count
        left_data = left_foot['pressure'][:fc_l].astype(np.float32)[:,:,:]
        right_data = right_foot['pressure'][:fc_r].astype(np.float32)[:,:,:]
        xsens_data = xsens['rotation_quat'][:fc_xsens].astype(np.float32)[:,:,:]

        xsens_data_subsampled = np.zeros((fc_min,23,4))
        left_subsampled = np.zeros((fc_min,32,32))
        right_subsampled = np.zeros((fc_min,32,32))

        id_xsens = findNearestFrame(ts_r,ts_xsens)
        id_left = findNearestFrame(ts_r,ts_l)
        id_right = findNearestFrame(ts_r,ts_r)

        for i in range(fc_min):
            idxx = id_xsens[i]
            idxx_l = id_left[i]
            idxx_r = id_right[i]
            xsens_data_subsampled[i,:,:] = xsens_data[idxx,:,:]        
            left_subsampled[i,:,:] = left_data[idxx_l,:,:]
            right_subsampled[i,:,:] = right_data[idxx_r,:,:]        

        left_total = np.concatenate((left_total,left_subsampled),axis = 0)
        right_total = np.concatenate((right_total,right_subsampled),axis = 0)
        xsens_total = np.concatenate((xsens_total,xsens_data_subsampled),axis = 0)

    # Convert the quaternion joint angle representation to the required MOCAP output format
    quat_joints = np.zeros((1,72))
    for i in range(xsens_total.shape[0]):
        temp = np.zeros((72))
        mat_con = [(1,19,0),(2,15,0),(3,2,0),(4,20,19),(5,16,15),
                 (6,3,2),(7,21,20),(8,17,16),(9,4,3),(10,22,21),
                 (11,18,17),(13,11,4),(14,7,4),(16,12,11),(17,8,7),
                (18,13,12),(19,9,8),(20,14,13),(21,10,9)]

        R = transformations.quaternion_matrix(xsens_total[i, 0,:])
        axis, theta = rotation_matrix.R_to_axis_angle(R)
        ord = np.array([1, 2, 0])
        temp[:3] = (axis * theta)[ord]  

        for j in range(len(mat_con)):
            temp_id = (mat_con[j][0])*3
            u_id = mat_con[j][1]s
            l_id = mat_con[j][2]
            R_A = transformations.quaternion_matrix(xsens_total[i, l_id, :])
            R_B = transformations.quaternion_matrix(xsens_total[i, u_id, :])    
            axis, theta = rotation_matrix.R_to_axis_angle(np.matmul(np.linalg.inv(R_A), R_B))
            ord = np.array([1, 2, 0])
            temp[temp_id:temp_id+3] = (axis * theta)[ord]
        quat_joints = np.concatenate((quat_joints,temp[None,:]),axis = 0)
        
    quat_joints[np.isnan(quat_joints)] = 0 
    to_save = [left_total[1:,:,:],right_total[1:,:,:],quat_joints[2:,:]]
    pickle.dump(to_save, open( "/data/vision/torralba/communication/projects/glove/mocap/dataset/"+typ+".p", "wb" ) )    