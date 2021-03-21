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
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from serialization import load_model
import h5py
import cv2
import transformations

## Load SMPL model (here we load the female model)
m = load_model('models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
import pickle
import numpy as np
import os

expmtname = 'expname'
dataset_type = 'Final'
os.mkdir('VISUALIZATION')
os.mkdir('VISUALIZATION/'+expmtname)
os.mkdir('VISUALIZATION/'+expmtname+'/INPUT/')
os.mkdir('VISUALIZATION/'+expmtname+'/OUTPUT/')
os.mkdir('VISUALIZATION/'+expmtname+'/TOGETHER/')


gt = pickle.load( open( 'dataset/test.p', "rb" ) )

    
pred = pickle.load(open('results/preds.p',"rb"))
input_ = pickle.load(open('results/inputs.p',"rb"))

predictions = np.array(pred)[:,0,:]
input_passed = np.array(input_)[:,0,:]


for i in range(0,gt[2].shape[0]):
  print(i)
  m.pose[:] = gt[2][i,:]

  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##   
  m.pose[3:] = input_passed[i,:]
  # Create OpenDR renderer
  rn = ColoredRenderer()
  ## Assign attributes to renderer
  w, h = (640, 480)

  rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
  rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
  rn.set(v=m, f=m.f, bgcolor=np.ones(3))

  ## Construct point light source
  rn.vc = LambertianPointLight(
      f=m.f,
      v=rn.v,
      num_verts=len(m),
      light_pos=np.array([-1000,-1000,-6000]),
      vc=np.ones_like(m)*.9,
      light_color=np.array([1., 1., 1.]))

  ## Show it using Open12
  INPUT = (rn.r).copy()
  cv2.imwrite('VISUALIZATION/'+expmtname+'/INPUT/'+str(i)+'.jpg', rn.r*255)
  ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
  m.pose[3:] = predictions[i,:]
  # Create OpenDR renderer
  rn = ColoredRenderer()
  ## Assign attributes to renderer
  w, h = (640, 480)

  rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
  rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
  rn.set(v=m, f=m.f, bgcolor=np.ones(3))

  ## Construct point light source
  rn.vc = LambertianPointLight(
      f=m.f,
      v=rn.v,
      num_verts=len(m),
      light_pos=np.array([-1000,-1000,-6000]),
      vc=np.ones_like(m)*.9,
      light_color=np.array([1., 1., 1.]))

  ## Show it using Open12
  OUTPUT =  (rn.r).copy()
  cv2.imwrite('VISUALIZATION/'+expmtname+'/OUTPUT/'+str(i)+'.jpg', rn.r*255)

  #############################################################
  TOGETHER = np.concatenate((INPUT,OUTPUT),axis = 1)
  cv2.imwrite('VISUALIZATION/'+expmtname+'/TOGETHER/'+str(i)+'.jpg', TOGETHER*255)
