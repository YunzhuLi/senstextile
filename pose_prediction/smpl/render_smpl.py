'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


Please Note:
============
This is a demo version of the script for driving the SMPL model with python.
We would be happy to receive comments, help and suggestions on improving this code 
and in making it available on more platforms. 


System Requirements:
====================
Operating system: OSX, Linux

Python Dependencies:
- Numpy & Scipy  [http://www.scipy.org/scipylib/download.html]
- Chumpy [https://github.com/mattloper/chumpy]
- OpenCV [http://opencv.org/downloads.html] 
  --> (alternatively: matplotlib [http://matplotlib.org/downloads.html])


About the Script:
=================
This script demonstrates loading the smpl model and rendering it using OpenDR 
to render and OpenCV to display (or alternatively matplotlib can also be used
for display, as shown in commented code below). 

This code shows how to:
  - Load the SMPL model
  - Edit pose & shape parameters of the model to create a new body in a new pose
  - Create an OpenDR scene (with a basic renderer, camera & light)
  - Render the scene using OpenCV / matplotlib


Running the Hello World code:
=============================
Inside Terminal, navigate to the smpl/webuser/hello_world directory. You can run 
the hello world script now by typing the following:
> python render_smpl.py


'''

import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from serialization import load_model
from IPython import embed
import h5py
import cv2
import transformations

## Load SMPL model (here we load the female model)
m = load_model('models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
d = '/data/vision/torralba/communication/projects/glove/mocap/data/dataset4/train/upper_1'
xsens = h5py.File(d+'/xsens.hdf5', 'r')
joint_angles = xsens['joint_angle']

joint_angles_arr = np.array(joint_angles)/float(180)*np.pi
euler_angles = np.array(xsens['rotation_euler'])
euler_angles_arr_temp = euler_angles/float(180)*np.pi


m.pose[:] = np.zeros(m.pose.size)#np.random.rand(m.pose.size) * .2
m.betas[:] = np.zeros(m.betas.size)#np.random.rand(m.betas.size) * .03
m.pose[0] = np.pi

connectivity = [(0,3),(2,6),(3,9),(4,12),(5,15),(6,14),(7,17),(8,19),(9,21),(10,13),(11,16),(12,18),(13,20),(14,2),(15,5),(16,8),(17,11),(18,1),(19,4),(20,7),(21,10)]
# connectivity = [(0,3),(2,6),(3,9),(4,12),(5,15),(6,13),(7,16),(8,18),(9,20),(10,14),(11,17),(12,19),(13,21),(14,1),(15,4),(16,7),(17,10),(18,2),(19,5),(20,8),(21,11)]

for i in range(25000,25100):
  m.pose[:] = np.zeros(m.pose.size)#np.random.rand(m.pose.size) * .2
  print(i-25000)
#   # for i in range(len(connectivity)): 
#   #   m.pose[connectivity[i][1]*3]= joint_angles_arr[j,connectivity[i][0],1]
#   #   m.pose[(connectivity[i][1]*3+1)]= joint_angles_arr[j,connectivity[i][0],2]
#   #   m.pose[(connectivity[i][1]*3+2)]= joint_angles_arr[j,connectivity[i][0],0]
#   #   m.pose[45:48] = [0,0,0]
#   #   m.pose[0] = np.pi
  euler_angles_arr = np.zeros((euler_angles_arr_temp.shape))
  temp = np.zeros((72))
  euler_angles_arr[i,:,0] = euler_angles_arr_temp[i,:,2] 
  euler_angles_arr[i,:,1] = euler_angles_arr_temp[i,:,0] 
  euler_angles_arr[i,:,2] = euler_angles_arr_temp[i,:,1]

  # Right instead of left
  mat_con = [(1,19,0),(2,15,0),(3,2,0),(4,20,19),(5,16,15),(6,3,2),(7,21,20),(8,17,16),(9,4,3),(10,22,21),(11,18,17),(12,5,4),(13,11,4),(14,7,4),(16,12,11),(17,8,7),(18,13,12),(19,9,8),(20,14,13),(21,10,9)]

  # for j in range(len(mat_con)):
  #     temp_id = (mat_con[j][0])*3
  #     u_id = mat_con[j][1]
  #     l_id = mat_con[j][2]
  #     temp[temp_id:(temp_id+3)] = transformations.euler_from_matrix(np.matmul(transformations.euler_matrix(euler_angles_arr[i,u_id,:][0],euler_angles_arr[i,u_id,:][1],euler_angles_arr[i,u_id,:][2]) , np.linalg.inv(transformations.euler_matrix(euler_angles_arr[i,l_id,:][0],euler_angles_arr[i,l_id,:][1],euler_angles_arr[i,l_id,:][2]))))


  #Rightleg
  m.pose[1*3]= joint_angles_arr[i,14,1]
  m.pose[(1*3+1)]= joint_angles_arr[i,14,2]
  m.pose[(1*3+2)]= joint_angles_arr[i,14,0]
  m.pose[4*3]= joint_angles_arr[i,15,1]
  m.pose[(4*3+1)]= joint_angles_arr[i,15,2]
  m.pose[(4*3+2)]= joint_angles_arr[i,15,0]
  m.pose[7*3]= joint_angles_arr[i,16,1]
  m.pose[(7*3+1)]= joint_angles_arr[i,16,2]
  m.pose[(7*3+2)]= joint_angles_arr[i,16,0]
  m.pose[10*3]= joint_angles_arr[i,17,1]
  m.pose[(10*3+1)]= joint_angles_arr[i,17,2]
  m.pose[(10*3+2)]= joint_angles_arr[i,17,0]

  #Left leg
  m.pose[2*3]= joint_angles_arr[i,18,1]
  m.pose[(2*3+1)]= joint_angles_arr[i,18,2]
  m.pose[(2*3+2)]= joint_angles_arr[i,18,0]
  m.pose[5*3]= joint_angles_arr[i,19,1]
  m.pose[(5*3+1)]= joint_angles_arr[i,19,2]
  m.pose[(5*3+2)]= joint_angles_arr[i,19,0]
  m.pose[8*3]= joint_angles_arr[i,20,1]
  m.pose[(8*3+1)]= joint_angles_arr[i,20,2]
  m.pose[(8*3+2)]= joint_angles_arr[i,20,0]
  m.pose[11*3]= joint_angles_arr[i,21,1]
  m.pose[(11*3+1)]= joint_angles_arr[i,21,2]
  m.pose[(11*3+2)]= joint_angles_arr[i,21,0]

  #Rightarm
  m.pose[20*3]= joint_angles_arr[i,9,1]
  m.pose[(20*3+1)]= joint_angles_arr[i,9,2]
  m.pose[(20*3+2)]= joint_angles_arr[i,9,0]
  m.pose[18*3]= joint_angles_arr[i,8,1]
  m.pose[(18*3+1)]= joint_angles_arr[i,8,2]-np.pi/2
  m.pose[(18*3+2)]= joint_angles_arr[i,8,0]
  m.pose[16*3]= joint_angles_arr[i,7,1]
  m.pose[(16*3+1)]= joint_angles_arr[i,7,2]
  m.pose[(16*3+2)]= joint_angles_arr[i,7,0]-np.pi/2
  m.pose[13*3]= joint_angles_arr[i,6,1]
  m.pose[(13*3+1)]= joint_angles_arr[i,6,2]
  m.pose[(13*3+2)]= joint_angles_arr[i,6,0]

  #Leftarm
  m.pose[21*3]= joint_angles_arr[i,13,1]
  m.pose[(21*3+1)]= joint_angles_arr[i,13,2]
  m.pose[(21*3+2)]= joint_angles_arr[i,13,0]
  m.pose[19*3]= joint_angles_arr[i,12,1]
  m.pose[(19*3+1)]= joint_angles_arr[i,12,2]
  m.pose[(19*3+2)]= joint_angles_arr[i,12,0]#+3*np.pi/2
  m.pose[17*3]= joint_angles_arr[i,11,1] 
  m.pose[(17*3+1)]= joint_angles_arr[i,11,2]
  m.pose[(17*3+2)]= joint_angles_arr[i,11,0]#+np.pi/2
  m.pose[14*3]= joint_angles_arr[i,10,1]
  m.pose[(14*3+1)]= joint_angles_arr[i,10,2]
  m.pose[(14*3+2)]= joint_angles_arr[i,10,0]

  # m.pose[:] = temp[:]
  m.pose[0] = np.pi

# ## Create OpenDR renderer
  rn = ColoredRenderer()

  ## Assign attributes to renderer
  w, h = (640, 480)

  rn.camera = ProjectPoints(v=m, rt=np.zeros(3), t=np.array([0, 0, 2.]), f=np.array([w,w])/2., c=np.array([w,h])/2., k=np.zeros(5))
  rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
  rn.set(v=m, f=m.f, bgcolor=np.zeros(3))

  ## Construct point light source
  rn.vc = LambertianPointLight(
      f=m.f,
      v=rn.v,
      num_verts=len(m),
      light_pos=np.array([-1000,-1000,-6000]),
      vc=np.ones_like(m)*.9,
      light_color=np.array([1., 1., 1.]))


  ## Show it using OpenCV
  cv2.imwrite('one_by_one/'+str(i-25000)+'.jpg', rn.r*255)
  # print ('..Print any key while on the display window')
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()


  ## Could also use matplotlib to display
  # import matplotlib.pyplot as plt
  # plt.ion()
  # plt.imshow(rn.r)
  # plt.show()
  # import pdb; pdb.set_trace()