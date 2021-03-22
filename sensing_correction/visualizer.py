import os
import h5py
import time
import numpy as np

import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from math import acos, atan2, cos, pi, sin
from numpy import array, cross, dot, float64, hypot, zeros
from numpy.linalg import norm
from random import gauss, uniform



def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def get_size_and_color_for_plt_scatter(size, lim_low, lim_high, sensor_type='glove', vis_type='color'):

    # vis_type: size, color
    if vis_type == 'size':
        size = np.sqrt(size)
        size = np.clip(size, lim_low, lim_high)
        s = size
        c = 'k'

    elif vis_type == 'color':
        s = 75 if sensor_type == 'sock' else 25
        size = np.clip(size, lim_low, lim_high)
        c = (size - lim_low) / (lim_high - lim_low)

    return s, c



class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)[:3]


def R_2vect(R, vector_orig, vector_fin):
    """Calculate the rotation matrix required to rotate from one vector to another.
    For the rotation of one vector to another, there are an infinit series of rotation matrices
    possible.  Due to axially symmetry, the rotation axis can be any vector lying in the symmetry
    plane between the two vectors.  Hence the axis-angle convention will be used to construct the
    matrix with the rotation axis defined as the cross product of the two vectors.  The rotation
    angle is the arccosine of the dot product of the two unit vectors.
    Given a unit vector parallel to the rotation axis, w = [x, y, z] and the rotation angle a,
    the rotation matrix R is::
              |  1 + (1-cos(a))*(x*x-1)   -z*sin(a)+(1-cos(a))*x*y   y*sin(a)+(1-cos(a))*x*z |
        R  =  |  z*sin(a)+(1-cos(a))*x*y   1 + (1-cos(a))*(y*y-1)   -x*sin(a)+(1-cos(a))*y*z |
              | -y*sin(a)+(1-cos(a))*x*z   x*sin(a)+(1-cos(a))*y*z   1 + (1-cos(a))*(z*z-1)  |
    @param R:           The 3x3 rotation matrix to update.
    @type R:            3x3 numpy array
    @param vector_orig: The unrotated vector defined in the reference frame.
    @type vector_orig:  numpy array, len 3
    @param vector_fin:  The rotated vector defined in the reference frame.
    @type vector_fin:   numpy array, len 3
    """

    # Convert the vectors to unit vectors.
    vector_orig = vector_orig / norm(vector_orig)
    vector_fin = vector_fin / norm(vector_fin)

    # The rotation axis (normalised).
    axis = cross(vector_orig, vector_fin)
    axis_len = norm(axis)
    if axis_len != 0.0:
        axis = axis / axis_len

    # Alias the axis coordinates.
    x = axis[0]
    y = axis[1]
    z = axis[2]

    # The rotation angle.
    angle = acos(dot(vector_orig, vector_fin))

    # Trig functions (only need to do this maths once!).
    ca = cos(angle)
    sa = sin(angle)

    # Calculate the rotation matrix elements.
    R[0,0] = 1.0 + (1.0 - ca)*(x**2 - 1.0)
    R[0,1] = -z*sa + (1.0 - ca)*x*y
    R[0,2] = y*sa + (1.0 - ca)*x*z
    R[1,0] = z*sa+(1.0 - ca)*x*y
    R[1,1] = 1.0 + (1.0 - ca)*(y**2 - 1.0)
    R[1,2] = -x*sa+(1.0 - ca)*y*z
    R[2,0] = -y*sa+(1.0 - ca)*x*z
    R[2,1] = x*sa+(1.0 - ca)*y*z
    R[2,2] = 1.0 + (1.0 - ca)*(z**2 - 1.0)



class visualizer(object):

    def __init__(self, sensor_type, side=None, root_dir='../data_sensing_correction/'):
        '''
        sensor_type: glove
        side: left, right

        sensor_type: kitchen_glove
        side: left, right

        sensor_type: sock
        side: left, right

        sensor_type: vest
        side: front, back

        sensor_type: kuka
        '''

        vis_path_prefix = os.path.join(root_dir, 'visualization/')

        if sensor_type in ['glove', 'vest', 'kitchen_glove']:
            vis_outline_path = os.path.join(
                vis_path_prefix, sensor_type, '%s_vis_outline.csv' % (sensor_type))
            vis_sensor_path = os.path.join(
                vis_path_prefix, sensor_type, '%s_%s_vis_sensor.csv' % (
                    sensor_type, side))
            vis_outline = open(vis_outline_path, 'r').readlines()
            vis_sensor = open(vis_sensor_path, 'r').readlines()

        elif sensor_type in ['sock']:
            vis_outline_path = os.path.join(
                vis_path_prefix, sensor_type, '%s_vis_outline.csv' % (sensor_type))
            vis_sensor_path = os.path.join(
                vis_path_prefix, sensor_type, '%s_%s_vis_sensor_low.csv' % (
                    sensor_type, side))
            vis_outline = open(vis_outline_path, 'r').readlines()
            vis_sensor = open(vis_sensor_path, 'r').readlines()

        elif sensor_type in ['kuka']:
            vis_mesh_path = os.path.join(vis_path_prefix, sensor_type, 'link_1_origin.obj')
            vis_sensor_path = os.path.join(vis_path_prefix, sensor_type, 'Sensor_info.txt')
            vis_sensor = open(vis_sensor_path, 'r').readlines()

        else:
            raise AssertionError("Unknown sensor_type %s" % sensor_type)

        if sensor_type == 'glove':
            self.init_vis_glove(vis_outline, vis_sensor, side)
        elif sensor_type == 'sock':
            self.init_vis_sock(vis_outline, vis_sensor, side)
        elif sensor_type == 'vest':
            self.init_vis_vest(vis_outline, vis_sensor, side)
        elif sensor_type == 'kitchen_glove':
            self.init_vis_kitchen_glove(vis_outline, vis_sensor, side)
        elif sensor_type == 'kuka':
            self.init_vis_kuka(vis_sensor, vis_mesh_path)

        self.sensor_type = sensor_type

    def init_vis_glove(self, vis_outline, vis_sensor, side='right'):
        '''
        glove contour
        '''
        contour_x, contour_y = [], []
        cx, cy = None, None

        for idx in range(1, len(vis_outline)):
            if vis_outline[idx][0] == '*':
                if cx is not None and cy is not None:
                    contour_x.append(cx)
                    contour_y.append(cy)
                cx, cy = [], []
            else:
                x, y = vis_outline[idx].strip().split(',')
                cx.append(float(x) if side == 'right' else -float(x))
                cy.append(float(y))

        self.contour_x = contour_x
        self.contour_y = contour_y

        '''
        glove sensor
        '''
        sensor_x, sensor_y, sensor_idx, mask = [], [], [], np.zeros((32, 32))
        for idx in range(0, len(vis_sensor), 4):
            x_gs = np.zeros(4)                  # x axis on glove visualization
            y_gs = np.zeros(4)                  # y axis on glove visualization
            x_ss = np.zeros(4, dtype=np.int)    # x axis on sensor grid
            y_ss = np.zeros(4, dtype=np.int)    # y axis on sensor grid

            idx_list = [idx, idx + 1, idx + 2, idx + 3]
            for ii, jj in enumerate(idx_list):
                x_g, y_g, x_s, y_s = vis_sensor[jj].strip().split(',')
                x_gs[ii] = float(x_g)
                y_gs[ii] = float(y_g)
                x_ss[ii] = int(x_s)
                y_ss[ii] = int(y_s)
                # print('ha', x_g, y_g, x_s, y_s)

            idx_rec = idx // 4
            if idx_rec in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18]:
                st_x, ed_x = x_ss[0], x_ss[2]
                st_y, ed_y = y_ss[0], y_ss[1] + 1
            elif idx_rec in [14, 15]:
                st_x, ed_x = x_ss[0], x_ss[2]
                st_y, ed_y = y_ss[0], y_ss[1]
            elif idx_rec in [16]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0], y_ss[1]
            elif idx_rec in [13, 19]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0], y_ss[1] + 1

            for xx in range(st_x, ed_x):
                for yy in range(st_y, ed_y):

                    xs_ratio = (yy - y_ss[0]) / float(y_ss[1] - y_ss[0])
                    ys_ratio = (xx - x_ss[0]) / float(x_ss[2] - x_ss[0])

                    p0 = [(x_gs[1] - x_gs[0]) * xs_ratio + x_gs[0], (y_gs[1] - y_gs[0]) * xs_ratio + y_gs[0]]
                    p1 = [(x_gs[2] - x_gs[1]) * ys_ratio + x_gs[1], (y_gs[2] - y_gs[1]) * ys_ratio + y_gs[1]]
                    p2 = [(x_gs[2] - x_gs[3]) * xs_ratio + x_gs[3], (y_gs[2] - y_gs[3]) * xs_ratio + y_gs[3]]
                    p3 = [(x_gs[3] - x_gs[0]) * ys_ratio + x_gs[0], (y_gs[3] - y_gs[0]) * ys_ratio + y_gs[0]]

                    R = intersection(line(p0, p2), line(p1, p3))

                    # print(R[0], R[1], xx, yy)

                    sensor_x.append(R[0] if side == 'right' else -R[0])
                    sensor_y.append(R[1])
                    sensor_idx.append(xx * 32 + yy)
                    mask[xx, yy] = 1

            self.sensor_x = sensor_x
            self.sensor_y = sensor_y
            self.sensor_idx = sensor_idx
            self.mask = mask

    def init_vis_sock(self, vis_outline, vis_sensor, side='right'):
        '''
        sock contour
        '''
        contour_x, contour_y = [], []
        cx, cy = None, None

        for idx in range(1, len(vis_outline)):
            if vis_outline[idx][0] == '*':
                if cx is not None and cy is not None:
                    contour_x.append(cx)
                    contour_y.append(cy)
                cx, cy = [], []
            else:
                x, y = vis_outline[idx].strip().split(',')
                cx.append(float(x))
                cy.append(float(y))

        self.contour_x = contour_x
        self.contour_y = contour_y

        '''
        sock sensor
        '''
        sensor_x, sensor_y, sensor_idx, mask = [], [], [], np.zeros((32, 32))
        for idx in range(0, len(vis_sensor), 4):
            x_gs = np.zeros(4)                  # x axis on glove visualization
            y_gs = np.zeros(4)                  # y axis on glove visualization
            x_ss = np.zeros(4, dtype=np.int)    # x axis on sensor grid
            y_ss = np.zeros(4, dtype=np.int)    # y axis on sensor grid

            idx_list = [idx, idx + 1, idx + 2, idx + 3]
            for ii, jj in enumerate(idx_list):
                x_g, y_g, x_s, y_s = vis_sensor[jj].strip().split(',')
                x_gs[ii] = float(x_g)
                y_gs[ii] = float(y_g)
                x_ss[ii] = int(x_s)
                y_ss[ii] = int(y_s)
                # print('ha', x_g, y_g, x_s, y_s)

            idx_rec = idx // 4
            if idx_rec in [0, 1, 2, 3, 4]:
                st_x, ed_x = x_ss[0], x_ss[2]
                st_y, ed_y = y_ss[0], y_ss[1] + 1
            elif idx_rec in [5]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0], y_ss[1] + 1

            for xx in range(st_x, ed_x):
                for yy in range(st_y, ed_y):

                    xs_ratio = (yy - y_ss[0]) / float(y_ss[1] - y_ss[0])
                    ys_ratio = (xx - x_ss[0]) / float(x_ss[2] - x_ss[0])

                    p0 = [(x_gs[1] - x_gs[0]) * xs_ratio + x_gs[0], (y_gs[1] - y_gs[0]) * xs_ratio + y_gs[0]]
                    p1 = [(x_gs[2] - x_gs[1]) * ys_ratio + x_gs[1], (y_gs[2] - y_gs[1]) * ys_ratio + y_gs[1]]
                    p2 = [(x_gs[2] - x_gs[3]) * xs_ratio + x_gs[3], (y_gs[2] - y_gs[3]) * xs_ratio + y_gs[3]]
                    p3 = [(x_gs[3] - x_gs[0]) * ys_ratio + x_gs[0], (y_gs[3] - y_gs[0]) * ys_ratio + y_gs[0]]

                    R = intersection(line(p0, p2), line(p1, p3))

                    # print(R[0], R[1], xx, yy)

                    sensor_x.append(R[0])
                    sensor_y.append(R[1])
                    sensor_idx.append(xx * 32 + yy)
                    mask[xx, yy] = 1

            self.sensor_x = sensor_x
            self.sensor_y = sensor_y
            self.sensor_idx = sensor_idx
            self.mask = mask

    def init_vis_vest(self, vis_outline, vis_sensor, side='back'):
        '''
        vest contour
        '''
        contour_x, contour_y = [], []
        cx, cy = None, None

        for idx in range(1, len(vis_outline)):
            if vis_outline[idx][0] == '*':
                if cx is not None and cy is not None:
                    contour_x.append(cx)
                    contour_y.append(cy)
                cx, cy = [], []
            else:
                x, y = vis_outline[idx].strip().split(',')
                cx.append(float(x) if side == 'back' else -float(x))
                cy.append(float(y))

        self.contour_x = contour_x
        self.contour_y = contour_y

        '''
        vest sensor
        '''
        sensor_x, sensor_y, sensor_idx, mask = [], [], [], np.zeros((32, 32))
        for idx in range(0, len(vis_sensor), 4):
            x_gs = np.zeros(4)                  # x axis on glove visualization
            y_gs = np.zeros(4)                  # y axis on glove visualization
            x_ss = np.zeros(4, dtype=np.int)    # x axis on sensor grid
            y_ss = np.zeros(4, dtype=np.int)    # y axis on sensor grid

            idx_list = [idx, idx + 1, idx + 2, idx + 3]
            for ii, jj in enumerate(idx_list):
                x_g, y_g, x_s, y_s = vis_sensor[jj].strip().split(',')
                x_gs[ii] = float(x_g)
                y_gs[ii] = float(y_g)
                x_ss[ii] = int(x_s)
                y_ss[ii] = int(y_s)
                # print('ha', x_g, y_g, x_s, y_s)

            idx_rec = idx // 4
            if idx_rec in [0]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0], y_ss[1] + 1

            for xx in range(st_x, ed_x):
                for yy in range(st_y, ed_y):

                    xs_ratio = (yy - y_ss[0]) / float(y_ss[1] - y_ss[0])
                    ys_ratio = (xx - x_ss[0]) / float(x_ss[2] - x_ss[0])

                    p0 = [(x_gs[1] - x_gs[0]) * xs_ratio + x_gs[0], (y_gs[1] - y_gs[0]) * xs_ratio + y_gs[0]]
                    p1 = [(x_gs[2] - x_gs[1]) * ys_ratio + x_gs[1], (y_gs[2] - y_gs[1]) * ys_ratio + y_gs[1]]
                    p2 = [(x_gs[2] - x_gs[3]) * xs_ratio + x_gs[3], (y_gs[2] - y_gs[3]) * xs_ratio + y_gs[3]]
                    p3 = [(x_gs[3] - x_gs[0]) * ys_ratio + x_gs[0], (y_gs[3] - y_gs[0]) * ys_ratio + y_gs[0]]

                    R = intersection(line(p0, p2), line(p1, p3))

                    # print(R[0], R[1], xx, yy)

                    sensor_x.append(R[0] if side == 'back' else -R[0])
                    sensor_y.append(R[1])
                    sensor_idx.append(xx * 32 + yy)
                    mask[xx, yy] = 1

            self.sensor_x = sensor_x
            self.sensor_y = sensor_y
            self.sensor_idx = sensor_idx
            self.mask = mask

    def init_vis_kitchen_glove(self, vis_outline, vis_sensor, side='right'):
        '''
        glove contour
        '''
        contour_x, contour_y = [], []
        cx, cy = None, None

        for idx in range(1, len(vis_outline)):
            if vis_outline[idx][0] == '*':
                if cx is not None and cy is not None:
                    contour_x.append(cx)
                    contour_y.append(cy)
                cx, cy = [], []
            else:
                x, y = vis_outline[idx].strip().split(',')
                cx.append(float(x) if side == 'right' else -float(x))
                cy.append(float(y))

        self.contour_x = contour_x
        self.contour_y = contour_y

        '''
        glove sensor
        '''
        sensor_x, sensor_y, sensor_idx, mask = [], [], [], np.zeros((32, 32))
        for idx in range(0, len(vis_sensor), 4):
            x_gs = np.zeros(4)                  # x axis on glove visualization
            y_gs = np.zeros(4)                  # y axis on glove visualization
            x_ss = np.zeros(4, dtype=np.int)    # x axis on sensor grid
            y_ss = np.zeros(4, dtype=np.int)    # y axis on sensor grid

            idx_list = [idx, idx + 1, idx + 2, idx + 3]
            for ii, jj in enumerate(idx_list):
                x_g, y_g, x_s, y_s = vis_sensor[jj].strip().split(',')
                x_gs[ii] = float(x_g)
                y_gs[ii] = float(y_g)
                x_ss[ii] = int(x_s)
                y_ss[ii] = int(y_s)
                # print('ha', x_g, y_g, x_s, y_s)

            idx_rec = idx // 4
            if idx_rec in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18]:
                st_x, ed_x = x_ss[0], x_ss[2]
                st_y, ed_y = y_ss[0], y_ss[1] + 1
            elif idx_rec in [14, 15]:
                st_x, ed_x = x_ss[0], x_ss[2]
                st_y, ed_y = y_ss[0], y_ss[1]
            elif idx_rec in [16]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0], y_ss[1]
            elif idx_rec in [11, 12]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0] + 1, y_ss[1] + 1
            elif idx_rec in [13, 19]:
                st_x, ed_x = x_ss[0], x_ss[2] + 1
                st_y, ed_y = y_ss[0], y_ss[1] + 1

            for xx in range(st_x, ed_x):
                for yy in range(st_y, ed_y):

                    xs_ratio = (yy - y_ss[0]) / float(y_ss[1] - y_ss[0])
                    ys_ratio = (xx - x_ss[0]) / float(x_ss[2] - x_ss[0])

                    p0 = [(x_gs[1] - x_gs[0]) * xs_ratio + x_gs[0], (y_gs[1] - y_gs[0]) * xs_ratio + y_gs[0]]
                    p1 = [(x_gs[2] - x_gs[1]) * ys_ratio + x_gs[1], (y_gs[2] - y_gs[1]) * ys_ratio + y_gs[1]]
                    p2 = [(x_gs[2] - x_gs[3]) * xs_ratio + x_gs[3], (y_gs[2] - y_gs[3]) * xs_ratio + y_gs[3]]
                    p3 = [(x_gs[3] - x_gs[0]) * ys_ratio + x_gs[0], (y_gs[3] - y_gs[0]) * ys_ratio + y_gs[0]]

                    R = intersection(line(p0, p2), line(p1, p3))

                    # print(R[0], R[1], xx, yy)

                    sensor_x.append(R[0] if side == 'right' else -R[0])
                    sensor_y.append(R[1])
                    sensor_idx.append(xx * 32 + yy)
                    mask[xx, yy] = 1

            self.sensor_x = sensor_x
            self.sensor_y = sensor_y
            self.sensor_idx = sensor_idx
            self.mask = mask

    def init_vis_kuka(self, vis_sensor, vis_mesh_path):
        import open3d as o3d

        '''
        kuka open3d setup
        '''
        mask = np.zeros((32, 32))
        sensor_x, sensor_y, sensor_z, sensor_xn, sensor_yn, sensor_zn, sensor_idx = \
                [], [], [], [], [], [], []
        for i in range(len(vis_sensor)):
            xs, ys, x, y, z, xn, yn, zn = [float(d) for d in vis_sensor[i].strip().split(' ')]
            xs, ys = int(xs), int(ys)
            mask[xs, ys] = 1
            sensor_idx.append(xs * 32 + ys)
            sensor_x.append(x)
            sensor_y.append(y)
            sensor_z.append(z)
            sensor_xn.append(xn)
            sensor_yn.append(yn)
            sensor_zn.append(zn)

        '''
        setup open3d geometries
        '''
        # mesh link
        mesh_link = o3d.io.read_triangle_mesh(vis_mesh_path)
        mesh_link.compute_vertex_normals()
        mesh_link.paint_uniform_color(np.array([0.75, 0.75, 0.75]))

        # mesh cylinder
        mesh_cylinders = []
        for i in range(len(vis_sensor)):
            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.004, height=0.004)
            mesh_cylinder.compute_vertex_normals()
            mesh_cylinder.paint_uniform_color([0.1, 0.1, 0.7])

            # transform mesh cylinder
            transformation = np.identity(4)
            transformation[0, 0] = transformation[1, 1] = transformation[2, 2] = 1  # scale
            transformation[:3, 3] = [sensor_x[i], sensor_y[i], sensor_z[i]]         # translation
            R_2vect(transformation[:3, :3], np.array([0., 0., 1.]),
                    np.array([sensor_xn[i], sensor_yn[i], sensor_zn[i]]))
            mesh_cylinder.transform(transformation)

            mesh_cylinders.append(mesh_cylinder)

        '''
        setup open3d visualizer
        '''
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=480, width=640)

        # add the geometries to the visualizer
        vis.add_geometry(mesh_link)
        for i in range(len(mesh_cylinders)):
            vis.add_geometry(mesh_cylinders[i])

        # setup camera parameters
        intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 480, 365.4020, 365.6674, 319.5, 239.5)
        extrinsic = np.array([
            [0.37972342, -0.07703326, -0.9218872, 0.12736532],
            [-0.92331447, 0.03032716, -0.38284546, 0.0593583],
            [0.05745005, 0.99656718, -0.05960999, 0.32494112],
            [0., 0., 0., 1.]])

        camera_param = o3d.camera.PinholeCameraParameters()
        camera_param.extrinsic = extrinsic
        camera_param.intrinsic = intrinsic

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_param)

        # store the information
        self.vis = vis
        self.mesh_cylinders = mesh_cylinders
        self.COL = MplColorHelper('viridis_r', 0., 1.)

        self.sensor_x = sensor_x
        self.sensor_y = sensor_y
        self.sensor_z = sensor_z
        self.sensor_idx = sensor_idx
        self.mask = mask

    def render(self, response, lim_low, lim_high, text=None, fancy=True):

        if self.sensor_type == 'kuka' and fancy == True:
            # plot kuka (fancy) using open3d

            # get response
            response = response.reshape(-1)[self.sensor_idx]
            s, c = get_size_and_color_for_plt_scatter(
                response, lim_low, lim_high, sensor_type=self.sensor_type)

            for i in range(len(self.mesh_cylinders)):
                mesh_cylinder = self.mesh_cylinders[i]
                mesh_cylinder.paint_uniform_color(self.COL.get_rgb(c[i]))
                self.vis.update_geometry(mesh_cylinder)
            self.vis.poll_events()
            self.vis.update_renderer()

            img = self.vis.capture_screen_float_buffer(False)
            img = np.asarray(img) * 255
            img = img.astype(np.uint8)[..., ::-1]

        else:
            # plot others using matplotlib
            plt.rcParams["figure.figsize"] = (6, 6)
            fig = plt.figure()
            ax = fig.add_subplot(111)

            # plot text
            if text is not None:
                font = {'family': 'serif',
                        'color':  'darkred',
                        'weight': 'normal',
                        'size': 16}
                plt.text(-5, 13, text, fontdict=font)

            if fancy:
                # get response
                response = response.reshape(-1)[self.sensor_idx]
                s, c = get_size_and_color_for_plt_scatter(
                    response, lim_low, lim_high, sensor_type=self.sensor_type)

                # plot contour
                for ii in range(len(self.contour_x)):
                    plt.plot(self.contour_x[ii], self.contour_y[ii], 'k-', linewidth=1)
                ax.set_aspect('equal')

                # plot sensors
                plt.scatter(self.sensor_x, self.sensor_y, s=s, c=c, cmap='viridis_r',
                            linewidths=.2, edgecolors='k')
                # plt.colorbar()
                plt.clim(0., 1.)
                plt.axis('off')
                # plt.show()

            else:
                _, c = get_size_and_color_for_plt_scatter(
                    response, lim_low, lim_high, sensor_type=self.sensor_type)
                plt.imshow(c)
                plt.colorbar()
                plt.clim(0., 1.)

            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[..., ::-1]

            plt.gcf().clear()
            plt.close()

        return img

    def merge_frames(self, frames, nx, ny):
        h, w = frames[0].shape[0:2]
        frame = np.stack(frames).reshape(nx, ny, h, w, 3)
        frame = frame.transpose(0, 2, 1, 3, 4).reshape(nx * h, ny * w, 3)
        return frame

