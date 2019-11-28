'''

Utility mesh function for LiDAR-Mesh association

Author: Hakam Shams
Date: Novemebr 2019

Input:  LiDAR_file     : LiDAR tile, first three column are XYZ, supported format [h5, txt]
        obj_file       : obj file
        face_file      : COG file of the labeled face, first three column are XYZ
        data_path_out  : output path to store the results
        out_format     : output format of the stored results txt or h5
        thresholds_pos : thresholds in normal vector direction
        thresholds_neg : thresholds in opposite direction of normal vector direction
        opt            : if opt == 1: set features for unassociated faces from k nearest neighborhood
                         if opt == 2: set features for unassociated faces to zeros
        k_voting       : k nearest neighborhood for opt 1

Output: labels_l       : LiDAR Points with its corresponding Face Id from mesh
        threshold      : Faces of mesh with its corresponding used threshold
        features       : Faces of mesh with its corresponding LiDAR features

Dependencies: numpy - os - h5py - scipy - sklearn - pywavefront - sys

'''

from __future__ import division
import numpy as np
import os
import h5py
import sys
from sklearn.neighbors import BallTree
import pywavefront
from scipy.spatial import cKDTree
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Input:

LiDAR_file = '/'
face_file  = '/'

obj_file = '/'
data_path_out = '/'

out_format = 'txt'

thresholds_pos = [0.05, 0.10, 0.15]
thresholds_neg = [-0.20, -0.40, -0.80]

opt = 2
k_voting = 50

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Check:

if not os.path.exists(data_path_out):
    os.mkdir(data_path_out)
if out_format != 'txt' and out_format != 'h5':
    print('unsupported output format, supported format are txt or h5')
    quit()
if len(thresholds_pos) != len(thresholds_neg):
    print('thresholds must have the same length')
    quit()
data_path_out = data_path_out.rstrip('/')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Algorithm:

if LiDAR_file.endswith('h5'):
    hdf = h5py.File(LiDAR_file, mode='r')
    data = np.array(hdf['data'])

elif LiDAR_file.endswith('txt'):
    data = np.loadtxt(LiDAR_file)
else:
    print('unsupported LiDAR file format, supported format are txt or h5')

print('Number of LiDAR Points: ', data.shape[0])

file_name = os.path.basename(os.path.splitext(obj_file)[0])
print('Tile: ', file_name)

# Compute normal vectors and Radii for search

scene = pywavefront.Wavefront(
        obj_file,
        create_materials=True,
        collect_faces=True)

F = np.array(scene.mesh_list[0].faces)
V = np.array(scene.vertices, dtype='float64')
print("Number of Faces: ", np.array(scene.mesh_list[0].faces).shape)
print("Number of Vertices: ", np.array(scene.vertices).shape)

C = np.zeros((F.shape[0],3))
N = np.zeros((F.shape[0],3))
R = np.zeros(F.shape[0])

def calc_distances(p0, p1):
    return np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p1[2]) ** 2)

for i in range(F.shape[0]):
    p1 = np.array([V[F[i, 0], 0], V[F[i, 0], 1], V[F[i, 0], 2]])
    p2 = np.array([V[F[i, 1], 0], V[F[i, 1], 1], V[F[i, 1], 2]])
    p3 = np.array([V[F[i, 2], 0], V[F[i, 2], 1], V[F[i, 2], 2]])

    v1 = p3 - p1
    v2 = p2 - p1

    N[i, :] = np.cross(v1, v2) * -1

    C[i, :] = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3, (p1[2] + p2[2] + p3[2]) / 3]

    R[i] = np.array([calc_distances(p1, C[i, :]), calc_distances(p2, C[i, :]), calc_distances(p3, C[i, :])]).max()

# consider just the labeled faces

if face_file.endswith('h5'):
    hdf = h5py.File(face_file, mode='r')
    labeled_data = np.array(hdf['data'])

elif face_file.endswith('txt'):
    labeled_data = np.loadtxt(face_file)
else:
    print('unsupported COG file format, supported format are txt or h5')

tree = cKDTree(C[:,0:3])

index_labeled = []

for i in range(labeled_data.shape[0]):
    _, ind = tree.query(labeled_data[i, 0:3], k=1)
    index_labeled.append(ind)

N = np.take(N, index_labeled, axis=0)
N = np.nan_to_num(N)
C = np.take(C, index_labeled, axis=0)
R = np.take(R, index_labeled, axis=0)
F = np.take(F, index_labeled, axis=0)

# Accelerate the search, cut the LiDAR tile to fit to the COG tile

min_x = V[:, 0].min()
max_x = V[:, 0].max()
min_y = V[:, 1].min()
max_y = V[:, 1].max()

data_valid = np.logical_and(np.logical_and((min_y <= data[:,1]), (data[:,1] <= max_y)),
                            np.logical_and((min_x <= data[:,0]), (data[:,0] <= max_x)))

data_valid = np.where(data_valid)

data = data[data_valid]

print('LiDAR Points in the tile: ', data.shape[0])

# Associate LiDAR feature and indices for LiDAR points based on barycentric coordinates

tree = BallTree(data[:, 0:3])
tree_K = cKDTree(data[:, 0:3])

features = np.zeros((C.shape[0], data.shape[1]))
features[:,0:3] = C[:,0:3]

labels_l = np.zeros((data.shape[0],4))
labels_l[:,0:3] = data[:,0:3]
labels_l[:,-1] = -1

threshold = np.zeros((C.shape[0],4))
threshold[:,0:3] = C[:,0:3]
threshold[:,-1] = -1

num_face = 0

for i in range(C.shape[0]):

    ind = tree.query_radius(np.expand_dims(C[i, :], axis=0), r=R[i])
    ind = ind.squeeze()
    ind = ind.tolist()

    if len(ind) == 0:
        if opt == 1:
            _, index = tree_K.query(np.expand_dims(C[i, :], axis=0), k=k_voting)
           # features[i, 3:] = np.hstack((np.median(data[index, 4]), np.median(data[index, 6]))) # take just two features
            features[i, 3:] = np.median(data[index,3:], axis=0)
        elif opt == 2:
            continue

    else:

        # vertices
        p1 = np.array([V[F[i, 0], 0], V[F[i, 0], 1], V[F[i, 0], 2]])
        p2 = np.array([V[F[i, 1], 0], V[F[i, 1], 1], V[F[i, 1], 2]])
        p3 = np.array([V[F[i, 2], 0], V[F[i, 2], 1], V[F[i, 2], 2]])

        # barycentric coordinates
        """ Based on Ericson, Christer. 2005. Real-time collision detection. Amsterdam, Elsevier"""

        v0 = p2 - p1
        v1 = p3 - p1

        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot11 = float(np.dot(v1, v1))

        Denom = float(dot00 * dot11 - dot01 * dot01)

        if Denom == 0 or np.sqrt(N[i, 0]**2 + N[i, 1]**2 + N[i, 2]**2) == 0:
            continue

        X = data[ind, 0:3]

        VX = np.array(X - C[i, :])

        n = N[i, :]/ np.sqrt(N[i, 0]**2 + N[i, 1]**2 + N[i, 2]**2)
        # distance to plane
        d = np.dot(VX, n.T)

        for t in range(len(thresholds_pos)):
            ind_j = np.logical_and((thresholds_neg[t] <= d), (d <= thresholds_pos[t]))
            j = np.squeeze(np.argwhere(ind_j), axis=-1)
            # project the points within thresholds
            projected = X[j, :] - (np.expand_dims(d[j], axis=-1) * n)

            v2 = projected - np.tile(p1, (projected.shape[0], 1))

            dot20 = np.dot(v2, v0.T).astype(np.float)
            dot21 = np.dot(v2, v1.T).astype(np.float)

            v = (dot11 * dot20 - dot01 * dot21) / Denom
            w = (dot00 * dot21 - dot01 * dot20) / Denom
            # check if point is in the plane based on barycentric coordinates
            k = np.logical_and(np.logical_and(np.logical_and((float(0) < v), (v < float(1))),
                                              np.logical_and((float(0) < w), (w < float(1)))), (v + w < float(1)))

            k = np.squeeze(np.argwhere(k), axis=-1)

            index = np.array(j)[k]
            index = ind[index]

            if len(index) != 0:
                num_face += 1
                # assign the used threshold to the face
                threshold[i, -1] = t
                # assign the face Id for liDAR points
                labels_l[index,-1] = i
                # assign LiDAR features for the face
                features[i, 3:] = np.median(data[index, 3:], axis=0)
                break
            else:
                if t == len(thresholds_pos) - 1:
                    if opt == 1:
                        _, index = tree_K.query(np.expand_dims(C[i, :], axis=0), k=k_voting)
                        features[i, 3:] = np.median(data[index, 3:], axis=0)

                    elif opt == 2:
                        continue
                continue
        continue

print('number of associated faces: ', num_face)
num_point = np.where(labels_l[:,-1] > -1)

print('Number of associated points: ', np.array(num_point).shape[1])

if out_format == 'txt':

    np.savetxt(data_path_out + '/' + file_name + '_labels.txt', labels_l, fmt='%.6f', delimiter=' ')
    np.savetxt(data_path_out + '/' + file_name + '_thresholds.txt', threshold, fmt='%.6f', delimiter=' ')
    np.savetxt(data_path_out + '/' + file_name + '_features.txt', features, fmt='%.6f', delimiter=' ')

elif out_format == 'h5':

    with h5py.File(data_path_out + '/' + file_name + '_labels.h5', 'w') as hdf:
        hdf.create_dataset('data', data=labels_l, dtype='float64')

    with h5py.File(data_path_out + '/' + file_name + '_threshold.h5', 'w') as hdf:
        hdf.create_dataset('data', data=threshold, dtype='float64')

    with h5py.File(data_path_out + '/' + file_name + '_features.h5', 'w') as hdf:
        hdf.create_dataset('data', data=features, dtype='float64')