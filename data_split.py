'''

Utility mesh function for data preparation, split the data into tiles

Author: Hakam Shams
Date: Novemebr 2019

Input:  data_path     : txt file NxF, first 3 columns are the centers of gravity of faces and last column is the labels
        data_path_out : output path to store the tiles
        out_format    : output format of the stored tiles txt or h5, default txt
        mode          : train or val
        min_points    : threshold for minimum number of points, default 4000 points
        std           : threshold for standard deviation, default 35 %
        tile_size     : size of the tile, default 50 m x 50 m
        overlap       : overlap of the tiles, default 80 %
        m             : approximate margin at borders, default 10 m

Output: data_cur      : numpy array nxF, first 3 columns are the centers of gravity of faces and last column is the labels

Dependencies: numpy - os - h5py - sys

'''

import h5py
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
np.set_printoptions(suppress=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Input:

data_file     = '/.txt'
data_path_out = os.path.join(BASE_DIR, 'data-val')

out_format = 'h5'
mode = 'val'

min_points = 4000
std = 0.35

tile_size = 50
overlap = 0.8
m = 10

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Check:

if not data_file.endswith('txt'):
    print('unsupported input file format is txt')
    quit()

if not os.path.exists(data_path_out):
    os.mkdir(data_path_out)

data_path_out = data_path_out.rstrip('/')

if out_format != 'txt' and out_format != 'h5':
    print('unsupported output format, supported format are txt or h5')
    quit()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Algorithm:

data_all = np.loadtxt(data_file)

file_name = os.path.basename(os.path.normpath(data_file))
print('File name:',file_name)
print('Number of faces:', data_all.shape[0])

d = tile_size - (tile_size * overlap)

x_min = data_all[:, 0].min(axis=0)
x_max = data_all[:, 0].max(axis=0)
y_min = data_all[:, 1].min(axis=0)
y_max = data_all[:, 1].max(axis=0)

x_lim = np.arange(x_min - m, (x_max - tile_size) + m, d)
y_lim = np.arange(y_min - m, (y_max - tile_size) + m, d)

if abs(x_lim[-1] + tile_size - x_max) < m:
    x_lim = np.append(x_lim, x_lim[-1] + d)
if abs(y_lim[-1] + tile_size - y_max) < m:
    y_lim = np.append(y_lim, y_lim[-1] + d)

n_tiles = 0

for i in y_lim:
    for k in x_lim:

        ind = np.where(np.logical_and(np.logical_and((k <= data_all[:,0]), (data_all[:,0] <= k + tile_size)),
                           np.logical_and((i <= data_all[:,1]),(data_all[:,1] <= i + tile_size))))

        data_cur = data_all[ind]
        if data_cur.shape[0] < min_points:
            continue

        if mode == 'train':
            h, _ = np.histogram(data_cur[:,-1])
            st = np.std(h/float(np.sum(h)), ddof=1)

            if st > std:
                continue

        n_tiles += 1

        if out_format == 'txt':
            np.savetxt(data_path_out + '/tile-{}-{}.txt'.format(np.round(i) + (tile_size / 2), np.round(k) + (tile_size / 2)),
                       data_cur, fmt='%.6f', delimiter=' ')
        elif out_format == 'h5':
            with h5py.File(data_path_out + '/tile-{}-{}.h5'.format(np.round(i) + (tile_size / 2), np.round(k) + (tile_size / 2)), 'w') as hdf:
                hdf.create_dataset('data', data=data_cur[:,:-1], dtype='float64')
                hdf.create_dataset('label', data=data_cur[:,-1], dtype='float32')

print('Number of tiles: ', n_tiles)

