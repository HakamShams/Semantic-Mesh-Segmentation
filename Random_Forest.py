'''

Utility function for Random forest with a simple grid search with two parameters

Author: Hakam Shams
Date: Novemebr 2019

Input:  train_file    : txt file NxF for training, first 3 columns are the centers of gravity of faces and last column is the labels
        test_file     : txt file NxF for testing, first 3 columns are the centers of gravity of faces and last column is the labels
        max_features  : number of features to consider when looking for the best split
        data_path_out : output path to store the results
        trees         : list of trees for grid search
        depthes       : list of depthes for grid search

        is_visualize   : visualize the results

Output: h5 file stored in the same directory as data_path_out , get items:
        data, first three column COG then features
        label, predicted label of best OA
        gt, ground truth label

Dependencies: numpy - os - sys - sklearn - matplotlib - h5py

'''

import numpy as np
from time import clock
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import h5py
from matplotlib.legend_handler import HandlerLine2D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Input:

train_file = ''
test_file  = ''

data_path_out = ''

trees = np.arange(10, 110, 5)
depthes = np.arange(5, 21, 1)

max_feature = 'auto'

is_visualize = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Algorithm:

data = np.loadtxt(train_file)
x = data[:,3:-1]
y = data[:,-1]
print('Number of faces for training: ', x.shape[0])

data_test = np.loadtxt(test_file)
x_test = data_test[:, 3:-1]
y_test = data_test[:, -1]
print('Number of faces for testing: ', x_test.shape[0])
print('Number of features: ', x.shape[1])

results_test, time_test = [], []
y_pred_all = None

for tree in trees:
    for depth in depthes:

        rfc = RandomForestClassifier(n_jobs=-1, n_estimators=tree, max_depth=depth, bootstrap=True, max_features=max_feature)

        starting_time = clock()

        rfc.fit(x, y)

        ending_time = clock()
        duration = ending_time - starting_time

        y_pred = rfc.predict(x_test)

        if y_pred_all is None:
            y_pred_all = y_pred
        else:
            y_pred_all = np.hstack((y_pred_all,y_pred))

        cm = confusion_matrix(y_test, y_pred)
        OA = np.diagonal(cm).sum() / np.matrix(cm).sum()

        results_test.append(OA)
        time_test.append(duration)

data_pred_all = y_pred_all.reshape(len(trees), -1, len(depthes))

data_all = np.array(results_test).reshape(-1, len(depthes))
data_time = np.array(time_test).reshape(-1, len(depthes))
best_tree, best_depth = np.unravel_index(np.argmax(data_all, axis=None), data_all.shape)
print('\nbest tree {} with depth {}'.format(trees[best_tree], depthes[best_depth]))
print('best OA {}'.format(data_all[best_tree, best_depth]))

if is_visualize:

    for i in range(len(depthes)):
        line, = plt.plot(trees, data_all[:, i],'-o', label='depth {}'.format(depthes[i]))
    plt.title('OA of testing tile')
    plt.ylabel('OA %')
    plt.xlabel('n_trees')
    plt.legend(loc='best')
    plt.xticks(trees)
    plt.grid()
    plt.show()

    for i in range(len(depthes)):
        line, = plt.plot(trees, data_time[:, i],'-o', label='depth {}'.format(depthes[i]))
    plt.title('Training time')
    plt.ylabel('sec')
    plt.xlabel('n_trees')
    plt.legend(loc='best')
    plt.xticks(trees)
    plt.grid()
    plt.show()

with h5py.File(data_path_out + 'RE_tree_{}_depth_{}.h5'.format(trees[best_tree],depthes[best_depth]), 'w') as hdf:
    hdf.create_dataset('data', data=data_test[:, :-1], dtype='float64')
    hdf.create_dataset('label', data=data_pred_all[best_tree, :, best_depth], dtype='float32')
    hdf.create_dataset('gt', data=y_test, dtype='float32')

