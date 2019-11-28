'''

Utility mesh function to generate predictions of a test tile

Author: Hakam Shams based on Charles R. Qi
Date: Novemebr 2019

Input:  data_path     : test tile for prediction
        data_path_out : path to store the GT of predicted batches
        pred_path_out : path to store the prediction of predicted batches

Output: h5 file stored in the same directory as data_path_out , get items:
    data, first three column COG then features
    label, predicted label based on naive majority voting
    label cat, aggregated predictions as one hot
    label_probability, predicted label based on averaged maximum probability across all classes
    label_prob, maximum averaged probability
    gt, ground truth label

Dependencies: numpy - os - sys - h5py - argparse - importlib - tensorflow - scipy - sklearn - keras

'''

import h5py
import sys
import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
from scipy.spatial import cKDTree
from sklearn import preprocessing
from keras.utils import np_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Input:

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet', help='Model name [default: pointnet]')
parser.add_argument('--num_class', type=int, default=8, help='Class Number [default: 8]')
parser.add_argument('--space', type=int, default=10, help='distance between batches [default: 10]')
parser.add_argument('--num_point', type=int, default=18000, help='Point Number [default: 18000]')
parser.add_argument('--scale', action='store_true', help='scale for PointNet [default: False]')
parser.add_argument('--ckpt', default='pointnet/model.ckpt', help='Checkpoint file')
FLAGS = parser.parse_args()

data_path = '/'
data_path_out = BASE_DIR + '/pred_knn'
pred_path_out = BASE_DIR + '/pred_knn'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Predicting:

MODEL = importlib.import_module(FLAGS.model)
is_scale = FLAGS.scale

CHECKPOINT = FLAGS.ckpt
SPACE = FLAGS.space
NUM_CLASSES = FLAGS.num_class
GPU_INDEX = FLAGS.gpu
num_points = FLAGS.num_point
m_name = os.path.basename(os.path.splitext(os.path.dirname(os.path.abspath(CHECKPOINT)))[0])

file_name = os.path.basename(os.path.splitext(data_path)[0])

if not os.path.exists(data_path_out):
    os.mkdir(data_path_out)

if not os.path.exists(pred_path_out):
    os.mkdir(data_path_out)

GT_path = os.path.join(data_path_out, "GT", m_name, file_name)
Pred_path = os.path.join(data_path_out, "Pred", m_name, file_name)

if not os.path.exists(GT_path):
    os.makedirs(GT_path)
if not os.path.exists(Pred_path):
    os.makedirs(Pred_path)


def predict_batch(sess, ops, data):
    is_training = False
    data = np.expand_dims(data,axis=0)
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['is_training_pl']: is_training}

    pred_val = sess.run([ops['pred']], feed_dict=feed_dict)
    pred_val = pred_val[0][0]
    pred_prob = np.array(pred_val)
    pred_val = np.argmax(pred_val, 1)
    return pred_val, pred_prob


def predict():
    with tf.device('/gpu:' + str(GPU_INDEX)):

        if data_path.endswith('h5'):
            hdf = h5py.File(data_path, mode='r')
            data_all = np.array(hdf['data'])
            label_all = np.array(hdf['label'])

        elif data_path.endswith('txt'):
            data_tmp = np.loadtxt(data_path)
            data_all = data_tmp[:, :-1]
            label_all = data_tmp[:, -1]

        num_feature = data_all.shape[1]

        data_cur = np.zeros((data_all.shape[0], data_all.shape[1]+1))
        for i in range(data_all.shape[0]):
            data_cur[i,0] = i
        data_cur[:,1:] = np.copy(data_all[:,:])

        # kNN Grid
        " KNN Grid is based on https://github.com/lwiniwar/alsNet "
        centers = []
        cur_ind = 0
        spacing = SPACE
        min_x = data_cur[:, 1].min()
        max_x = data_cur[:, 1].max()
        min_y = data_cur[:, 2].min()
        max_y = data_cur[:, 2].max()

        num_rows = (max_x - min_x - spacing / 2) // (spacing) + 1
        num_cols = (max_y - min_y - spacing / 2) // (spacing) + 1
        num_batches = int(num_cols * num_rows)

        for i in range(num_batches):
            if cur_ind >= num_batches:
                break
            centers.append([min_x + spacing / 2 + (cur_ind // num_cols) * spacing,
                            min_y + spacing / 2 + (cur_ind % num_cols) * spacing, data_cur[:, 3].mean()])

            cur_ind += 1
        #

        tree = cKDTree(data_all[:, 0:3])

        pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(1, num_points, num_feature=num_feature)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Restore variables from disk.
        saver.restore(sess, CHECKPOINT)
        #print("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred}

        for center in centers:

            _, ind = tree.query(center, k=num_points)

            data_test = data_cur[ind,:]
            label_test = label_all[ind]

            with h5py.File("{}/{}.h5".format(GT_path, np.round(center)), 'w') as hdf:
                hdf.create_dataset('data', data=data_test, dtype='float64')
                hdf.create_dataset('label', data=label_test, dtype='float32')

            # Prediction
            if is_scale:
                data_test_scale = np.copy(data_test)
                scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
                data_test_scale[:,1:4] = scaler.fit_transform(data_test[:, 1:4])
                pred_labels, pred_prob = predict_batch(sess, ops, data_test_scale[:, 1:])
            else:
                pred_labels, pred_prob = predict_batch(sess, ops, data_test[:,1:])

            with h5py.File("{}/{}.h5".format(Pred_path, np.round(center)), 'w') as hdf:
                hdf.create_dataset('data', data=data_test, dtype='float64')
                hdf.create_dataset('label', data=pred_labels, dtype='float32')
                hdf.create_dataset('label_prob', data=pred_prob, dtype='float32')

        files = os.listdir(Pred_path)

        label_cat = np.zeros((data_all.shape[0], NUM_CLASSES))
        a_prob = np.zeros((data_all.shape[0], NUM_CLASSES))

        for file in files:
            if file.endswith('h5') == False:
                continue
            hdf = h5py.File(Pred_path + '/{}'.format(file), mode='r')
            data_batch = hdf['data']
            label_batch = hdf['label']
            Y_prob = hdf['label_prob']

            Y = np_utils.to_categorical(np.array(label_batch), NUM_CLASSES)

            for i, k in enumerate(data_batch[:, 0]):
                k = int(k)
                label_cat[k] = Y[i] + label_cat[k]
                a_prob[k] = Y_prob[i] + a_prob[k]

        Y_n = label_cat.sum(axis=1)

        label = np.ndarray.argmax(label_cat, axis=1)
        label_probability = np.zeros((data_all.shape[0], NUM_CLASSES))

        for j in range(len(label_probability)):
            label_probability[j] = a_prob[j] / Y_n[j]

        label_prob = np.ndarray.max(label_probability, axis=1)
        label_p = np.ndarray.argmax(label_probability, axis=1)

        index = np.array(np.argwhere(label_cat.sum(axis=1) == 0))
        index_not = np.isin(np.arange(label_p.shape[0]), index)
        index_not = np.squeeze(np.where(index_not == False), axis=0)
        tree = cKDTree(np.copy(data_all[index_not, 0:3]))
        for i in index:
            _, j = tree.query(data_all[i, 0:3], k=25)
            label_p[i] = np.argmax(np.bincount(np.squeeze(label_p[j], axis=0)))

        with h5py.File(pred_path_out + '/{}-{}.h5'.format(m_name,file_name), 'w') as hdf:
            hdf.create_dataset('data', data=data_all, dtype='float64')
            hdf.create_dataset('label', data=label, dtype='float32')
            hdf.create_dataset('label_cat', data=label_cat, dtype='float32')
            hdf.create_dataset('label_probability', data=label_p, dtype='float32')
            hdf.create_dataset('label_prob', data=label_prob, dtype='float32')
            hdf.create_dataset('gt', data=label_all, dtype='float32')

if __name__ == "__main__":

    predict()