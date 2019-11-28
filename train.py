'''

Utility mesh function for training

'''

import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import batch_prep

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
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--sampling', default='random', help='Sampling method [default: random]')
parser.add_argument('--num_point', type=int, default=18000, help='Point Number [default: 18000]')
parser.add_argument('--num_class', type=int, default=8, help='Class Number [default: 8]')
parser.add_argument('--num_feature', type=int, default=9, help='Feature Number [default: 9]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--scale', action='store_true', help='scale for PointNet [default: False]')
parser.add_argument('--eval', default='acc', help='evaluate best epoch based on acc/loss [default: acc]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default= 20000, help='Decay step for lr decay [default: 20000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

DATA_PATH_train = os.path.join(BASE_DIR, 'data-train')
DATA_PATH_test = os.path.join(BASE_DIR, 'data-val')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Training:

EPOCH_CNT = 0
SAMPLING = FLAGS.sampling
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_CLASSES = FLAGS.num_class
NUM_FEATURE = FLAGS.num_feature
SCALE = FLAGS.scale
EVAL = FLAGS.eval

MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,          # Decay step.
        DECAY_RATE,          # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def train():

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, smpws_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)

            loss = MODEL.get_loss(pred, labels_pl, smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'smpws_pl': smpws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        TRAIN_DATASET = batch_prep.HessigheimDataset(root=DATA_PATH_train, nb_classes=NUM_CLASSES, scale=SCALE,
                                                     num_faces=NUM_POINT,sampling=SAMPLING,mode='train')
        TEST_DATASET = batch_prep.HessigheimDataset(root=DATA_PATH_test, nb_classes=NUM_CLASSES, scale=SCALE,
                                                    num_faces=NUM_POINT,sampling=SAMPLING,mode='val')

        best_acc  = -1
        best_loss = -1

        def get_batch_wdp(dataset, idxs, start_idx, end_idx, num_feature):
            """ get batch with drop-out, default 0-40 % drop-out"""
            bsize = end_idx-start_idx
            batch_data = np.zeros((bsize, NUM_POINT, num_feature))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
            for i in range(bsize):
                ps,seg,smpw = dataset[idxs[i+start_idx]]
                batch_data[i,...] = ps
                batch_label[i,:] = seg
                batch_smpw[i,:] = smpw

                dropout_ratio = np.random.random() * 0.4
                drop_idx = np.where(np.random.random((ps.shape[0]))<=dropout_ratio)[0]
                batch_data[i,drop_idx,:] = batch_data[i,0,:]
                batch_label[i,drop_idx] = batch_label[i,0]
                batch_smpw[i,drop_idx] *= 0
            return batch_data, batch_label, batch_smpw

        def get_batch(dataset, idxs, start_idx, end_idx, num_feature):
            """ get batch without dropout"""
            bsize = end_idx-start_idx
            batch_data = np.zeros((bsize, NUM_POINT, num_feature))
            batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
            batch_smpw = np.zeros((bsize, NUM_POINT), dtype=np.float32)
            for i in range(bsize):
                ps,seg,smpw = dataset[idxs[i+start_idx]]
                batch_data[i,...] = ps
                batch_label[i,:] = seg
                batch_smpw[i,:] = smpw
            return batch_data, batch_label, batch_smpw

        def train_one_epoch(sess, ops, train_writer, TRAIN_DATASET):
            """ ops: dict mapping from string to tf ops """
            is_training = True

            # Shuffle train samples
            train_idxs = np.arange(0, len(TRAIN_DATASET))
            np.random.shuffle(train_idxs)
            num_batches = len(TRAIN_DATASET)//BATCH_SIZE

            log_string(str(datetime.now()))

            total_correct = 0
            total_seen = 0
            loss_sum = 0
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx+1) * BATCH_SIZE
                batch_data, batch_label, batch_smpw = get_batch_wdp(TRAIN_DATASET, train_idxs, start_idx, end_idx, NUM_FEATURE)

                feed_dict = {ops['pointclouds_pl']: batch_data,
                             ops['labels_pl']: batch_label,
                             ops['smpws_pl']:batch_smpw,
                             ops['is_training_pl']: is_training,}
                summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                    ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)

                train_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum(pred_val == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE*NUM_POINT)
                loss_sum += loss_val

                if (batch_idx+1)%10 == 0:
                    log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
                    log_string('mean loss: %f' % (loss_sum / 10))
                    log_string('accuracy: %f' % (total_correct / float(total_seen)))
                    total_correct = 0
                    total_seen = 0
                    loss_sum = 0

        def eval_one_epoch(sess, ops, test_writer):
            """ ops: dict mapping from string to tf ops """
            global EPOCH_CNT
            is_training = False
            test_idxs = np.arange(0, len(TEST_DATASET))
            num_batches = len(TEST_DATASET) // BATCH_SIZE

            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]

            log_string(str(datetime.now()))
            log_string('---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

            labelweights = np.zeros(8)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                batch_data, batch_label, batch_smpw = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx, NUM_FEATURE)

                feed_dict = {ops['pointclouds_pl']: batch_data,
                             ops['labels_pl']: batch_label,
                             ops['smpws_pl']: batch_smpw,
                             ops['is_training_pl']: is_training}

                summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                              ops['loss'], ops['pred']], feed_dict=feed_dict)
                test_writer.add_summary(summary, step)
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label) & (batch_label > -1) & (batch_smpw > -1))
                total_correct += correct
                total_seen += np.sum((batch_label > -1) & (batch_smpw > -1))
                loss_sum += loss_val
                tmp, _ = np.histogram(batch_label, range(9))
                labelweights += tmp
                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l) & (batch_smpw > -1))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l) & (batch_smpw > -1))

            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class[:]) / (np.array(total_seen_class[:], dtype=np.float) + 1e-6))))

            EPOCH_CNT += 1

            return total_correct / float(total_seen), loss_sum / float(num_batches)

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, TRAIN_DATASET)
            if epoch % 3 == 0:
                acc_all, loss_all = eval_one_epoch(sess, ops, test_writer)

            # Save the variables
            if EVAL == 'acc':
                if acc_all > best_acc:
                    best_acc = acc_all
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Model saved in file: %s" % save_path)
            else:
                if loss_all > best_loss:
                    best_loss = loss_all
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                    log_string("Model saved in file: %s" % save_path)
            # Save the variables
            if epoch % 5 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

    LOG_FOUT.close()

if __name__ == "__main__":

    train()