"""

PointNet model
based on Charles R. Qi

"""

import os
import sys
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import tf_util


def placeholder_inputs(batch_size, num_point, num_feature):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_feature))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl

def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNxF, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # CONV layers
    net = tf_util.conv2d(input_image, 64, [1, 9], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    points_feat1 = tf_util.conv2d(net, 2048, [1, 1], padding='VALID', stride=[1, 1],
                                  bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # MAX Pooling
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point, 1], padding='VALID', scope='maxpool1')
    # FC layers
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # CONCAT
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV Layers
    net = tf_util.conv2d(points_feat1_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv6')
    net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv8')
    net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training, scope='conv9')

    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

    net = tf_util.conv2d(net, num_class, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, scope='conv10')
    net = tf.squeeze(net, [2])

    return net, end_points


def get_loss(pred, label, smpw):

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__ == "__main__":
    with tf.Graph().as_default():
        inputs = tf.zeros((12,8000,9))
        net = get_model(inputs, tf.constant(True), 8)
        print(net)