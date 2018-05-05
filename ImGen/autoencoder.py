from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import image_reader


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


# G(z)
def autoencoder(x, isTrain=True, reuse=False):

    conv1 = tf.layers.conv2d(x, 64, [4, 4], strides=(2, 2), padding='same')
    h1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

    conv2 = tf.layers.conv2d(h1, 128, [4, 4], strides=(2, 2), padding='same')
    h2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

    conv3 = tf.layers.conv2d(h2, 256, [4, 4], strides=(2, 2), padding='same')
    h3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

    conv4 = tf.layers.conv2d(h3, 512, [4, 4], strides=(2, 2), padding='same')
    h4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

    conv5 = tf.layers.conv2d(h4, 1024, [4, 4], strides=(2, 2), padding='same')
    h5 = lrelu(tf.layers.batch_normalization(conv5, training=isTrain), 0.2)

    conv6 = tf.layers.conv2d(h5, 1024, [4, 4], strides=(2, 2), padding='same')
    h6 = lrelu(tf.layers.batch_normalization(conv6, training=isTrain), 0.2)

    conv7 = tf.layers.conv2d(h6, 1024, [4, 4], strides=(2, 2), padding='same')
    h7 = lrelu(tf.layers.batch_normalization(conv7, training=isTrain), 0.2)

    ############################################################################

    conv8 = tf.layers.conv2d_transpose(h7, 1024, [4, 4], strides=(1, 1), padding='valid')
    h8 = lrelu(tf.layers.batch_normalization(conv8, training=isTrain), 0.2)

    conv9 = tf.layers.conv2d_transpose(h8, 512, [4, 4], strides=(2, 2), padding='same')
    h9 = lrelu(tf.layers.batch_normalization(conv9, training=isTrain), 0.2)

    conv10 = tf.layers.conv2d_transpose(h9, 256, [4, 4], strides=(2, 2), padding='same')
    h10 = lrelu(tf.layers.batch_normalization(conv10, training=isTrain), 0.2)

    conv11 = tf.layers.conv2d_transpose(h10, 128, [4, 4], strides=(2, 2), padding='same')
    h11 = lrelu(tf.layers.batch_normalization(conv11, training=isTrain), 0.2)

    conv12 = tf.layers.conv2d_transpose(h11, 64, [4, 4], strides=(2, 2), padding='same')
    h12 = lrelu(tf.layers.batch_normalization(conv12, training=isTrain), 0.2)

    conv13 = tf.layers.conv2d_transpose(h12, 3, [4, 4], strides=(2, 2), padding='same')
    h13 = lrelu(tf.layers.batch_normalization(conv13, training=isTrain), 0.2)

    return h13



x = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
isTrain = tf.placeholder(dtype=tf.bool)

net = autoencoder(x, isTrain)

train_set = image_reader.get_images(
    '----',
    200000,
    shape=(128, 128, 3))
