import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools


# def show_result(test_images, dir='LFW'):
#     idx = 0
#     for img in test_images:
#         path = dir + '/' + str(idx)
#         idx += 1
#         plt.imshow(img, interpolation='nearest')
#         plt.savefig(path)
#     plt.close()


def show_result(test_images, d=5, path='inter.jpg'):
    size_figure_grid = d
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(d, d))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k], cmap='gray')
    plt.savefig(path)
    plt.close()


def extract_2dim_interpolation():
    fmeta = '/home/juraj/Desktop/DIPLOMSKI/results/VAE/MNIST/mnist_2dim_vae/model/model.ckpt.meta'
    f = '/home/juraj/Desktop/DIPLOMSKI/results/VAE/MNIST/mnist_2dim_vae/model'
    sess = tf.Session()
    saver = tf.train.import_meta_graph(fmeta)
    saver.restore(sess, tf.train.latest_checkpoint(f))

    graph = tf.get_default_graph()
    z = graph.get_tensor_by_name('encoder/add:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    img = graph.get_tensor_by_name('decoder/Reshape_1:0')

    # z_ = np.random.normal(0, 1, (10, 2))
    vals = []
    for i in np.arange(-2, 2, 0.1):
        for j in np.arange(-2, 2, 0.1):
            vals.append([i, j])
    z_ = np.asarray(vals)

    feed_dict = {z: z_, keep_prob: 1.0}

    res = sess.run(img, feed_dict)

    show_result(res, d=40)

    print('done')


def extract_2dim_cvae_labels():
    fmeta = '/home/juraj/Desktop/DIPLOMSKI/results/VAE/MNIST/mnist_2dim_cvae/model/model.ckpt.meta'
    f = '/home/juraj/Desktop/DIPLOMSKI/results/VAE/MNIST/mnist_2dim_cvae/model'
    sess = tf.Session()
    saver = tf.train.import_meta_graph(fmeta)
    saver.restore(sess, tf.train.latest_checkpoint(f))

    graph = tf.get_default_graph()
    sample = graph.get_tensor_by_name('encoder/add:0') # (?, 2)
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    cond = graph.get_tensor_by_name('cond:0') # (?, 57)
    z = tf.concat([sample, cond], axis=1, name='z_cond')
    img = graph.get_tensor_by_name('decoder/Reshape_1:0')

    sample_ = np.random.normal(0, 1, (100, 2))
    cond_ = np.zeros((100, 57))
    for i in range(10):
        start = i*10
        cond_[start:start+10, i] = 1.0

    feed_dict = {sample: sample_, cond: cond_, keep_prob: 1.0}

    res = sess.run(img, feed_dict)

    show_result(res, d=10)

    print('done')


def extract_8dim_cvae_labels():
    fmeta = '/home/juraj/Desktop/DIPLOMSKI/results/VAE/MNIST/MNIST_CVAE_results_1/model/model.ckpt.meta'
    f = '/home/juraj/Desktop/DIPLOMSKI/results/VAE/MNIST/MNIST_CVAE_results_1/model'
    sess = tf.Session()
    saver = tf.train.import_meta_graph(fmeta)
    saver.restore(sess, tf.train.latest_checkpoint(f))

    graph = tf.get_default_graph()
    sample = graph.get_tensor_by_name('encoder/add:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    cond = graph.get_tensor_by_name('cond:0')
    img = graph.get_tensor_by_name('decoder/Reshape_1:0')

    sample_ = np.random.normal(0, 1, (100, 8))
    cond_ = np.zeros((100, 57))
    for i in range(10):
        start = i*10
        cond_[start:start+10, i] = 1.0

    feed_dict = {sample: sample_, cond: cond_, keep_prob: 1.0}

    res = sess.run(img, feed_dict)

    show_result(res, d=10)

    print('done')


extract_8dim_cvae_labels()
