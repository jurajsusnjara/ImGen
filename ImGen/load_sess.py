import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools


def show_result(test_images, dir='LFW'):
    idx = 0
    for img in test_images:
        path = dir + '/' + str(idx)
        idx += 1
        plt.imshow(img, interpolation='nearest')
        plt.savefig(path)
    plt.close()


fmeta = '/home/juraj/Desktop/DIPLOMSKI/results/img/DCGAN_results_20epoch_13k_lfw_images(128x128)/model/model.ckpt.meta'
f = '/home/juraj/Desktop/DIPLOMSKI/results/img/DCGAN_results_20epoch_13k_lfw_images(128x128)/model/'
sess = tf.Session()
saver = tf.train.import_meta_graph(fmeta)
saver.restore(sess, tf.train.latest_checkpoint(f))

graph = tf.get_default_graph()
z = graph.get_tensor_by_name('Placeholder_1:0')
isTrain = graph.get_tensor_by_name('Placeholder_2:0')
G_z = graph.get_tensor_by_name('generator/Sigmoid:0')
z_ = np.random.normal(0, 1, (10, 1, 1, 100))

feed_dict = {z: z_, isTrain: False}

test_images = sess.run(G_z, feed_dict)

show_result(test_images)

print('done')







