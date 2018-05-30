import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import itertools, time


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


def show_result(test_images, path):
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
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


mnist = input_data.read_data_sets('MNIST_data')

tf.reset_default_graph()

batch_size = 64
use_condition = True

X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
X_flat = tf.reshape(X_in, shape=[-1, 28 * 28], name='X_flat')
condition = tf.placeholder(dtype=tf.float32, shape=[None, 57], name='cond')
inp = tf.concat([X_flat, condition], axis=1, name='input') if use_condition else X_flat

Y = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='Y')
Y_flat = tf.reshape(Y, shape=[-1, 28 * 28])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 8

reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49 * dec_in_channels / 2)


def encoder(inp, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        shp = [-1, 29, 29, 1] if use_condition else [-1, 28, 28, 1]
        inp = tf.reshape(inp, shape=shp)
        x = tf.layers.conv2d(inp, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=inputs_decoder, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28])

        return img


def gen_results(i, res_dir = 'results/MNIST_CVAE_results'):
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(25)]
    r = np.random.choice(10)
    f_cond_labels.write('Image' + str(i) + 'Label' + str(r))
    rand_cond = [np.eye(57)[r] for _ in range(25)]
    imgs = sess.run(dec, feed_dict={sampled: randoms, condition: rand_cond, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
    show_result(imgs, res_dir + '/img_' + str(i) + '.jpg')


sampled, mn, sd = encoder(inp, keep_prob)
z = tf.concat([sampled, condition], axis=1, name='z_cond') if use_condition else sampled
dec = decoder(z, keep_prob)

unreshaped = tf.reshape(dec, [-1, 28*28])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
tf.summary.scalar('loss', loss)
tf.summary.scalar('img_loss', tf.reduce_mean(img_loss))
tf.summary.scalar('latent_loss', tf.reduce_mean(latent_loss))

saver = tf.train.Saver()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('summary/summary_cvae_mnist', sess.graph)
sess.run(tf.global_variables_initializer())

N = mnist.train.num_examples
n_batches = N//batch_size
train_epoch = 20
epoch_durations = []
f_cond_labels = open('results/cvae_cond_labels.txt', 'w')
for epoch in range(train_epoch):
    epoch_start = time.time()
    for i in range(n_batches):
        start = time.time()
        curr_batch_no = epoch*n_batches + i
        batch, labels = mnist.train.next_batch(batch_size=batch_size)
        batch = [np.reshape(b, [28, 28]) for b in batch]
        labels = [l for l in np.eye(57)[labels.reshape(-1)]]
        sess.run(optimizer, feed_dict={X_in: batch, condition: labels, Y: batch, keep_prob: 0.8})

        summary, ls, d, i_ls, d_ls, mu, sigm = sess.run([merged, loss, dec, img_loss, latent_loss, mn, sd],
                                                   feed_dict={X_in: batch, condition: labels, Y: batch, keep_prob: 1.0})
        writer.add_summary(summary, curr_batch_no)
        duration = time.time() - start
        print('Epoch', epoch+1, '/', train_epoch,
              'Batch', i+1, '/', n_batches, ':',
              'Loss', ls,
              'Image loss', np.mean(i_ls),
              'Latent loss', np.mean(d_ls),
              'Duration', duration)
        if i % 100 == 0:
            gen_results(str(epoch+1) + '-' + str(i))
    gen_results(str(epoch+1))
    epoch_duration = time.time() - epoch_start
    epoch_durations.append(epoch_duration)

f_cond_labels.close()
print('Epoch durations')
print(epoch_durations)
print("Training finished!")
saver.save(sess, "/home/juraj/Desktop/model/model.ckpt")
sess.close()
