import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import image_reader
import time


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


res_fixed_dir = 'results/fixed'
res_random_dir = 'results/random'
tf.reset_default_graph()

batch_size = 100
img_dim = (32, 32, 3)
shape = [None, img_dim[0], img_dim[1], img_dim[2]]

X_in = tf.placeholder(dtype=tf.float32, shape=shape, name='X')
Y = tf.placeholder(dtype=tf.float32, shape=shape, name='Y')
Y_flat = tf.reshape(Y, shape=[-1, img_dim[0]*img_dim[1]*img_dim[2]])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')
isTrain = tf.placeholder(dtype=tf.bool, name='is_train')

n_latent = 100


def encoder(inp, keep_prob, isTrain):
    with tf.variable_scope("encoder", reuse=None):
        x = tf.layers.conv2d(inp, filters=128, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=1024, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d(x, filters=n_latent, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.contrib.layers.flatten(x)
        mn = x
        # mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * x
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob, isTrain):
    with tf.variable_scope("decoder", reuse=None):
        # x = tf.layers.dense(sampled_z, units=3072, activation=lrelu)
        x = tf.reshape(sampled_z, [-1, 1, 1, n_latent])
        x = tf.layers.conv2d_transpose(x, filters=1024, kernel_size=4, strides=1, padding='valid')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same')
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = lrelu(x)
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=4, strides=2, padding='same')

        return x


fixed_randoms = [np.random.normal(0, 1, n_latent) for _ in range(25)]
def gen_results(name, fixed, res_dir):
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(25)]
    imgs = sess.run(dec, feed_dict={sampled: fixed_randoms, keep_prob: 1.0, isTrain: False}) if fixed else sess.run(dec, feed_dict={sampled: randoms, keep_prob: 1.0, isTrain: False})
    imgs = [np.reshape(imgs[i], [img_dim[0], img_dim[1], img_dim[2]]) for i in range(len(imgs))]
    show_result(imgs, res_dir + '/img_' + name + '.jpg')


sampled, mn, sd = encoder(X_in, keep_prob, isTrain)
z = sampled
dec = decoder(z, keep_prob, isTrain)

unreshaped = tf.reshape(dec, [-1, img_dim[0]*img_dim[1]*img_dim[2]])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0002).minimize(loss)
tf.summary.scalar('loss', loss)
tf.summary.scalar('img_loss', tf.reduce_mean(img_loss))
tf.summary.scalar('latent_loss', tf.reduce_mean(latent_loss))

saver = tf.train.Saver()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('summary', sess.graph)
sess.run(tf.global_variables_initializer())

print('Getting images')
train_set = image_reader.get_cifar10_images()
N = len(train_set)
print('Got', N, 'images')
n_batches = N//batch_size
np.random.seed(int(time.time()))
train_epoch = 50
epoch_durations = []
print('Training start')

gen_results('init', True, res_fixed_dir)
gen_results('init', False, res_random_dir)
for epoch in range(train_epoch):
    epoch_start = time.time()
    for i in range(n_batches):
        start = time.time()
        curr_batch_no = epoch*n_batches + i
        batch = train_set[i*batch_size:(i+1)*batch_size]
        sess.run(optimizer, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8, isTrain: True})

        summary, ls, d, i_ls, d_ls, mu, sigm = sess.run([merged, loss, dec, img_loss, latent_loss, mn, sd],
                                                   feed_dict={X_in: batch, Y: batch, keep_prob: 1.0, isTrain: True})
        writer.add_summary(summary, curr_batch_no)
        duration = time.time() - start
        print('Epoch', epoch+1, '/', train_epoch,
              'Batch', i+1, '/', n_batches, ':',
              'Loss', ls,
              'Image loss', np.mean(i_ls),
              'Latent loss', np.mean(d_ls),
              'Duration', duration)
    gen_results(str(epoch+1), True, res_fixed_dir)
    gen_results(str(epoch+1), False, res_random_dir)

saver.save(sess, 'model/model.ckpt')
sess.close()
