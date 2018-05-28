import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import image_reader


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


tf.reset_default_graph()

batch_size = 64
img_dim = (64, 64, 3)
shape = [None, img_dim[0], img_dim[1], img_dim[2]]

X_in = tf.placeholder(dtype=tf.float32, shape=shape, name='X')
X_flat = tf.reshape(X_in, shape=[-1, img_dim[0]*img_dim[1]*img_dim[2]], name='X_flat')

Y = tf.placeholder(dtype=tf.float32, shape=shape, name='Y')
Y_flat = tf.reshape(Y, shape=[-1, img_dim[0]*img_dim[1]*img_dim[2]])
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

n_latent = 32

reshaped_dim = [-1, 32, 32, 3]


def encoder(inp, keep_prob):
    activation = lrelu
    with tf.variable_scope("encoder", reuse=None):
        shp = [-1, img_dim[0], img_dim[1], img_dim[2]]
        inp = tf.reshape(inp, shape=shp)
        x = tf.layers.conv2d(inp, filters=256, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=1024, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=n_latent, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = x
        # mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def decoder(sampled_z, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        x = tf.layers.dense(sampled_z, units=3072, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=1024, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=512, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=3, kernel_size=4, strides=1, padding='same', activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, img_dim[0], img_dim[1], img_dim[2]])

        return img


def gen_results(name, res_dir='results/food_VAE_results'):
    randoms = [np.random.normal(0, 1, n_latent) for _ in range(25)]
    r = np.random.choice(10)
    rand_cond = [np.eye(57)[r] for _ in range(25)]
    imgs = sess.run(dec, feed_dict={sampled: randoms, keep_prob: 1.0})
    imgs = [np.reshape(imgs[i], [img_dim[0], img_dim[1], img_dim[2]]) for i in range(len(imgs))]
    show_result(imgs, res_dir + '/img_' + name + '.jpg')


sampled, mn, sd = encoder(X_flat, keep_prob)
z = sampled
dec = decoder(z, keep_prob)

unreshaped = tf.reshape(dec, [-1, img_dim[0]*img_dim[1]*img_dim[2]])
img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, Y_flat), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
loss = tf.reduce_mean(img_loss + latent_loss)
optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
tf.summary.scalar('loss', loss)
tf.summary.scalar('img_loss', tf.reduce_mean(img_loss))
tf.summary.scalar('latent_loss', tf.reduce_mean(latent_loss))

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('summary/summary_vae_food', sess.graph)
sess.run(tf.global_variables_initializer())

print('Getting images')
train_set = image_reader.get_images(
    '/home/juraj/Desktop/image_net/image_net_food(64x64)',
    2000,
    shape=(img_dim[0], img_dim[1], 3))
N = len(train_set)
print('Got', N, 'images')
n_batches = N//batch_size
train_epoch = 2
print('Training start')
for epoch in range(train_epoch):
    for i in range(n_batches):
        curr_batch_no = epoch*n_batches + i
        batch = train_set[i*batch_size:(i+1)*batch_size]
        sess.run(optimizer, feed_dict={X_in: batch, Y: batch, keep_prob: 0.8})

        summary, ls, d, i_ls, d_ls, mu, sigm = sess.run([merged, loss, dec, img_loss, latent_loss, mn, sd],
                                                   feed_dict={X_in: batch, Y: batch, keep_prob: 1.0})
        writer.add_summary(summary, curr_batch_no)
        print('Epoch', epoch, '/', train_epoch, 'Batch', i, '/', n_batches, ':', 'Loss', ls, 'Image loss', np.mean(i_ls), 'Latent loss', np.mean(d_ls))
        if i % 100:
            gen_results(str(epoch) + '-' + str(i))
    gen_results(epoch)


