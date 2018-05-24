import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import config as cfg


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)


# Read config
cfg.parse_config('config.cfg')
batch_size = int(cfg.config['dcgan']['batch_size'])
lr = float(cfg.config['dcgan']['lr'])
train_epoch = int(cfg.config['dcgan']['train_epoch'])
k = int(cfg.config['dcgan']['kernel'])
kernel = [k, k]
g_depths = [int(el) for el in cfg.config['dcgan']['g_depths'].split(',')]
d_depths = [int(el) for el in cfg.config['dcgan']['d_depths'].split(',')]
batch_norm = True if cfg.config['dcgan']['batch_norm'] == 'true' else False
g_strides = [int(el) for el in cfg.config['dcgan']['g_strides'].split(',')]
d_strides = [int(el) for el in cfg.config['dcgan']['d_strides'].split(',')]
z_size = int(cfg.config['dcgan']['z_size'])
img_dim_str = cfg.config['dcgan']['img_dim'].split('x')
img_dim = (int(img_dim_str[0]), int(img_dim_str[1]))

use_condition = False


# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='valid', activation=lrelu)
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=lrelu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.tanh)
        img = tf.reshape(x, shape=[-1, 28, 28, 1])

        # idx = 0
        # inp = x
        # for depth, stride in zip(g_depths[:-1],g_strides[:-1]):
        #     padding, strides = ('valid', (1, 1)) if i dx == 0 else ('same', (stride, stride))
        #     conv = tf.layers.conv2d_transpose(inp, depth, kernel, strides=strides, padding=padding)
        #     temp = tf.layers.batch_normalization(conv, training=isTrain) if batch_norm else conv
        #     inp = lrelu(temp, 0.2)
        #     idx += 1
        # padding, strides = ('same', (g_strides[-1], g_strides[-1]))
        # conv = tf.layers.conv2d_transpose(inp, g_depths[-1], kernel, strides=strides, padding=padding)
        # o = tf.nn.tanh(conv)

        return img


# D(x)
def discriminator(x, isTrain=True, reuse=False):
    activation = lrelu
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.layers.batch_normalization(x, training=isTrain)
        x = tf.contrib.layers.flatten(x)
        logit = tf.layers.dense(x, units=1)
        o = tf.nn.sigmoid(logit)
        # idx = 0
        # inp = x
        # for depth, stride in zip(d_depths[:-1], d_strides[:-1]):
        #     conv = tf.layers.conv2d(inp, depth, kernel, strides=(stride, stride), padding='same')
        #     inp = lrelu(conv, 0.2) if idx == 0 else lrelu(tf.layers.batch_normalization(conv, training=isTrain), 0.2)
        #     idx += 1
        # conv = tf.layers.conv2d(inp, d_depths[-1], kernel, strides=(1, 1), padding='valid')
        # o = tf.nn.sigmoid(conv)

        return o, logit


fixed_z_ = np.random.normal(0, 1, (25, 1, 1, z_size))
def show_result(num_epoch, show=False, save=False, path='result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, isTrain: False})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# variables : input
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, z_size))
isTrain = tf.placeholder(dtype=tf.bool)

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# loss for each network
# cross entropy with logits se koristi radi numericke stabilnosti + kad gledamo to u odnosu na labele jedinice ili nule
# onda se dobije prakticki ista stvar ko u mnist_gan u izrazima: (1 - nest) ili (nesto)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1])))
D_loss_summary = tf.summary.scalar('D_loss', D_loss)
G_loss_summary = tf.summary.scalar('G_loss', G_loss)

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter('summary/summary_dcgan_mnist', sess.graph)
tf.global_variables_initializer().run()

# MNIST resize and normalization
# train_set = tf.image.resize_images(mnist.train.images, [img_dim[0], img_dim[1]]).eval()
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# results save folder
root = 'results/MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
N = mnist.train.num_examples
n_batches = N//batch_size
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for i in range(n_batches):
        curr_batch_no = epoch * n_batches + i
        # update discriminator
        batch, _ = mnist.train.next_batch(batch_size=batch_size)
        # batch = tf.image.resize_images(batch, [img_dim[0], img_dim[1]]).eval()
        x_ = (batch - 0.5) / 0.5
        # x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_size))

        loss_d_, _, summary = sess.run([D_loss, D_optim, D_loss_summary], {x: x_, z: z_, isTrain: True})
        D_losses.append(loss_d_)
        writer.add_summary(summary, curr_batch_no)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_size))
        loss_g_, _, summary = sess.run([G_loss, G_optim, G_loss_summary], {z: z_, x: x_, isTrain: True})
        writer.add_summary(summary, curr_batch_no)
        G_losses.append(loss_g_)
        print('[%d/%d] [%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, (i+1), n_batches, np.mean(D_losses), np.mean(G_losses)))
        if i % 100 == 0:
            fixed_p = root + 'Fixed_results/' + model + str(epoch) + '_' + str(i) + '.png'
            show_result((epoch + 1), save=True, path=fixed_p)
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()
