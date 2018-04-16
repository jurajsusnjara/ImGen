import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import config as cfg


# Read config
cfg.parse_config('config.cfg')
batch_size = int(cfg.config['gan']['batch_size'])
lr = float(cfg.config['gan']['lr'])
train_epoch = int(cfg.config['gan']['train_epoch'])
g_layers = [int(el) for el in cfg.config['gan']['g_layers'].split(',')]
d_layers = [int(el) for el in cfg.config['gan']['d_layers'].split(',')]
z_size = int(cfg.config['gan']['z_size'])
drop_out_p = float(cfg.config['gan']['drop_out'])


# G(z)
def generator(x, layers):
    # initializers
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    # Hidden layers
    idx = 0
    h = x
    for layer in layers:
        w = tf.get_variable('G_w' + str(idx), [h.get_shape()[1], layer], initializer=w_init)
        b = tf.get_variable('G_b' + str(idx), [layer], initializer=b_init)
        if idx == len(layers)-1:
            h = tf.nn.tanh(tf.matmul(h, w) + b)
        else:
            h = tf.nn.relu(tf.matmul(h, w) + b)
        idx += 1

    return h


# D(x)
def discriminator(x, drop_out, layers):
    w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
    b_init = tf.constant_initializer(0.)

    idx = 0
    h = x
    for layer in layers:
        w = tf.get_variable('D_w' + str(idx), [h.get_shape()[1], layer], initializer=w_init)
        b = tf.get_variable('D_b' + str(idx), [layer], initializer=b_init)
        if idx == len(layers)-1:
            h = tf.sigmoid(tf.matmul(h, w) + b)
        else:
            h = tf.nn.relu(tf.matmul(h, w) + b)
            h = tf.nn.dropout(h, drop_out)
        idx += 1

    return h


fixed_z_ = np.random.normal(0, 1, (25, z_size))
def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = np.random.normal(0, 1, (25, z_size))

    if isFix:
        test_images = sess.run(G_z, {z: fixed_z_, drop_out: 0.0})
    else:
        test_images = sess.run(G_z, {z: z_, drop_out: 0.0})

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (28, 28)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
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
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = (mnist.train.images - 0.5) / 0.5  # normalization; range: -1 ~ 1
# TODO [-1,1] vs [0,1] ?
# train_set = mnist.train.images

# networks : generator[1024, 512, 256, 1]
with tf.variable_scope('G'):
    z = tf.placeholder(tf.float32, shape=(None, z_size))
    G_z = generator(z, g_layers)

# networks : discriminator
with tf.variable_scope('D') as scope:
    drop_out = tf.placeholder(dtype=tf.float32, name='drop_out')
    x = tf.placeholder(tf.float32, shape=(None, 784))
    D_real = discriminator(x, drop_out, d_layers)
    scope.reuse_variables()
    D_fake = discriminator(G_z, drop_out, d_layers)


# loss for each network
eps = 1e-2
D_loss = tf.reduce_mean(-tf.log(D_real + eps) - tf.log(1 - D_fake + eps))
G_loss = tf.reduce_mean(-tf.log(D_fake + eps))

# trainable variables for each network
t_vars = tf.trainable_variables()
D_vars = [var for var in t_vars if 'D_' in var.name]
G_vars = [var for var in t_vars if 'G_' in var.name]

# optimizer for each network
D_optim = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=D_vars)
G_optim = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=G_vars)

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# results save folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/Random_results'):
    os.mkdir('MNIST_GAN_results/Random_results')
if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
    os.mkdir('MNIST_GAN_results/Fixed_results')

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
start_time = time.time()
for epoch in range(train_epoch):
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    for iter in range(train_set.shape[0] // batch_size):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, z_size))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, drop_out: drop_out_p})
        D_losses.append(loss_d_)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, z_size))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, drop_out: drop_out_p})
        G_losses.append(loss_g_)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=p, isFix=False)
    show_result((epoch + 1), save=True, path=fixed_p, isFix=True)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

images = []
for e in range(train_epoch):
    img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

sess.close()
