import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import config as cfg
import image_reader


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

res_fixed_dir = 'results/fixed'
res_random_dir = 'results/random'


# G(z)
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        idx = 0
        inp = x
        for depth, stride in zip(g_depths[:-1], g_strides[:-1]):
            padding = 'valid' if idx == 0 else 'same'
            conv = tf.layers.conv2d_transpose(inp, depth, kernel, strides=(stride, stride), padding=padding)
            temp = tf.layers.batch_normalization(conv, training=isTrain) if batch_norm else conv
            inp = lrelu(temp, 0.2)
            idx += 1
        padding = 'same'
        stride = g_strides[-1]
        conv = tf.layers.conv2d_transpose(inp, g_depths[-1], kernel, strides=(stride, stride), padding=padding)
        o = tf.nn.sigmoid(conv)

        return o


# D(x)
def discriminator(x, isTrain=True, reuse=False, return_layer=-1):
    with tf.variable_scope('discriminator', reuse=reuse):
        idx = 0
        inp = x
        for depth, stride in zip(d_depths[:-1], d_strides[:-1]):
            conv = tf.layers.conv2d(inp, depth, kernel, strides=(stride, stride), padding='same')
            inp = lrelu(conv, 0.2) if idx == 0 else lrelu(tf.layers.batch_normalization(conv, training=isTrain), 0.2)
            if idx == return_layer:
                return inp, conv
            idx += 1
        conv = tf.layers.conv2d(inp, d_depths[-1], kernel, strides=(d_strides[-1], d_strides[-1]), padding='valid')
        o = tf.nn.sigmoid(conv)

        return o, conv


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


fixed_z_ = np.random.normal(0, 1, (25, 1, 1, z_size))
def gen_results(i, fixed, res_dir):
    z_ = np.random.normal(0, 1, (25, 1, 1, z_size))
    imgs = sess.run(G_z, {z: fixed_z_, isTrain: False}) if fixed else sess.run(G_z, {z: z_, isTrain: False})
    imgs = [np.reshape(imgs[i], [img_dim[0], img_dim[1], 3]) for i in range(len(imgs))]
    show_result(imgs, res_dir + '/img_' + str(i) + '.jpg')


# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_dim[0], img_dim[1], 3))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, z_size))
isTrain = tf.placeholder(dtype=tf.bool, name='is_train')

# networks : generator
G_z = generator(z, isTrain)

# networks : discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# discriminator activations
D_real_act, D_real_act_ = discriminator(x, isTrain, reuse=True)
D_fake_act, D_fake_act_ = discriminator(G_z, isTrain, reuse=True)


# loss for each network
# cross entropy with logits se koristi radi numericke stabilnosti + kad gledamo to u odnosu na labele jedinice ili nule
# onda se dobije prakticki ista stvar ko u mnist_gan u izrazima: (1 - nest) ili (nesto)
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))
G_loss = tf.norm(D_real_act - D_fake_act)

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
saver = tf.train.Saver()
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter('summary', sess.graph)
tf.global_variables_initializer().run()

train_set = image_reader.get_cifar10_images()

# training-loop
N = len(train_set)
print('Got', N, 'images')
n_batches = N//batch_size
np.random.seed(int(time.time()))
print('training start!')

gen_results('init', True, res_fixed_dir)
gen_results('init', False, res_random_dir)
for epoch in range(train_epoch):
    epoch_start = time.time()
    for i in range(n_batches):
        start = time.time()
        curr_batch_no = epoch * n_batches + i
        # update discriminator
        x_ = train_set[i*batch_size:(i+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_size))

        loss_d_, _, summary = sess.run([D_loss, D_optim, D_loss_summary], {x: x_, z: z_, isTrain: True})
        writer.add_summary(summary, curr_batch_no)

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_size))
        loss_g_, _, summary = sess.run([G_loss, G_optim, G_loss_summary], {z: z_, x: x_, isTrain: True})
        writer.add_summary(summary, curr_batch_no)
        duration = time.time() - start
        print('Epoch', epoch + 1, '/', train_epoch,
              'Batch', i + 1, '/', n_batches, ':',
              'D_loss', np.mean(loss_d_),
              'G_loss', np.mean(loss_g_),
              'Duration', duration)
    gen_results(str(epoch+1), True, res_fixed_dir)
    gen_results(str(epoch+1), False, res_random_dir)

saver.save(sess, "model/model.ckpt")
sess.close()

