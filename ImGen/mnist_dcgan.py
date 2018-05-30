import os, time, itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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


z_size = 8
img_dim = (28, 28)
batch_size = 64
cond_size = 57
train_epoch = 20
lr = 0.0005
reshaped_dim = [-1, 7, 7, 1]
inputs_generator = int(49 * 1 / 2)
use_condition = True


# G(z)
def generator(x, keep_prob, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.layers.dense(x, units=inputs_generator, activation=lrelu)
        x = tf.layers.dense(x, units=inputs_generator * 2 + 1, activation=lrelu)
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        img = tf.reshape(x, shape=[-1, 28, 28, 1])
        # x = tf.contrib.layers.flatten(x)
        # x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid)
        # img = tf.reshape(x, shape=[-1, 28, 28, 1])

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
def discriminator(x, keep_prob, isTrain=True, reuse=False):
    activation = lrelu
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        x = tf.nn.dropout(x, keep_prob)
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


def gen_results(i, res_dir='results/MNIST_CDCGAN_results'):
    z_ = np.random.normal(0, 1, (25, 1, 1, z_size))
    r = np.random.choice(10)
    f_cond_labels.write('Image ' + str(i) + ' Label ' + str(r) + '\n')
    rand_cond = [np.eye(cond_size)[r] for _ in range(25)]
    imgs = sess.run(G_z, {z: z_, isTrain: True, keep_prob: 1.0, condition: rand_cond})
    imgs = [np.reshape(imgs[i], [img_dim[0], img_dim[1]]) for i in range(len(imgs))]
    show_result(imgs, res_dir + '/img_' + str(i) + '.jpg')


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
mnist = input_data.read_data_sets("MNIST_data", reshape=[])

# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_dim[0], img_dim[1], 1), name='x')
x_flat = tf.reshape(x, shape=[-1, img_dim[0]*img_dim[1]], name='x_flat')
condition = tf.placeholder(dtype=tf.float32, shape=[None, cond_size], name='cond')
z = tf.placeholder(tf.float32, shape=(None, 1, 1, z_size))
z_flat = tf.reshape(z, shape=[-1, z_size], name='z_flat')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
isTrain = tf.placeholder(dtype=tf.bool, name='is_train')

# networks : generator
inp_z = tf.concat([z_flat, condition], axis=1, name='inp_z') if use_condition else z_flat
G_z = generator(inp_z, keep_prob, isTrain)
G_z_flat = tf.reshape(G_z, shape=[-1, 28*28], name='g_z_flat')

# networks : discriminator
inp_D_real = tf.reshape(tf.concat([x_flat, condition], axis=1), shape=[-1, img_dim[0]+1, img_dim[1]+1, 1], name='inp_d_real') if use_condition else x
inp_D_fake = tf.reshape(tf.concat([G_z_flat, condition], axis=1), shape=[-1, img_dim[0]+1, img_dim[1]+1, 1], name='inp_d_fake') if use_condition else G_z
D_real, D_real_logits = discriminator(inp_D_real, keep_prob, isTrain)
D_fake, D_fake_logits = discriminator(inp_D_fake, keep_prob, isTrain, reuse=True)

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
saver = tf.train.Saver()
sess = tf.Session()
writer = tf.summary.FileWriter('summary/summary_cdcgan_mnist', sess.graph)
sess.run(tf.global_variables_initializer())

# MNIST resize and normalization
# train_set = tf.image.resize_images(mnist.train.images, [img_dim[0], img_dim[1]]).eval()
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
N = mnist.train.num_examples
n_batches = N//batch_size
epoch_durations = []
f_cond_labels = open('results/cvae_cond_labels.txt', 'w')
for epoch in range(train_epoch):
    epoch_start = time.time()
    for i in range(n_batches):
        start = time.time()
        curr_batch_no = epoch * n_batches + i
        batch, labels = mnist.train.next_batch(batch_size=batch_size)
        # batch = tf.image.resize_images(batch, [img_dim[0], img_dim[1]]).eval()
        batch = np.asarray([np.reshape(b, [28, 28, 1]) for b in batch])
        # batch = (batch - 0.5) / 0.5
        labels = [l for l in np.eye(cond_size)[labels.reshape(-1)]]
        # x_ = train_set[i*batch_size:(i+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_size))

        loss_d_, _, summary = sess.run([D_loss, D_optim, D_loss_summary],
                                       {x: batch, z: z_, isTrain: True, keep_prob: 0.8, condition: labels})
        writer.add_summary(summary, curr_batch_no)

        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_size))
        loss_g_, _, summary = sess.run([G_loss, G_optim, G_loss_summary],
                                       {z: z_, x: batch, isTrain: True, keep_prob: 0.8, condition: labels})
        writer.add_summary(summary, curr_batch_no)
        duration = time.time() - start
        print('Epoch', epoch+1, '/', train_epoch,
              'Batch', i+1, '/', n_batches, ':',
              'D_loss', np.mean(loss_d_),
              'G_loss', np.mean(loss_g_),
              'Duration', duration)
        if i % 100 == 0:
            gen_results(str(epoch+1) + '-' + str(i))
    gen_results(epoch+1)
    epoch_duration = time.time() - epoch_start
    epoch_durations.append(epoch_duration)

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

# TODO zasto su pixelasti rezultati ? jel treba full conv mreza bit da bude okej ?

f_cond_labels.close()
print('Epoch durations')
print(epoch_durations)
print("Training finished!")
saver.save(sess, "/home/juraj/Desktop/model/model.ckpt")
sess.close()
