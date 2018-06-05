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


z_size = 100
img_dim = (64, 64)
batch_size = 100
cond_size = 129
train_epoch = 20
lr = 0.0002
use_condition = False

res_fixed_dir = 'results/fixed'
res_random_dir = 'results/random'


# G(z)
def generator(x, keep_prob, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 1, 1, x.shape[1]])
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)

    return o


# D(x)
def discriminator(x, keep_prob, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)

    return o, conv5


fixed_z_ = np.random.normal(0, 1, (25, 1, 1, z_size))
def gen_results(i, fixed, res_dir):
    z_rand = np.random.normal(0, 1, (25, 1, 1, z_size))
    z_ = fixed_z_ if fixed else z_rand
    r = np.random.choice(10)
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
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))


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
writer = tf.summary.FileWriter('summary', sess.graph)
sess.run(tf.global_variables_initializer())

train_set = tf.image.resize_images(mnist.train.images, [img_dim[0], img_dim[1]]).eval(session=sess)
train_labels = mnist.train.labels
train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
N = mnist.train.num_examples
n_batches = N//batch_size

gen_results('init', True, res_fixed_dir)
gen_results('init', False, res_random_dir)
for epoch in range(train_epoch):
    epoch_start = time.time()
    for i in range(n_batches):
        start = time.time()
        curr_batch_no = epoch * n_batches + i
        batch = train_set[i * batch_size:(i + 1) * batch_size]
        labels = train_labels[i * batch_size:(i + 1) * batch_size]
        # batch, labels = mnist.train.next_batch(batch_size=batch_size)
        # batch = tf.image.resize_images(batch, [img_dim[0], img_dim[1]]).eval()
        # batch = np.asarray([np.reshape(b, [28, 28, 1]) for b in batch])
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
    gen_results(str(epoch+1), True, res_fixed_dir)
    gen_results(str(epoch+1), False, res_random_dir)

# images = []
# for e in range(train_epoch):
#     img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
#     images.append(imageio.imread(img_name))
# imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

saver.save(sess, "model/model.ckpt")
sess.close()

