import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os, time, itertools, imageio, pickle


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_set = mnist.train.images
test_images = [train_set[np.random.randint(0, 55000)] for i in range(0, 25)]


def show_result(show=False, path='result.png'):

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

    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

show_result()
