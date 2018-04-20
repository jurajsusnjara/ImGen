import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


num_channels = 3
img_size = 32


def plot_img(img):
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')
    plt.show()


def get_class_images(selected_cls, dir='cifar-10'):
    images = []
    classes = []
    for fname in os.listdir(dir):
        fname = dir + '/' + fname
        with open(fname, mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
        raw_images = data[b'data']
        cls = np.array(data[b'labels'])
        raw_float = np.array(raw_images, dtype=float) / 255.0
        fimages = raw_float.reshape([-1, num_channels, img_size, img_size])
        fimages = fimages.transpose([0, 2, 3, 1])
        images.append(fimages)
        classes.append(cls)

    images = np.concatenate(images, axis=0)
    classes = np.concatenate(classes, axis=0)

    selected_images = []
    for img, cls in zip(images, classes):
        if cls == selected_cls:
            selected_images.append(img)

    return selected_images
