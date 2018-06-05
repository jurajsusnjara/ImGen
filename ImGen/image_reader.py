import scipy.ndimage as scpimg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import urllib.request
from io import BytesIO
from multiprocessing import Pool
import pickle
import os


def plot_img(img):
    plt.imshow(img, interpolation='nearest')
    plt.axis('off')
    plt.show()


def get_cifar10_images(selected_cls=-1, dir='cifar-10'):
    images = []
    classes = []
    for fname in os.listdir(dir):
        fname = dir + '/' + fname
        with open(fname, mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
        raw_images = data[b'data']
        cls = np.array(data[b'labels'])
        raw_float = np.array(raw_images, dtype=float) / 255.0
        fimages = raw_float.reshape([-1, 3, 32, 32])
        fimages = fimages.transpose([0, 2, 3, 1])
        images.append(fimages)
        classes.append(cls)

    images = np.concatenate(images, axis=0)
    classes = np.concatenate(classes, axis=0)

    if selected_cls == -1:
        return images

    selected_images = []
    for img, cls in zip(images, classes):
        if cls == selected_cls:
            selected_images.append(img)

    return selected_images


def load_img(fname):
    img = scpimg.imread(fname)
    return np.array(img, dtype=float) / 255.0


def get_images(dir, n, shape=(128, 128, 3)):
    imgs = []
    idx = 0
    for fname in os.listdir(dir):
        if idx == n:
            break
        idx += 1
        img = None
        try:
            img = load_img(dir + '/' + fname)
        except:
            print('Cannot load:', fname)
        if img is not None and img.shape == shape:
            imgs.append(img)
    return imgs


def download_img(url):
    img = None
    try:
        req = urllib.request.urlopen(url)
        if req.url != url:
            raise Exception('Non existent image')
        img = Image.open(BytesIO(req.read()))
    except:
        print('Cannot download image:', url)
    return img


def resize_and_save(img, dim, fname, format):
    img = img.resize(dim)
    img.save(fname, format, optimize=True)


def get_random_urls(fname, n):
    urls = []
    with open(fname, 'r') as f:
        for line in f:
            url = line.split()[1]
            if url.endswith('.jpg') or url.endswith('.jpeg'):
                urls.append(url)
            if len(urls) >= 1000000:
                break
    if n == -1:
        return urls
    else:
        return np.random.choice(urls, n, replace=False).tolist()


def get_category_urls(fname, category):
    urls = []
    with open(fname, 'r') as f:
        for line in f:
            split = line.split()
            current_category = split[0]
            url = split[1]
            if current_category == category:
                urls.append(url)
    return urls


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def execute_url(url, idx):
    print('Downloading image', idx, ':', url)
    img = download_img(url)
    if img is not None:
        print('Saving image', idx, ':', url)
        resize_and_save(img, dim, out_dir + '/image_' + str(idx) + '.jpg', 'jpeg')
    idx += 1


def resize_lfw(dir, out_dir, dim=(128, 128), fmt='jpeg'):
    for root, subFolders, files in os.walk(dir):
        for f in files:
            print(f)
            fname = root + '/' + f
            img = Image.open(fname)
            resize_and_save(img, dim, out_dir + '/' + f, fmt)


if __name__ == '__main__':
    # resize_lfw('/home/juraj/Desktop/LFW/lfw-deepfunneled', '/home/juraj/Desktop/LFW/lfw-deepfunneled(64x64)', dim=(64,64))
    inp_file = '/home/juraj/Desktop/food.txt'
    out_dir = '/home/juraj/Desktop/image_net_food(64x64)'
    n = -1
    dim = (64, 64)
    p = Pool(32)

    print('Getting urls')
    urls = get_random_urls(inp_file, n)
    p.starmap(execute_url, zip(urls, range(len(urls))))
    print('Done')
