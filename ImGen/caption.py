import os
import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import vgg19
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
import numpy as np


def load_pickle(fname):
    with open(fname + '.pickle', 'rb') as handle:
        a = pickle.load(handle)
    return a


def save_pickle(fname, obj):
    with open(fname + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle)


def get_id_from(fname):
    start_idx = 4
    end_idx = fname.find('.')
    return int(fname[start_idx:end_idx])


class Mapping:
    def __init__(self):
        self.mapping = {}

    @staticmethod
    def get_tags_from(fname):
        tags = []
        with open(fname, 'r') as f:
            for line in f:
                tags.append(line.replace('\n', ''))
        return tags

    @staticmethod
    def get_ids_from(fname):
        ids = []
        with open(fname, 'r') as f:
            for line in f:
                ids.append(int(line.replace('\n', '')))
        return ids

    def append_from_ann(self, fname, ann):
        ids = self.get_ids_from(fname)
        for id in ids:
            self.append2dict(id, [ann])

    def append2dict(self, id, tags):
        if self.mapping.get(id, None) is None:
            self.mapping[id] = tags
        else:
            self.mapping[id] += tags

    def extract_id2tags_mapping(self, tag_dir):
        for fname in os.listdir(tag_dir):
            full_path = tag_dir + '/' + fname
            tags = self.get_tags_from(full_path)
            id = get_id_from(fname)
            self.append2dict(id, tags)
        return self.mapping

    def extract_annotations(self, ann_dir):
        for fname in os.listdir(ann_dir):
            full_path = ann_dir + '/' + fname
            ann = fname[0:fname.find('.')]
            self.append_from_ann(full_path, ann)
        return self.mapping

    def save_pickle(self, f):
        with open(f + '.pickle', 'wb') as handle:
            pickle.dump(self.mapping, handle)


class ImageFeatures:
    def __init__(self):
        self.model = vgg19.VGG19(weights='imagenet')
        self.get_features = K.function([self.model.layers[0].input],
                              [self.model.layers[-2].output])
        self.img_feats = {}

    def img2feats(self, fname):
        original = load_img(fname, target_size=(224, 224))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        processed_image = vgg19.preprocess_input(image_batch.copy())

        return self.get_features([processed_image])[0]

    def imgs2feats(self, dir):
        c = 0
        for fname in os.listdir(dir):
            c += 1
            print(c, '/ 25000')
            full_path = dir + '/' + fname
            id = get_id_from(fname)
            self.img_feats[id] = self.img2feats(full_path)
        return self.img_feats


def generate_mapping():
    tags_dir = '/home/juraj/Desktop/flickr_data/mirflickr/meta/tags'
    ann_dir = '/home/juraj/Desktop/flickr_data/mirflickr25k_annotations_v080'
    mapping = Mapping()
    mapping.extract_annotations(ann_dir)
    mapping.extract_id2tags_mapping(tags_dir)
    mapping.save_pickle('mapping')


if __name__ == '__main__':
    dir = 'imgs'
    save_path = 'img_features'
    img = ImageFeatures()
    res = img.imgs2feats(dir)
    print('Saving results')
    save_pickle(save_path, res)
