import os
import pickle
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import vgg19
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
import numpy as np
import gensim
import gensim.downloader as api
import tensorflow as tf
import time
from sklearn.neighbors import NearestNeighbors


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


def generate_mapping():
    tags_dir = '/home/juraj/Desktop/flickr_data/mirflickr/meta/tags'
    ann_dir = '/home/juraj/Desktop/flickr_data/mirflickr25k_annotations_v080'
    mapping = Mapping()
    mapping.extract_annotations(ann_dir)
    mapping.extract_id2tags_mapping(tags_dir)
    mapping.save_pickle('mapping')


def get_vocabulary(map_dict):
    vocabulary = set()
    for _, value in map_dict.items():
        for v in value:
            vocabulary.add(v)
    return list(vocabulary)


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


class WordVectors:
    def __init__(self, fmodel):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(fmodel, binary=True)
        # self.model = api.load("glove-twitter-25")
        self.w2v = {}

    def word2vec(self, word):
        try:
            res = self.model[word]
        except:
            res = []
        return res

    def create_dict(self, vocabulary):
        N = len(vocabulary)
        i = 0
        for word in vocabulary:
            i += 1
            print(i, '/', N)
            self.w2v[word] = self.word2vec(word)
        return self.w2v

    def get_nearest_words(self, vec, n):
        return self.model.most_similar(positive=[vec.reshape(300)], negative=[], topn=n)

    def word_count(self, word):
        return self.model.vocab[word].count


class Dataset:
    def __init__(self, mapping, img_features, word2vec):
        self.mapping = mapping
        self.img_features = img_features
        self.word2vec = word2vec
        self.dataset = []

    def create_dataset(self):
        for id, words in self.mapping.items():
            words = list(set(words))
            features = self.img_features[id]
            vecs = [self.word2vec[word] for word in words]
            for vec in vecs:
                self.dataset.append((features, vec))

    def dataset2file(self, fname):
        with open(fname, 'w') as f:
            for data in self.dataset:
                f.write(str(data[0]))
                f.write('\t')
                f.write(str(data[1]))
                f.write('\n')

    def clean_dataset(self):
        cleaned_dataset = []
        for data in self.dataset:
            if data[1] != []:
                cleaned_dataset.append((data[0], data[1].reshape(1, 300)))
        self.dataset = cleaned_dataset


class CaptionNet:
    def __init__(self, batch_size, lr, dropout, save_model, save_summary):
        self.sess = None
        self.saver = None
        self.writer = None
        self.save_summary = save_summary
        self.save_model = save_model
        self.batch_size = batch_size
        self.dropout = dropout
        self.noise = tf.placeholder(tf.float32, shape=(None, 1, 100), name='noise')
        self.img_features = tf.placeholder(tf.float32, shape=(None, 1, 4096), name='img_features')
        self.word_vec = tf.placeholder(tf.float32, shape=(None, 1, 300), name='word_vec')
        G = self.generator(self.noise, self.img_features)
        D_real, D_real_logits = self.discriminator(
            self.img_features, self.word_vec)
        D_fake, D_fake_logits = self.discriminator(
            self.img_features, G, reuse=True)
        D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_real_logits, labels=tf.ones([batch_size, 1, 1])))
        D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1])))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1])))
        self.D_loss_summary = tf.summary.scalar('D_loss', self.D_loss)
        self.G_loss_summary = tf.summary.scalar('G_loss', self.G_loss)
        T_vars = tf.trainable_variables()
        D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
        G_vars = [var for var in T_vars if var.name.startswith('generator')]
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.D_loss, var_list=D_vars)
            self.G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.G_loss, var_list=G_vars)

    def generator(self, noise, img_features, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            x1 = tf.layers.dense(noise, units=500, activation=tf.nn.relu)
            x1 = tf.nn.dropout(x1, self.dropout)
            x2 = tf.layers.dense(img_features, units=200, activation=tf.nn.relu)
            x2 = tf.nn.dropout(x2, self.dropout)
            x = tf.concat([x1, x2], axis=2)
            x = tf.nn.dropout(x, self.dropout)
            out = tf.layers.dense(x, units=300, name='g_out')
        return out

    def discriminator(self, img_features, word_vec, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            x1 = tf.layers.dense(img_features, units=1200, activation=tf.nn.relu)
            x1 = tf.nn.dropout(x1, self.dropout)
            x2 = tf.layers.dense(word_vec, units=500, activation=tf.nn.relu)
            x2 = tf.nn.dropout(x2, self.dropout)
            x = tf.concat([x1, x2], axis=2)
            # x = tf.contrib.layers.maxout(x, 1000)
            x = tf.layers.dense(x, units=1000)
            x = tf.nn.dropout(x, self.dropout)
            logit = tf.layers.dense(x, units=1)
            out = tf.nn.sigmoid(logit)
        return out, logit

    def init_session(self):
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.writer = tf.summary.FileWriter(self.save_summary, self.sess.graph)
        tf.global_variables_initializer().run()

    def train(self, train_set, epochs):
        N = len(train_set)
        n_batches = N // self.batch_size
        np.random.seed(int(time.time()))
        for epoch in range(epochs):
            for i in range(n_batches):
                start = time.time()
                curr_batch_no = epoch * n_batches + i
                data = train_set[i * self.batch_size:(i + 1) * self.batch_size]
                img_features_data = [el[0] for el in data]
                word_vec_data = [el[1] for el in data]
                noise = np.random.normal(0, 1, (self.batch_size, 1, 100))
                loss_d_, _, summary = self.sess.run([self.D_loss, self.D_optim, self.D_loss_summary],
                                      {self.img_features: img_features_data, self.word_vec: word_vec_data, self.noise: noise})
                self.writer.add_summary(summary, curr_batch_no)
                noise = np.random.normal(0, 1, (self.batch_size, 1, 100))
                loss_g_, _, summary = self.sess.run([self.G_loss, self.G_optim, self.G_loss_summary],
                                      {self.img_features: img_features_data, self.word_vec: word_vec_data, self.noise: noise})
                self.writer.add_summary(summary, curr_batch_no)
                duration = time.time() - start
                print('Epoch', epoch + 1, '/', epochs,
                      'Batch', i + 1, '/', n_batches, ':',
                      'D_loss', np.mean(loss_d_),
                      'G_loss', np.mean(loss_g_),
                      'Duration', duration)

    def end_session(self):
        self.saver.save(self.sess, self.save_model)
        self.sess.close()


class NearestWordVectors:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.words = []
        self.vectors = []
        self.extract_from_dict()
        self.nbrs = NearestNeighbors(algorithm='brute', metric='cosine').fit(self.vectors)

    def extract_from_dict(self):
        for word, vec in self.word2vec.items():
            self.words.append(word)
            self.vectors.append(vec.reshape(3))

    def get_n_nearest(self, n, query):
        distances, indices = self.nbrs.kneighbors(query, n_neighbors=n)
        return [self.words[i] for i in indices[0]]


class Evaluation:
    def __init__(self, model_meta, model_dir, wv: WordVectors, img_f: ImageFeatures):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(model_meta)
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))
        graph = tf.get_default_graph()
        self.noise = graph.get_tensor_by_name('noise:0')
        self.img_features = graph.get_tensor_by_name('img_features:0')
        self.vector = graph.get_tensor_by_name('generator/g_out/BiasAdd:0')
        self.wv = wv
        self.img_f = img_f

    def generate_samples(self, n, img_path):
        samples = []

        img_feature = self.img_f.img2feats(img_path)
        noise = np.random.normal(0, 1, (n, 1, 100))
        img_repeats = np.repeat(img_feature, n, axis=0).reshape(n, 1, 4096)
        feed_dict = {self.noise: noise, self.img_features: img_repeats}
        result = self.sess.run(self.vector, feed_dict=feed_dict)
        for r in result:
            samples.append(r)
        return samples

    def get_closest_words(self, samples, n, k):
        closest_words = set()
        for sample in samples:
            nearest_words = self.wv.get_nearest_words(sample, n)
            for t in nearest_words:
                closest_words.add(t[0])
        return self.select_most_common(k, closest_words)

    def select_most_common(self, n, words):
        word_freq = []
        for word in words:
            count = self.wv.word_count(word)
            word_freq.append((word, count))
        word_freq.sort(key=lambda tup: tup[1], reverse=True)
        return word_freq[:n]


def train_caption_net():
    mapping = load_pickle('caption_data/mapping')
    img_features = load_pickle('caption_data/img_features')
    word2vec = load_pickle('caption_data/word2vec')
    dataset = Dataset(mapping, img_features, word2vec)
    dataset.create_dataset()
    dataset.clean_dataset()
    lr = 0.0005
    batch_size = 100
    dropout = 0.5
    epochs = 20
    net = CaptionNet(batch_size, lr, dropout, 'model/model.ckpt', 'summary')
    net.init_session()
    net.train(dataset.dataset, epochs)
    net.end_session()


def define_nearest_model():
    d = {'a': np.array([[0, 0, 0]]),
         'b': np.array([[1, 0, 0]]),
         'c': np.array([[1, 1, 0]]),
         'd': np.array([[3, 5, 3]]),
         'e': np.array([[5, 4, 5]])}
    query = np.array([[2, 2, 2]])
    nv = NearestWordVectors(d)
    res = nv.get_n_nearest(3, query)
    print(res)


def words_frequency():
    fname = 'caption_data/mapping'
    mapping = load_pickle(fname)
    word_freq = {}
    for id, words in mapping.items():
        for word in words:
            if word_freq.get(word, None) is None:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    print(len(word_freq))


def test_gmodel():
    word = 'pigs'
    wv = WordVectors('GoogleNews-vectors-negative300.bin')
    nearest = wv.get_nearest_words(wv.word2vec(word), 10)
    print(nearest)


def evaluation():
    img_path = 'apples.jpg'
    print('Loading w2v model')
    wv = WordVectors('GoogleNews-vectors-negative300.bin')
    # wv = None
    print('Loading img features')
    img_feats = ImageFeatures()
    eval = Evaluation('model/model.ckpt.meta', 'model', wv, img_feats)
    print('Generating samples')
    samples = eval.generate_samples(100, img_path)
    print('Getting closest words')
    closest = eval.get_closest_words(samples, 20, 10)
    print(closest)


if __name__ == '__main__':
    train_caption_net()


# TODO reshapeat ulaze u (4096,) i (300,) ?
