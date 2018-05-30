import numpy as np
import gensim
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications import vgg19
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def vgg_test():
    vgg_model = vgg19.VGG19(weights='imagenet')
    get_features = K.function([vgg_model.layers[0].input],
                              [vgg_model.layers[-2].output])

    fname = 'dog.jpg'
    original = load_img(fname, target_size=(224, 224))
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)

    processed_image = vgg19.preprocess_input(image_batch.copy())
    fts1 = get_features([image_batch])[0]
    fts2 = get_features([processed_image])[0]
    predictions = vgg_model.predict(processed_image)
    label = decode_predictions(predictions)
    print(label)

    print('DONE')


def words_test():
    fmodel = '/home/juraj/Desktop/GoogleNews-vectors-negative300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(fmodel, binary=True)
    dog = model['dog']
    print(dog.shape)
    print(dog[:10])
    print(model.most_similar(positive=['woman', 'king'], negative=['man']))
    print(model.similarity('woman', 'man'))



