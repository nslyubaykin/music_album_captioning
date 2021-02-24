import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def concat_artist_album(artist_album):
    artist, album = artist_album
    return str(artist) + ' <SEP> ' + str(album)


def parse_image(image_filename, album_name, img_hw=224):
    image = tf.io.read_file(image_filename)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [img_hw, img_hw])
    if tf.shape(image)[-1] == 1:
        image = tf.image.grayscale_to_rgb(image)
    # norm to (-1, 1)
    image = tf.math.subtract(tf.math.multiply(image, 2), 1)
    image = tf.math.minimum(image, tf.ones(tf.shape(image)[-1]))
    image = tf.math.maximum(image, -tf.ones(tf.shape(image)[-1]))
    return (image, album_name[:-1]), album_name[1:]


def show(image, title=None):
    plt.figure()
    plt.imshow((image + 1) / 2)
    plt.axis('off')
    if title is not None:
        plt.title(title)

        
def split_artist_album(gen_in):
    artist, album = '', ''
    if '<SEP>' in gen_in:
        artist, album = gen_in.split('<SEP>')[:2]
    else:
        artist = gen_in
    return artist.strip(), album.strip()
