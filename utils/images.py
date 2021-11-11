import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def numpy_load(file):
    return np.load(file)


def preprocessing(image, image_dim):

    def replace_nan(image):
        image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
        return image

    def resize(image):
        image = tf.image.resize(image, [image_dim, image_dim])
        return image

    def normalization(image):
        channels = tf.unstack(image, axis=2)
        channels = [ch_normalization(channel) for channel in channels]
        image = tf.stack(channels, axis=2)
        return image

    def ch_normalization(channel):
        max = channel.numpy().max()
        min = channel.numpy().min()

        if (max - min) > 0:
            channel = tf.keras.layers.Lambda(
                lambda ch: (ch - min) / (max - min))(channel)

        return channel

    processes = [replace_nan, resize, normalization]

    for p in processes:
        image = p(image)

    return image


def augmentation(images):

    def flip(images):
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)
        return images

    def rotate(images):
        return tfa.image.rotate(
            images, 45 * tf.random.uniform([], -1, 1)
        )

    def shift(images):
        # 10% of image
        magnitude = tf.cast(tf.math.divide(images.shape[1], 10), tf.float32)
        return tfa.image.translate(
            images, magnitude * tf.random.uniform([2], -1, 1)
        )

    augmentations = [flip, shift, rotate]

    for a in augmentations:
        images = a(images)

    return images


@tf.function
def tf_numpy_load(file):
    return tf.numpy_function(numpy_load, [file], tf.float32)


@tf.function
def tf_preprocessing(image, image_dim):
    return tf.py_function(preprocessing, [image, image_dim], tf.float32)


@tf.function
def tf_augmentation(images):
    return tf.py_function(augmentation, [images], tf.float32)
