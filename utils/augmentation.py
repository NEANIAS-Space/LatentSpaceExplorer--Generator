import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


def augmentation(images, flip_x, flip_y, rotate, rotate_degrees, shift, shift_percentage):

    def flip_x_fn(images):
        images = tf.image.random_flip_left_right(images)
        return images

    def flip_y_fn(images):
        images = tf.image.random_flip_up_down(images)
        return images

    def rotate_fn(degrees):
        def rotate(images, degrees=degrees):
            radians = tf.math.multiply(
                degrees,
                tf.math.divide(np.pi, tf.cast(180, dtype=tf.float32))
            )
            radians = tf.math.multiply(
                radians, tf.random.uniform([], -1, 1)
            )
            return tfa.image.rotate(images, radians, interpolation="bilinear")
        return rotate

    def shift_fn(percentage):
        def shift(images, percentage=percentage):
            magnitude = tf.cast(
                tf.math.multiply(
                    images.shape[1],
                    tf.math.divide(percentage, 100)
                ),
                tf.float32
            )
            magnitude = tf.math.floor(
                tf.math.multiply(
                    magnitude,
                    tf.random.uniform([2], -1, 1)
                )
            )
            return tfa.image.translate(images, magnitude)
        return shift

    augmentations = []

    if flip_x:
        augmentations.append(flip_x_fn)
    if flip_y:
        augmentations.append(flip_y_fn)
    if rotate:
        augmentations.append(rotate_fn(rotate_degrees))
    if shift:
        augmentations.append(shift_fn(shift_percentage))

    for a in augmentations:
        images = a(images)

    return images


@tf.function
def tf_augmentation(images, flip_x, flip_y, rotate, rotate_degrees, shift, shift_percentage):
    return tf.py_function(
        augmentation,
        [images, flip_x, flip_y, rotate, rotate_degrees, shift, shift_percentage],
        tf.float32
    )
