import numpy as np
import tensorflow as tf


def numpy_load(file):
    return np.load(file)


def preprocessing(image, image_dim, normalization_type):

    def replace_nan_fn(image):
        image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
        return image

    def resize_fn(image):
        image = tf.image.resize(image, [image_dim, image_dim])
        return image

    def normalization_fn(type):

        def default(image):
            return tf.math.divide(image, 255)

        def multichannel(image):
            def ch_normalization(channel):
                max = channel.numpy().max()
                min = channel.numpy().min()

                if (max - min) > 0:
                    channel = tf.keras.layers.Lambda(
                        lambda ch: (ch - min) / (max - min))(channel)

                return channel

            channels = tf.unstack(image, axis=2)
            channels = [ch_normalization(channel) for channel in channels]
            image = tf.stack(channels, axis=2)
            return image

        def snr(image):
            def ch_normalization(channel):
                channel = channel.numpy()

                mask_low_ids = channel <= 1e-7
                mask_up_ids = channel > 1.0

                min = channel[np.logical_not(mask_low_ids)].min()
                max = channel[np.logical_not(mask_up_ids)].max()

                mask_ids = np.logical_or(
                    np.logical_not(mask_low_ids),
                    np.logical_not(mask_up_ids)
                )

                if (max - min) > 0:
                    channel[mask_ids] = (channel[mask_ids] - min) / (max - min)

                channel[mask_low_ids] = 0.0
                channel[mask_up_ids] = 1.0

                return tf.convert_to_tensor(channel)

            channels = tf.unstack(image, axis=2)
            channels = [ch_normalization(channel) for channel in channels]
            image = tf.stack(channels, axis=2)
            return image

        if type == default.__name__:
            return default

        elif type == multichannel.__name__:
            return multichannel

        elif type == snr.__name__:
            return snr

    processes = [
        replace_nan_fn,
        resize_fn,
        normalization_fn(normalization_type)
    ]

    for p in processes:
        image = p(image)

    return image


@tf.function
def tf_numpy_load(file):
    return tf.numpy_function(numpy_load, [file], tf.float32)


@tf.function
def tf_preprocessing(image, image_dim, normalization_type):
    return tf.py_function(preprocessing, [image, image_dim, normalization_type], tf.float32)
