import tensorflow as tf
from tensorflow.keras import Model, layers, activations


class DownSampling(layers.Layer):

    def __init__(self, name, filter):
        super(DownSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.act = layers.LeakyReLU()

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.act(x)

        return x


class UpSampling(layers.Layer):

    def __init__(self, name, filter):
        super(UpSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2DTranspose(
            filter, (3, 3), strides=1, padding='same')
        self.act = layers.LeakyReLU()

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.act(x)

        return x


class Generator():

    def __init__(self):
        pass


class Discriminator():

    def __init__(self, filters):
        super(Discriminator, self).__init__()
        self._name = 'discriminator'

        self.downsamplings = []
        for id, filter in enumerate(filters):
            self.downsamplings.append(
                DownSampling(f'down_sampling_{id}', filter)
            )

        self.flat = layers.Flatten()
        self.drop = layers.Dropout(0.2)

        self.dense = layers.Dense(1)
        self.act = layers.Activation(activations.sigmoid)


class DeepConvolutionalGenerativeAdversarialNetwork():

    def __init__(self):

        filters = [64, 128, 128]
        pass
