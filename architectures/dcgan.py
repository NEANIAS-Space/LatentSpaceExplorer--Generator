import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, activations


class DownSampling(layers.Layer):

    def __init__(self, name, filter):
        super(DownSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2D(filter, (3, 3), strides=2, padding='same')
        self.act = layers.LeakyReLU()
        self.drop = layers.Dropout(0.3)

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.act(x)
        x = self.drop(x)

        return x


class UpSampling(layers.Layer):

    def __init__(self, name, filter):
        super(UpSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2DTranspose(
            filter, (3, 3), strides=2, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU()

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)

        return x


class Discriminator(Model):

    def __init__(self, filters):
        super(Discriminator, self).__init__()
        self._name = 'discriminator'

        self.downsamplings = []
        for id, filter in enumerate(filters):
            self.downsamplings.append(
                DownSampling(f'down_sampling_{id}', filter)
            )

        self.flat = layers.Flatten()
        self.dense = layers.Dense(units=1)

    def call(self, inputs):

        x = self.downsamplings[0](inputs)

        for ds in self.downsamplings[1:]:
            x = ds(x)

        x = self.flat(x)
        output = self.dense(x)

        return output


class Generator(Model):

    def __init__(self, flat_dim, pre_flat_shape, filters, channels_num):
        super(Generator, self).__init__()
        self._name = 'generator'

        self.dense = layers.Dense(units=flat_dim)
        self.bn = layers.BatchNormalization()
        self.act1 = layers.LeakyReLU()

        self.reshape = layers.Reshape(pre_flat_shape)

        self.upsamplings = []
        for id, filter in enumerate(filters):
            self.upsamplings.append(
                UpSampling(f'up_sampling_{id}', filter)
            )

        self.conv = layers.Conv2DTranspose(
            channels_num, (3, 3), strides=1, padding='same')
        self.act2 = layers.Activation(activations.tanh)

    def call(self, inputs):

        x = self.dense(inputs)
        x = self.bn(x)
        x = self.act1(x)

        x = self.reshape(x)

        for us in self.upsamplings:
            x = us(x)

        x = self.conv(x)
        output = self.act2(x)

        return output


class DCGAN(Model):

    def __init__(self, image_dim, channels_num, latent_dim, filters):
        super(DCGAN, self).__init__()
        self._name = 'dcgan'
        self.channels_num = channels_num
        self.latent_dim = latent_dim

        discriminator_input_shape = (None, image_dim, image_dim, channels_num)
        generator_input_shape = (None, latent_dim)

        self.discriminator = Discriminator(filters)
        self.discriminator.build(input_shape=discriminator_input_shape)
        self.discriminator.call(layers.Input(
            shape=discriminator_input_shape[1:]))
        self.discriminator.summary()

        flat_dim = self.discriminator.flat.output_shape[1]
        pre_flat_shape = self.discriminator.flat.input_shape[1:]

        self.generator = Generator(
            flat_dim, pre_flat_shape, filters[::-1], channels_num)
        self.generator.build(input_shape=generator_input_shape)
        self.generator.call(layers.Input(shape=generator_input_shape[1:]))
        self.generator.summary()

        self.build(input_shape=generator_input_shape)
        self.call(layers.Input(shape=generator_input_shape[1:]))
        self.summary()

    def compile(self, optimizer, learning_rate):
        super(DCGAN, self).compile()

        self.discriminator_optimizer = tf.keras.optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })
        self.generator_optimizer = tf.keras.optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })

        self.loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=0.2)

        self.training_tracker_discriminator_loss = tf.keras.metrics.Mean()
        self.training_tracker_generator_loss = tf.keras.metrics.Mean()
        self.test_tracker_discriminator_loss = tf.keras.metrics.Mean()
        self.test_tracker_generator_loss = tf.keras.metrics.Mean()

    @property
    def metrics(self):
        return [
            self.training_tracker_discriminator_loss,
            self.training_tracker_generator_loss,
            self.test_tracker_discriminator_loss,
            self.test_tracker_generator_loss
        ]

    def call(self, inputs, training=True):
        fake = self.generator(inputs)

        if not training:
            return fake

        judgment = self.discriminator(fake)

        return judgment

    @tf.function
    def train_step(self, epoch, train_batch):
        label_real = tf.zeros([train_batch.shape[0]])
        label_real += 0.05 * tf.random.uniform(tf.shape(label_real))

        label_fake = tf.ones([train_batch.shape[0]])
        label_fake += 0.05 * tf.random.uniform(tf.shape(label_fake))

        noise = tf.random.normal([train_batch.shape[0], self.latent_dim])
        decay_noise = self.decay_function(epoch)

        real = train_batch
        real = self.add_gaussian_noise(real, std=decay_noise)

        fake = self.generator(noise)
        fake = self.add_gaussian_noise(fake, std=decay_noise)

        # Discriminator
        with tf.GradientTape() as tape:
            output_real = self.discriminator(real)
            output_fake = self.discriminator(fake)

            loss_real = self.loss(label_real, output_real)
            loss_fake = self.loss(label_fake, output_fake)
            loss_discriminator = loss_real + loss_fake

        grads = tape.gradient(
            loss_discriminator, self.discriminator.trainable_weights)

        self.discriminator_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights))

        self.training_tracker_discriminator_loss.update_state(
            loss_discriminator)

        # Generator
        with tf.GradientTape() as tape:
            output_fake = self.discriminator(
                self.add_gaussian_noise(
                    self.generator(noise), std=decay_noise)
            )

            loss_generator = self.loss(label_real, output_fake)

        grads_generator = tape.gradient(
            loss_generator, self.generator.trainable_weights)

        self.generator_optimizer.apply_gradients(
            zip(grads_generator, self.generator.trainable_weights))

        self.training_tracker_generator_loss.update_state(loss_generator)

        return {
            "discriminator_loss": self.training_tracker_discriminator_loss.result(),
            "generator_loss": self.training_tracker_generator_loss.result()
        }

    @tf.function
    def test_step(self, epoch, test_batch):
        label_real = tf.ones([test_batch.shape[0]])
        label_real += 0.05 * tf.random.uniform(tf.shape(label_real))

        label_fake = tf.zeros([test_batch.shape[0]])
        label_fake += 0.05 * tf.random.uniform(tf.shape(label_fake))

        noise = tf.random.normal([test_batch.shape[0], self.latent_dim])
        decay_noise = self.decay_function(epoch)

        real = test_batch
        real = self.add_gaussian_noise(real, std=decay_noise)

        fake = self.generator(noise)
        fake = self.add_gaussian_noise(fake, std=decay_noise)

        output_real = self.discriminator(real)
        output_fake = self.discriminator(fake)

        loss_real = self.loss(label_real, output_real)
        loss_fake = self.loss(label_fake, output_fake)
        loss_discriminator = loss_real + loss_fake

        loss_generator = self.loss(label_real, output_fake)

        self.test_tracker_discriminator_loss.update_state(loss_discriminator)
        self.test_tracker_generator_loss.update_state(loss_generator)

        return {
            "discriminator_loss": self.test_tracker_discriminator_loss.result(),
            "generator_loss": self.test_tracker_generator_loss.result()
        }

    @tf.function
    def log(self, epoch, train_batch, test_batch):
        tf.summary.scalar(
            'Discriminator Loss/Training',
            self.training_tracker_discriminator_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Generator Loss/Training',
            self.training_tracker_generator_loss.result(),
            step=epoch
        )

        tf.summary.scalar(
            'Discriminator Loss/Test',
            self.test_tracker_discriminator_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Generator Loss/Test',
            self.test_tracker_generator_loss.result(),
            step=epoch
        )

        self.training_tracker_discriminator_loss.reset_state()
        self.training_tracker_generator_loss.reset_state()
        self.test_tracker_discriminator_loss.reset_state()
        self.test_tracker_generator_loss.reset_state()

        noise = tf.random.normal([test_batch.shape[0], self.latent_dim])
        decay_noise = self.decay_function(epoch)

        train_image = self.add_gaussian_noise(
            train_batch[0, :, :, :], std=decay_noise)
        test_image = self.add_gaussian_noise(
            test_batch[0, :, :, :], std=decay_noise)

        channels = tf.transpose(train_image, perm=[2, 0, 1])
        channels = tf.expand_dims(channels, -1)
        tf.summary.image(
            "Images/Train",
            channels,
            step=epoch,
            max_outputs=self.channels_num
        )

        channels = tf.transpose(test_image, perm=[2, 0, 1])
        channels = tf.expand_dims(channels, -1)
        tf.summary.image(
            "Images/Test",
            channels,
            step=epoch,
            max_outputs=self.channels_num
        )

        fake_batch = self.generator(noise)
        fake_image = fake_batch[0, :, :, :]

        channels = tf.transpose(fake_image, perm=[2, 0, 1])
        channels = tf.expand_dims(channels, -1)
        tf.summary.image(
            "Images/Fake",
            channels,
            step=epoch,
            max_outputs=self.channels_num
        )

    @ tf.function
    def add_gaussian_noise(self, input, std):
        noise = tf.random.normal(
            tf.shape(input), mean=0.0, stddev=std, dtype=tf.float32)
        return input + noise

    @ tf.function
    def decay_function(self, x):
        return 2.71 ** (tf.cast(-x, dtype=tf.float32) * 0.05)
