import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, activations


class DownSampling(Model):

    def __init__(self, name, filter):
        super(DownSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU()
        self.pool = layers.MaxPooling2D((2, 2))

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)

        return x


class UpSampling(Model):

    def __init__(self, name, filter):
        super(UpSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2DTranspose(
            filter, (3, 3), strides=1, padding='same')
        self.bn = layers.BatchNormalization()
        self.act = layers.LeakyReLU()
        self.sampling = layers.UpSampling2D((2, 2), interpolation='nearest')

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.sampling(x)

        return x


class Sampling(Model):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Model):

    def __init__(self, latent_dim, filters):
        super(Encoder, self).__init__()
        self._name = 'encoder'

        self.downsamplings = []
        for id, filter in enumerate(filters):
            self.downsamplings.append(
                DownSampling(f'down_sampling_{id}', filter)
            )

        self.flat = layers.Flatten()
        self.dense = layers.Dense(units=latent_dim)
        self.act = layers.LeakyReLU()

        self.z_mean = layers.Dense(latent_dim, name="z_mean")
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")
        self.z = Sampling()

    def call(self, inputs):

        x = self.downsamplings[0](inputs)

        for ds in self.downsamplings[1:]:
            x = ds(x)

        x = self.flat(x)

        x = self.dense(x)
        x = self.act(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.z([z_mean, z_log_var])

        return z_mean, z_log_var, z


class Decoder(Model):

    def __init__(self, flat_dim, pre_flat_shape, channels_num, filters):
        super(Decoder, self).__init__()
        self._name = 'decoder'

        self.dense = layers.Dense(units=flat_dim)
        self.reshape = layers.Reshape(pre_flat_shape)

        self.upsamplings = []
        for id, filter in enumerate(filters):
            self.upsamplings.append(
                UpSampling(f'up_sampling_{id}', filter)
            )

        self.transpose = layers.Conv2DTranspose(
            channels_num, (3, 3), padding='same')
        self.act = layers.Activation(activations.sigmoid)

    def call(self, inputs):

        x = self.dense(inputs)
        x = self.reshape(x)

        for us in self.upsamplings:
            x = us(x)

        x = self.transpose(x)
        decoded = self.act(x)

        return decoded


class CVAE(Model):
    def __init__(self, image_dim, channels_num, latent_dim, filters):
        super(CVAE, self).__init__()
        self._name = 'cvae'
        self.channels_num = channels_num
        self.best_loss = np.inf

        encoder_input_shape = (None, image_dim, image_dim, channels_num)
        decoder_input_shape = (None, latent_dim)

        self.encoder = Encoder(latent_dim, filters)
        self.encoder.build(input_shape=encoder_input_shape)
        self.encoder.call(layers.Input(shape=encoder_input_shape[1:]))
        self.encoder.summary()

        flat_dim = self.encoder.flat.output_shape[1]
        pre_flat_shape = self.encoder.flat.input_shape[1:]

        self.decoder = Decoder(
            flat_dim, pre_flat_shape, channels_num, filters[::-1]
        )
        self.decoder.build(input_shape=decoder_input_shape)
        self.decoder.call(layers.Input(shape=decoder_input_shape[1:]))
        self.decoder.summary()

        self.build(input_shape=encoder_input_shape)
        self.call(layers.Input(shape=encoder_input_shape[1:]))
        self.summary()

    def compile(self, optimizer, learning_rate, loss):
        super(CVAE, self).compile()

        self.optimizer = tf.keras.optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })
        self.decoded_loss = tf.keras.losses.get({
            "class_name": loss,
            "config": {
                "reduction": "none"
            }
        })

        self.training_tracker_decoded_loss = tf.keras.metrics.Mean()
        self.training_tracker_kl_loss = tf.keras.metrics.Mean()
        self.training_tracker_total_loss = tf.keras.metrics.Mean()
        self.test_tracker_decoded_loss = tf.keras.metrics.Mean()
        self.test_tracker_kl_loss = tf.keras.metrics.Mean()
        self.test_tracker_total_loss = tf.keras.metrics.Mean()

    @property
    def metrics(self):
        return [
            self.training_tracker_decoded_loss,
            self.training_tracker_kl_loss,
            self.training_tracker_total_loss,
            self.test_tracker_decoded_loss,
            self.test_tracker_kl_loss,
            self.test_tracker_total_loss
        ]

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs)

        if not training:
            return z

        decoded = self.decoder(z)

        return decoded

    @tf.function
    def train_step(self, train_batch):

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(train_batch)
            decoded = self.decoder(z)
            decoded_loss = tf.reduce_mean(
                tf.reduce_sum(
                    self.decoded_loss(train_batch, decoded), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var -
                              tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = decoded_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.training_tracker_decoded_loss.update_state(decoded_loss)
        self.training_tracker_kl_loss.update_state(kl_loss)
        self.training_tracker_total_loss.update_state(total_loss)

        return {
            "decoded_loss": self.training_tracker_decoded_loss.result(),
            "kl_loss": self.training_tracker_kl_loss.result(),
            "total_loss": self.training_tracker_total_loss.result(),
        }

    @tf.function
    def test_step(self, test_batch):

        z_mean, z_log_var, z = self.encoder(test_batch)
        decoded = self.decoder(z)
        decoded_loss = tf.reduce_mean(
            tf.reduce_sum(
                self.decoded_loss(test_batch, decoded), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var -
                          tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = decoded_loss + kl_loss

        self.test_tracker_decoded_loss.update_state(decoded_loss)
        self.test_tracker_kl_loss.update_state(kl_loss)
        self.test_tracker_total_loss.update_state(total_loss)

        return {
            "decoded_loss": self.test_tracker_decoded_loss.result(),
            "kl_loss": self.test_tracker_kl_loss.result(),
            "total_loss": self.test_tracker_total_loss.result(),
        }

    def save_best_model(self, model_dir):
        loss_to_monitor = self.test_tracker_total_loss.result()

        if self.best_loss > loss_to_monitor:
            self.best_loss = loss_to_monitor
            self.save(model_dir)

    def log(self, epoch, train_batch, test_batch):
        tf.summary.scalar(
            'Decoded Loss/Training',
            self.training_tracker_decoded_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'KL Loss/Training',
            self.training_tracker_kl_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Total Loss/Training',
            self.training_tracker_total_loss.result(),
            step=epoch
        )

        tf.summary.scalar(
            'Decoded Loss/Test',
            self.test_tracker_decoded_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'KL Loss/Test',
            self.test_tracker_kl_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Total Loss/Test',
            self.test_tracker_total_loss.result(),
            step=epoch
        )

        # Log images
        if epoch % 100 == 0:
            train_image = train_batch[0, :, :, :]
            test_image = test_batch[0, :, :, :]

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

            train_image = tf.expand_dims(train_image, 0)
            predicted_image = self(train_image, training=True)
            predicted_channels = tf.transpose(
                predicted_image, perm=[3, 1, 2, 0])
            tf.summary.image(
                "Images/Train/Predicted",
                predicted_channels,
                step=epoch,
                max_outputs=self.channels_num
            )

            test_image = tf.expand_dims(test_image, 0)
            predicted_image = self(test_image, training=True)
            predicted_channels = tf.transpose(
                predicted_image, perm=[3, 1, 2, 0])
            tf.summary.image(
                "Images/Test/Predicted",
                predicted_channels,
                step=epoch,
                max_outputs=self.channels_num
            )

    def reset_losses_state(self):
        self.training_tracker_decoded_loss.reset_state()
        self.training_tracker_kl_loss.reset_state()
        self.training_tracker_total_loss.reset_state()
        self.test_tracker_decoded_loss.reset_state()
        self.test_tracker_kl_loss.reset_state()
        self.test_tracker_total_loss.reset_state()
