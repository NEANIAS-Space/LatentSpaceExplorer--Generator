import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, activations


class DownSampling(Model):

    def __init__(self, name, filter):
        super(DownSampling, self).__init__()
        self._name = name

        self.conv1 = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filter, (1, 1), strides=1)

        self.add = layers.Add()
        self.act2 = layers.ReLU()

        self.pool = layers.MaxPooling2D((2, 2))

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        res = self.conv3(inputs)

        x = self.add([res, x])
        x = self.act2(x)

        x = self.pool(x)

        return x


class UpSampling(Model):

    def __init__(self, name, filter):
        super(UpSampling, self).__init__()
        self._name = name

        self.conv1 = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filter, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filter, (1, 1), strides=1)

        self.add = layers.Add()
        self.act2 = layers.ReLU()

        self.sampling = layers.UpSampling2D((2, 2), interpolation='nearest')

    def call(self, inputs):

        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        res = self.conv3(inputs)

        x = self.add([res, x])
        x = self.act2(x)

        x = self.sampling(x)

        return x


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

    def call(self, inputs):

        x = self.downsamplings[0](inputs)

        for ds in self.downsamplings[1:]:
            x = ds(x)

        x = self.flat(x)
        encoded = self.dense(x)

        return encoded


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
            channels_num, (3, 3), strides=1, padding='same')
        self.act = layers.Activation(activations.sigmoid)

    def call(self, inputs):

        x = self.dense(inputs)
        x = self.reshape(x)

        for us in self.upsamplings:
            x = us(x)

        x = self.transpose(x)
        decoded = self.act(x)

        return decoded


class SCAE(Model):

    def __init__(self, image_dim, channels_num, latent_dim, filters):
        super(SCAE, self).__init__()
        self._name = 'scae'
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
        super(SCAE, self).compile()

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
        self.test_tracker_decoded_loss = tf.keras.metrics.Mean()

    @property
    def metrics(self):
        return [
            self.training_tracker_decoded_loss,
            self.test_tracker_decoded_loss
        ]

    def call(self, inputs, training=False):
        encoded = self.encoder(inputs)

        if not training:
            return encoded

        decoded = self.decoder(encoded)

        return decoded

    @tf.function
    def train_step(self, train_batch):

        with tf.GradientTape() as tape:
            decoded = self(train_batch, training=True)
            decoded_loss = self.decoded_loss(train_batch, decoded)

        grads = tape.gradient(decoded_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.training_tracker_decoded_loss.update_state(decoded_loss)

        return {
            "decoded_loss": self.training_tracker_decoded_loss.result()
        }

    @tf.function
    def test_step(self, test_batch):
        decoded = self(test_batch, training=True)
        decoded_loss = self.decoded_loss(test_batch, decoded)

        self.test_tracker_decoded_loss.update_state(decoded_loss)

        return {
            "decoded_loss": self.test_tracker_decoded_loss.result()
        }

    def save_best_model(self, model_dir):
        loss_to_monitor = self.test_tracker_decoded_loss.result()

        if self.best_loss > loss_to_monitor:
            self.best_loss = loss_to_monitor
            self.save(model_dir)

    def log(self, epoch, train_batch, test_batch):

        # Log scalars
        tf.summary.scalar(
            'Decoded Loss/Training',
            self.training_tracker_decoded_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Decoded Loss/Test',
            self.test_tracker_decoded_loss.result(),
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
        self.test_tracker_decoded_loss.reset_state()
