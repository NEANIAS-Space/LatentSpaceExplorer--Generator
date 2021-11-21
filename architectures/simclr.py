import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, layers


class DownSampling(layers.Layer):

    def __init__(self, name, filter):
        super(DownSampling, self).__init__()
        self._name = name

        self.conv = layers.Conv2D(filter, (3, 3), strides=2, padding='same')
        self.act = layers.ReLU()

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.act(x)

        return x


class Augmenter(Model):

    def __init__(self, min_area=0.25):
        super(Augmenter, self).__init__()
        self._name = 'augmenter'

        # TODO: tf.Variable().numpy() make the variable serializable.
        # Look to better solution
        zoom_factor = tf.Variable(
            1.0 - tf.sqrt(min_area), trainable=False).numpy()

        self.flip = layers.RandomFlip("horizontal")
        self.translation = layers.RandomTranslation(
            zoom_factor / 2, zoom_factor / 2)
        self.zoom = layers.RandomZoom(
            (-zoom_factor, 0.0), (-zoom_factor, 0.0))

    def call(self, inputs):

        x = self.flip(inputs)
        x = self.translation(x)
        x = self.zoom(x)

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
        self.act = layers.ReLU()

    def call(self, inputs):

        x = self.downsamplings[0](inputs)

        for ds in self.downsamplings[1:]:
            x = ds(x)

        x = self.flat(x)
        x = self.dense(x)
        x = self.act(x)

        return x


class ProjectionHead(Model):

    def __init__(self, latent_dim):
        super(ProjectionHead, self).__init__()
        self._name = 'projection_head'

        self.dense1 = layers.Dense(units=latent_dim)
        self.act = layers.ReLU()

        self.dense2 = layers.Dense(units=latent_dim)

    def call(self, inputs):

        x = self.dense1(inputs)
        x = self.act(x)

        x = self.dense2(x)

        return x


class SimCLR(Model):

    def __init__(self, image_dim, channels_num, latent_dim, filters, optimizer, learning_rate):
        super(SimCLR, self).__init__()
        self._name = 'simclr'

        encoder_input_shape = (None, image_dim, image_dim, channels_num)
        projection_head_input_shape = (None, latent_dim)

        self.contrastive_augmenter = Augmenter()
        self.contrastive_augmenter.build(input_shape=encoder_input_shape)
        self.contrastive_augmenter.call(
            layers.Input(shape=encoder_input_shape[1:]))
        self.contrastive_augmenter.summary()

        self.encoder = Encoder(latent_dim, filters)
        self.encoder.build(input_shape=encoder_input_shape)
        self.encoder.call(layers.Input(shape=encoder_input_shape[1:]))
        self.encoder.summary()

        self.projection_head = ProjectionHead(latent_dim)
        self.projection_head.build(input_shape=projection_head_input_shape)
        self.projection_head.call(
            layers.Input(shape=projection_head_input_shape[1:]))
        self.projection_head.summary()

        self.build(input_shape=encoder_input_shape)
        self.call(layers.Input(shape=encoder_input_shape[1:]))
        self.summary()

        self.optimizer = tf.keras.optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })
        self.contrastive_loss = tfa.losses.ContrastiveLoss()

        self.training_tracker_contrastive_loss = tf.keras.metrics.Mean()
        self.test_tracker_contrastive_loss = tf.keras.metrics.Mean()

        self.compile(optimizer=optimizer)

    @property
    def metrics(self):
        return [
            self.training_tracker_contrastive_loss,
            self.test_tracker_contrastive_loss
        ]

    def call(self, inputs, training=True):
        if training:
            inputs = self.contrastive_augmenter(inputs)

        encoded = self.encoder(inputs)
        projection = self.projection_head(encoded)

        return projection

    @tf.function
    def train_step(self, train_batch):

        with tf.GradientTape() as tape:
            projections_1 = self(train_batch, training=True)
            projections_2 = self(train_batch, training=True)

            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2)

        grads = tape.gradient(contrastive_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.training_tracker_contrastive_loss.update_state(contrastive_loss)

    @tf.function
    def test_step(self, test_batch):
        projections_1 = self(test_batch, training=False)
        projections_2 = self(test_batch, training=False)

        contrastive_loss = self.contrastive_loss(projections_1, projections_2)

        self.test_tracker_contrastive_loss.update_state(contrastive_loss)

    @tf.function
    def log(self, epoch, train_batch, test_batch):
        tf.summary.scalar(
            'Contrastive Loss/Training',
            self.training_tracker_contrastive_loss.result(),
            step=epoch
        )

        tf.summary.scalar(
            'Contrastive Loss/Test',
            self.test_tracker_contrastive_loss.result(),
            step=epoch
        )

        self.training_tracker_contrastive_loss.reset_state()
        self.test_tracker_contrastive_loss.reset_state()
