import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, layers


class RandomColorAffine(layers.Layer):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )
            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, 3, 3), minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales
                + jitter_matrices
            )
            images = tf.clip_by_value(
                tf.matmul(images, color_transforms), 0, 1)
        return images


class Augmenter(Model):

    def __init__(self, min_area=0.25):
        super(Augmenter, self).__init__()
        self._name = 'augmenter'

        # TODO: tf.Variable().numpy() make the variable serializable.
        # Look to better solution

        zoom_factor = tf.Variable(
            1.0 - tf.sqrt(min_area), trainable=False).numpy()
        brightness = tf.Variable(0.6, trainable=False).numpy()
        jitter = tf.Variable(0.2, trainable=False).numpy()

        self.flip = layers.RandomFlip("horizontal")
        self.translation = layers.RandomTranslation(
            zoom_factor / 2, zoom_factor / 2)
        self.zoom = layers.RandomZoom(
            (-zoom_factor, 0.0), (-zoom_factor, 0.0))
        self.color = RandomColorAffine(brightness, jitter)

    def call(self, inputs):

        x = self.flip(inputs)
        x = self.translation(x)
        x = self.zoom(x)
        x = self.color(x)

        return x


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

    def __init__(self, image_dim, channels_num, latent_dim, filters):
        super(SimCLR, self).__init__()
        self._name = 'simclr'
        self.channels_num = channels_num

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

    def compile(self, optimizer, learning_rate):
        super(SimCLR, self).compile()

        self.optimizer = tf.keras.optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })
        self.contrastive_loss = tfa.losses.ContrastiveLoss()

        self.training_tracker_contrastive_loss = tf.keras.metrics.Mean()
        self.test_tracker_contrastive_loss = tf.keras.metrics.Mean()

    @property
    def metrics(self):
        return [
            self.training_tracker_contrastive_loss,
            self.test_tracker_contrastive_loss
        ]

    def call(self, inputs):
        augmented = self.contrastive_augmenter(inputs)

        encoded = self.encoder(augmented)
        projection = self.projection_head(encoded)

        return projection

    @tf.function
    def train_step(self, train_batch):

        projections_1 = self(tf.random.shuffle(train_batch))

        with tf.GradientTape() as tape:
            projections_2 = self(train_batch)
            projections_3 = self(train_batch)

            label_positive = tf.zeros([train_batch.shape[0]])
            prediction_positive = tf.norm(
                projections_2 - projections_3, axis=1)

            label_negative = tf.ones([train_batch.shape[0]])
            prediction_negative = tf.norm(
                projections_1 - projections_2, axis=1)

            labels = tf.concat([label_positive, label_negative], axis=0)
            predictions = tf.concat(
                [prediction_positive, prediction_negative], axis=0)

            contrastive_loss = self.contrastive_loss(labels, predictions)

        grads = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights
        )
        self.optimizer.apply_gradients(
            zip(
                grads,
                self.encoder.trainable_weights + self.projection_head.trainable_weights
            )
        )

        self.training_tracker_contrastive_loss.update_state(contrastive_loss)

    @tf.function
    def test_step(self, test_batch):
        projections_1 = self(tf.random.shuffle(test_batch))
        projections_2 = self(test_batch)
        projections_3 = self(test_batch)

        label_positive = tf.zeros([test_batch.shape[0]])
        prediction_positive = tf.norm(
            projections_2 - projections_3, axis=1)

        label_negative = tf.ones([test_batch.shape[0]])
        prediction_negative = tf.norm(
            projections_1 - projections_2, axis=1)

        labels = tf.concat([label_positive, label_negative], axis=0)
        predictions = tf.concat(
            [prediction_positive, prediction_negative], axis=0)

        contrastive_loss = self.contrastive_loss(labels, predictions)

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

        train_image = tf.expand_dims(train_image, 0)
        augmented_1 = self.contrastive_augmenter(train_image)
        predicted_channels = tf.transpose(augmented_1, perm=[3, 1, 2, 0])
        tf.summary.image(
            "Images/Train/Projection 1",
            predicted_channels,
            step=epoch,
            max_outputs=self.channels_num
        )

        augmented_2 = self.contrastive_augmenter(train_image)
        predicted_channels = tf.transpose(augmented_2, perm=[3, 1, 2, 0])
        tf.summary.image(
            "Images/Train/Projection 2",
            predicted_channels,
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

        test_image = tf.expand_dims(test_image, 0)
        augmented_1 = self.contrastive_augmenter(test_image)
        predicted_channels = tf.transpose(augmented_1, perm=[3, 1, 2, 0])
        tf.summary.image(
            "Images/Test/Projection 1",
            predicted_channels,
            step=epoch,
            max_outputs=self.channels_num
        )

        augmented_2 = self.contrastive_augmenter(test_image)
        predicted_channels = tf.transpose(augmented_2, perm=[3, 1, 2, 0])
        tf.summary.image(
            "Images/Test/Projection 2",
            predicted_channels,
            step=epoch,
            max_outputs=self.channels_num
        )
