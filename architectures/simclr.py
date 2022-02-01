import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers


class RandomColorAffine(Model):
    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def call(self, images, training=False):

        if training:
            batch_size = tf.shape(images)[0]
            channels_num = tf.shape(images)[-1]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness
            )

            # Different for all colors
            jitter_matrices = tf.random.uniform(
                (batch_size, 1, channels_num, channels_num),
                minval=-self.jitter, maxval=self.jitter
            )

            color_transforms = (
                tf.eye(
                    channels_num, batch_shape=[batch_size, 1]
                ) * brightness_scales + jitter_matrices
            )

            images = tf.clip_by_value(
                tf.matmul(images, color_transforms), 0, 1)

        return images


class Augmenter(Model):

    def __init__(self, min_area, brightness, jitter):
        super(Augmenter, self).__init__()
        self._name = 'augmenter'

        zoom_factor = tf.Variable(
            1.0 - tf.sqrt(min_area), trainable=False).numpy()

        self.flip = layers.RandomFlip("horizontal")
        self.translation = layers.RandomTranslation(
            zoom_factor / 2, zoom_factor / 2)
        self.zoom = layers.RandomZoom(
            (-zoom_factor, 0.0), (-zoom_factor, 0.0))
        self.color = RandomColorAffine(brightness, jitter)

    def call(self, inputs, training=False):

        x = self.flip(inputs)
        x = self.translation(x)
        x = self.zoom(x)
        x = self.color(x, training)

        return x


class DownSampling(Model):

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

    def __init__(self, image_dim, channels_num, latent_dim, filters, temperature):
        super(SimCLR, self).__init__()
        self._name = 'simclr'
        self.channels_num = channels_num
        self.temperature = temperature
        self.best_loss = np.inf

        contrastive_augmentation = {
            "min_area": 0.5, "brightness": 0.6, "jitter": 0.2}

        encoder_input_shape = (None, image_dim, image_dim, channels_num)
        projection_head_input_shape = (None, latent_dim)

        self.contrastive_augmenter = Augmenter(**contrastive_augmentation)
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

        self.contrastive_optimizer = tf.keras.optimizers.get({
            "class_name": optimizer,
            "config": {
                "learning_rate": learning_rate
            }
        })

        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        self.training_tracker_contrastive_loss = tf.keras.metrics.Mean()
        self.training_tracker_contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        self.test_tracker_contrastive_loss = tf.keras.metrics.Mean()
        self.test_tracker_contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    @property
    def metrics(self):
        return [
            self.training_tracker_contrastive_loss,
            self.training_tracker_contrastive_accuracy,
            self.test_tracker_contrastive_loss,
            self.test_tracker_contrastive_accuracy
        ]

    def contrastive_loss(self, projections_1, projections_2):
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # Cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = tf.math.l2_normalize(projections_1, axis=1)
        projections_2 = tf.math.l2_normalize(projections_2, axis=1)

        similarities = (
            tf.matmul(projections_1, projections_2,
                      transpose_b=True) / self.temperature
        )

        # The similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(projections_1)[0]
        contrastive_labels = tf.range(batch_size)

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )

        contrastive_loss = (loss_1_2 + loss_2_1) / 2

        return contrastive_labels, similarities, contrastive_loss

    def call(self, inputs, training=False):
        augmented = self.contrastive_augmenter(inputs, training)

        x = self.encoder(augmented)

        return x

    @tf.function
    def train_step(self, train_batch):

        with tf.GradientTape() as tape:
            features_1 = self(train_batch, training=True)
            features_2 = self(train_batch, training=True)

            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_labels, similarities, contrastive_loss = self.contrastive_loss(
                projections_1, projections_2)

        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )

        self.training_tracker_contrastive_loss.update_state(contrastive_loss)

        self.training_tracker_contrastive_accuracy.update_state(
            contrastive_labels, similarities)
        self.training_tracker_contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        return {
            "contrastive_loss": self.training_tracker_contrastive_loss.result(),
            "contrastive_accuracy": self.training_tracker_contrastive_accuracy.result()
        }

    @tf.function
    def test_step(self, test_batch):

        features_1 = self(test_batch, training=True)
        features_2 = self(test_batch, training=True)

        projections_1 = self.projection_head(features_1)
        projections_2 = self.projection_head(features_2)
        contrastive_labels, similarities, contrastive_loss = self.contrastive_loss(
            projections_1, projections_2)

        self.test_tracker_contrastive_loss.update_state(contrastive_loss)

        self.test_tracker_contrastive_accuracy.update_state(
            contrastive_labels, similarities)
        self.test_tracker_contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        return {
            "contrastive_loss": self.test_tracker_contrastive_loss.result(),
            "contrastive_accuracy": self.test_tracker_contrastive_accuracy.result()
        }

    def save_best_model(self, model_dir):
        loss_to_monitor = self.test_tracker_contrastive_loss.result()

        if self.best_loss > loss_to_monitor:
            self.best_loss = loss_to_monitor
            self.save(model_dir)

    def log(self, epoch, train_batch, test_batch):
        tf.summary.scalar(
            'Contrastive Loss/Training',
            self.training_tracker_contrastive_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Contrastive Accuracy/Training',
            self.training_tracker_contrastive_accuracy.result(),
            step=epoch
        )

        tf.summary.scalar(
            'Contrastive Loss/Test',
            self.test_tracker_contrastive_loss.result(),
            step=epoch
        )
        tf.summary.scalar(
            'Contrastive Accuracy/Test',
            self.test_tracker_contrastive_accuracy.result(),
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

            train_image = tf.expand_dims(train_image, 0)
            augmented_1 = self.contrastive_augmenter(
                train_image, training=True)
            predicted_channels = tf.transpose(augmented_1, perm=[3, 1, 2, 0])
            tf.summary.image(
                "Images/Train/Augmented 1",
                predicted_channels,
                step=epoch,
                max_outputs=self.channels_num
            )

            augmented_2 = self.contrastive_augmenter(
                train_image, training=True)
            predicted_channels = tf.transpose(augmented_2, perm=[3, 1, 2, 0])
            tf.summary.image(
                "Images/Train/Augmented 2",
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
            augmented_1 = self.contrastive_augmenter(test_image, training=True)
            predicted_channels = tf.transpose(augmented_1, perm=[3, 1, 2, 0])
            tf.summary.image(
                "Images/Test/Augmented 1",
                predicted_channels,
                step=epoch,
                max_outputs=self.channels_num
            )

            augmented_2 = self.contrastive_augmenter(test_image, training=True)
            predicted_channels = tf.transpose(augmented_2, perm=[3, 1, 2, 0])
            tf.summary.image(
                "Images/Test/Augmented 2",
                predicted_channels,
                step=epoch,
                max_outputs=self.channels_num
            )

    def reset_losses_state(self):
        self.training_tracker_contrastive_loss.reset_state()
        self.training_tracker_contrastive_accuracy.reset_state()
        self.test_tracker_contrastive_loss.reset_state()
        self.test_tracker_contrastive_accuracy.reset_state()
