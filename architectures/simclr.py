# https://github.com/keras-team/keras-io/blob/master/examples/vision/semisupervised_simclr.py\

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow.keras import Model, layers
import matplotlib.pyplot as plt


class RandomColorAffine(Model):
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

    def __init__(self, min_area, brightness, jitter):
        super(Augmenter, self).__init__()
        self._name = 'augmenter'

        zoom_factor = tf.Variable(
            1.0 - tf.sqrt(min_area), trainable=False).numpy()

        self.rescale = layers.Rescaling(1 / 255)
        self.flip = layers.RandomFlip("horizontal")
        self.translation = layers.RandomTranslation(
            zoom_factor / 2, zoom_factor / 2)
        self.zoom = layers.RandomZoom(
            (-zoom_factor, 0.0), (-zoom_factor, 0.0))
        self.color = RandomColorAffine(brightness, jitter)

    def call(self, inputs):

        x = self.rescale(inputs)
        x = self.flip(x)
        x = self.translation(x)
        x = self.zoom(x)
        x = self.color(x)

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


class LinearProbe(Model):

    def __init__(self, units):
        super(LinearProbe, self).__init__()
        self._name = 'linear_probe'

        self.dense = layers.Dense(units=units)

    def call(self, inputs):

        x = self.dense(inputs)

        return x


class SimCLR(Model):

    def __init__(self, image_dim, channels_num, latent_dim, filters, temperature):
        super(SimCLR, self).__init__()
        self._name = 'simclr'
        self.channels_num = channels_num
        self.temperature = temperature
        self.best_loss = np.inf

        self.contrastive_augmentation = {
            "min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
        self.classification_augmentation = {
            "min_area": 0.75, "brightness": 0.3, "jitter": 0.1}

        encoder_input_shape = (None, image_dim, image_dim, channels_num)
        projection_head_input_shape = (None, latent_dim)

        self.contrastive_augmenter = Augmenter(**contrastive_augmentation)
        self.contrastive_augmenter.build(input_shape=encoder_input_shape)
        self.contrastive_augmenter.call(
            layers.Input(shape=encoder_input_shape[1:]))
        self.contrastive_augmenter.summary()

        self.classification_augmenter = Augmenter(
            **classification_augmentation)
        self.classification_augmenter.build(input_shape=encoder_input_shape)
        self.classification_augmenter.call(
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

        # self.linear_probe = LinearProbe(units=10)
        # self.linear_probe.build(input_shape=projection_head_input_shape)
        # self.linear_probe.call(
        #     layers.Input(shape=projection_head_input_shape[1:]))
        # self.linear_probe.summary()

        self.build(input_shape=encoder_input_shape)
        self.call(layers.Input(shape=encoder_input_shape[1:]))
        self.summary()

    def compile(self, contrastive_optimizer, probe_optimizer):
        super(SimCLR, self).compile()

        self.contrastive_optimizer = contrastive_optimizer
        # self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        self.contrastive_loss_tracker = tf.keras.metrics.Mean(name="c_loss")
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="c_acc"
        )
        # self.probe_loss_tracker = tf.keras.metrics.Mean(name="p_loss")
        # self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        #     name="p_acc")

    @property
    def metrics(self):
        return [
            self.contrastive_loss_tracker,
            self.contrastive_accuracy,
            # self.probe_loss_tracker,
            # self.probe_accuracy,
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
        self.contrastive_accuracy.update_state(
            contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(
            contrastive_labels, tf.transpose(similarities)
        )

        # The temperature-scaled similarities are used as logits for cross-entropy
        # a symmetrized version of the loss is used here
        loss_1_2 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, similarities, from_logits=True
        )
        loss_2_1 = tf.keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, tf.transpose(similarities), from_logits=True
        )
        return (loss_1_2 + loss_2_1) / 2

    def call(self, inputs):
        augmented = self.contrastive_augmenter(inputs)

        encoded = self.encoder(augmented)
        projection = self.projection_head(encoded)

        return projection

    @tf.function
    def train_step(self, data):
        (unlabeled_images, _), (labeled_images, labels) = data

        # Both labeled and unlabeled images are used, without labels
        images = tf.concat((unlabeled_images, labeled_images), axis=0)
        # Each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images, training=True)
        augmented_images_2 = self.contrastive_augmenter(images, training=True)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1, training=True)
            features_2 = self.encoder(augmented_images_2, training=True)
            # The representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1, training=True)
            projections_2 = self.projection_head(features_2, training=True)
            contrastive_loss = self.contrastive_loss(
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
        self.contrastive_loss_tracker.update_state(contrastive_loss)

        # # Labels are only used in evalutation for an on-the-fly logistic regression
        # preprocessed_images = self.classification_augmenter(
        #     labeled_images, training=True
        # )
        # with tf.GradientTape() as tape:
        #     # the encoder is used in inference mode here to avoid regularization
        #     # and updating the batch normalization paramers if they are used
        #     features = self.encoder(preprocessed_images, training=False)
        #     class_logits = self.linear_probe(features, training=True)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(
        #     probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_loss_tracker.update_state(probe_loss)
        # self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    # @tf.function
    # def test_step(self, data):
    #     labeled_images, labels = data

    #     # For testing the components are used with a training=False flag
    #     preprocessed_images = self.classification_augmenter(
    #         labeled_images, training=False
    #     )
    #     features = self.encoder(preprocessed_images, training=False)
    #     class_logits = self.linear_probe(features, training=False)
    #     probe_loss = self.probe_loss(labels, class_logits)
    #     self.probe_loss_tracker.update_state(probe_loss)
    #     self.probe_accuracy.update_state(labels, class_logits)

    #     # Only the probe metrics are logged at test time
    #     return {m.name: m.result() for m in self.metrics[2:]}

    # def save_best_model(self, model_dir):
    #     loss_to_monitor = self.test_tracker_contrastive_loss.result()

    #     if self.best_loss > loss_to_monitor:
    #         self.best_loss = loss_to_monitor
    #         self.save(model_dir)

    # def log(self, epoch, train_batch, test_batch):
    #     tf.summary.scalar(
    #         'Contrastive Loss/Training',
    #         self.training_tracker_contrastive_loss.result(),
    #         step=epoch
    #     )

    #     tf.summary.scalar(
    #         'Contrastive Loss/Test',
    #         self.test_tracker_contrastive_loss.result(),
    #         step=epoch
    #     )

    #     # Log images
    #     if epoch % 100 == 0:
    #         train_image = train_batch[0, :, :, :]
    #         test_image = test_batch[0, :, :, :]

    #         channels = tf.transpose(train_image, perm=[2, 0, 1])
    #         channels = tf.expand_dims(channels, -1)
    #         tf.summary.image(
    #             "Images/Train",
    #             channels,
    #             step=epoch,
    #             max_outputs=self.channels_num
    #         )

    #         train_image = tf.expand_dims(train_image, 0)
    #         augmented_1 = self.contrastive_augmenter(train_image)
    #         predicted_channels = tf.transpose(augmented_1, perm=[3, 1, 2, 0])
    #         tf.summary.image(
    #             "Images/Train/Projection 1",
    #             predicted_channels,
    #             step=epoch,
    #             max_outputs=self.channels_num
    #         )

    #         augmented_2 = self.contrastive_augmenter(train_image)
    #         predicted_channels = tf.transpose(augmented_2, perm=[3, 1, 2, 0])
    #         tf.summary.image(
    #             "Images/Train/Projection 2",
    #             predicted_channels,
    #             step=epoch,
    #             max_outputs=self.channels_num
    #         )

    #         channels = tf.transpose(test_image, perm=[2, 0, 1])
    #         channels = tf.expand_dims(channels, -1)
    #         tf.summary.image(
    #             "Images/Test",
    #             channels,
    #             step=epoch,
    #             max_outputs=self.channels_num
    #         )

    #         test_image = tf.expand_dims(test_image, 0)
    #         augmented_1 = self.contrastive_augmenter(test_image)
    #         predicted_channels = tf.transpose(augmented_1, perm=[3, 1, 2, 0])
    #         tf.summary.image(
    #             "Images/Test/Projection 1",
    #             predicted_channels,
    #             step=epoch,
    #             max_outputs=self.channels_num
    #         )

    #         augmented_2 = self.contrastive_augmenter(test_image)
    #         predicted_channels = tf.transpose(augmented_2, perm=[3, 1, 2, 0])
    #         tf.summary.image(
    #             "Images/Test/Projection 2",
    #             predicted_channels,
    #             step=epoch,
    #             max_outputs=self.channels_num
    #         )

    # def reset_losses_state(self):
    #     self.training_tracker_contrastive_loss.reset_state()
    #     self.test_tracker_contrastive_loss.reset_state()


if __name__ == "__main__":

    # Dataset hyperparameters
    unlabeled_dataset_size = 100000
    labeled_dataset_size = 5000
    image_size = 96
    image_channels = 3

    # Algorithm hyperparameters
    num_epochs = 10
    batch_size = 525  # Corresponds to 200 steps per epoch
    width = 64
    temperature = 0.1
    # Stronger augmentations for contrastive, weaker ones for supervised training
    contrastive_augmentation = {
        "min_area": 0.25, "brightness": 0.6, "jitter": 0.2}
    classification_augmentation = {
        "min_area": 0.75, "brightness": 0.3, "jitter": 0.1}

    steps_per_epoch = (unlabeled_dataset_size +
                       labeled_dataset_size) // batch_size
    unlabeled_batch_size = unlabeled_dataset_size // steps_per_epoch
    labeled_batch_size = labeled_dataset_size // steps_per_epoch
    print(
        f"batch size is {unlabeled_batch_size} (unlabeled) + {labeled_batch_size} (labeled)"
    )

    unlabeled_train_dataset = (
        tfds.load("stl10", split="unlabelled",
                  as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=10 * unlabeled_batch_size)
        .batch(unlabeled_batch_size)
    )
    labeled_train_dataset = (
        tfds.load("stl10", split="train",
                  as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=10 * labeled_batch_size)
        .batch(labeled_batch_size)
    )
    test_dataset = (
        tfds.load("stl10", split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    ###########################################################################
    # Baseline
    ###########################################################################

    # baseline_model = tf.keras.Sequential(
    #     [
    #         tf.keras.Input(shape=(image_size, image_size, image_channels)),
    #         Augmenter(**classification_augmentation),
    #         Encoder(width, [width, width, width, width]),
    #         layers.Dense(10),
    #     ],
    #     name="baseline_model",
    # )
    # baseline_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    # )

    # baseline_history = baseline_model.fit(
    #     labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
    # )

    # print(
    #     "Maximal validation accuracy: {:.2f}%".format(
    #         max(baseline_history.history["val_acc"]) * 100
    #     )
    # )

    ###########################################################################
    # Contrastive Model
    ###########################################################################

    pretraining_model = SimCLR(
        image_size, image_channels,
        width, [width, width, width, width],
        temperature
    )

    pretraining_model.compile(
        contrastive_optimizer=tf.keras.optimizers.Adam(),
        probe_optimizer=tf.keras.optimizers.Adam(),
    )

    pretraining_history = pretraining_model.fit(
        train_dataset, epochs=num_epochs, validation_data=test_dataset
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(pretraining_history.history["val_p_acc"]) * 100
        )
    )

    ###########################################################################
    # Contrastive Model
    ###########################################################################

    # Supervised finetuning of the pretrained encoder
    finetuning_model = tf.keras.Sequential(
        [
            layers.Input(shape=(image_size, image_size, image_channels)),
            Augmenter(**classification_augmentation),
            pretraining_model.encoder,
            layers.Dense(10),
        ],
        name="finetuning_model",
    )
    finetuning_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )

    finetuning_history = finetuning_model.fit(
        labeled_train_dataset, epochs=num_epochs, validation_data=test_dataset
    )
    print(
        "Maximal validation accuracy: {:.2f}%".format(
            max(finetuning_history.history["val_acc"]) * 100
        )
    )

    # The classification accuracies of the baseline and the pretraining + finetuning process:
    def plot_training_curves(pretraining_history, finetuning_history, baseline_history):
        for metric_key, metric_name in zip(["acc", "loss"], ["accuracy", "loss"]):
            plt.figure(figsize=(8, 5), dpi=100)
            plt.plot(
                baseline_history.history[f"val_{metric_key}"], label="supervised baseline"
            )
            plt.plot(
                pretraining_history.history[f"val_p_{metric_key}"],
                label="self-supervised pretraining",
            )
            plt.plot(
                finetuning_history.history[f"val_{metric_key}"],
                label="supervised finetuning",
            )
            plt.legend()
            plt.title(f"Classification {metric_name} during training")
            plt.xlabel("epochs")
            plt.ylabel(f"validation {metric_name}")
            plt.show()

    # plot_training_curves(pretraining_history,
    #                      finetuning_history, baseline_history)
