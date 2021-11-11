import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import datetime
import argparse
from tqdm import tqdm
import tensorflow as tf

from utils.images import tf_numpy_load, tf_preprocessing, tf_augmentation
from architectures.cae import ConvolutionalAutoencoder
from architectures.cvae import ConvolutionalVariationalAutoencoder
from architectures.simclr import SimCLR


# Structure
DATA_DIR = os.path.join(os.getcwd(), 'data')
DATA_NUMPY_DIR = os.path.join(DATA_DIR, 'numpy')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
INFERENCE_DIR = os.path.join(os.getcwd(), 'inference')

# Config file
CONFIG_FILE = os.path.join(os.getcwd(), 'config.json')
with open(CONFIG_FILE) as f:
    CONFIG = json.load(f)


parser = argparse.ArgumentParser()
parser.add_argument('step', choices=['training', 'inference'])
parser.add_argument('--experiment', action='store', type=str, default=None)
args = parser.parse_args()


if __name__ == "__main__":

    if args.step == 'training':
        print('Training...')
        print(f'Detected {len(CONFIG)} experiments from the config file')

        for exp_id, experiment in enumerate(CONFIG):
            print(f'Starting experiment {exp_id + 1}')

            # Experiment config
            EXPERIMENT_NAME = CONFIG[exp_id]['name']
            IMAGE_DIM = CONFIG[exp_id]['image']['dim']
            CHANNELS_NUM = CONFIG[exp_id]['image']['channels']
            SPLIT_THRESHOLD = CONFIG[exp_id]['dataset']['split_threshold']
            AUGMENTATION_THRESHOLD = CONFIG[exp_id]['dataset']['augmentation_threshold']
            ARCHITECTURE = CONFIG[exp_id]['architecture']['name']
            FILTERS = CONFIG[exp_id]['architecture']['filters']
            LATENT_DIM = CONFIG[exp_id]['architecture']['latent_dim']
            EPOCHS = CONFIG[exp_id]['training']['epochs']
            BATCH_SIZE = CONFIG[exp_id]['training']['batch_size']
            OPTIMIZER = CONFIG[exp_id]['training']['optimizer']
            LOSS = CONFIG[exp_id]['training']['loss']

            experiment_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_dir = "{}-{}".format(experiment_dir, EXPERIMENT_NAME)

            # Dataset
            pattern = os.path.join(DATA_NUMPY_DIR, '*.npy')
            dataset = tf.data.Dataset.list_files(pattern)

            dataset = dataset.map(
                lambda file: tf_numpy_load(file),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Preprocessing
            dataset = dataset.map(
                lambda image: tf_preprocessing(image, tf.constant(IMAGE_DIM)),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            length = tf.data.experimental.cardinality(dataset).numpy()
            print(f'Dataset: {length}')

            # Split
            train_set = dataset.take(round(length * SPLIT_THRESHOLD))
            test_set = dataset.skip(round(length * SPLIT_THRESHOLD))

            print('Training set: {}'.format(
                tf.data.experimental.cardinality(train_set).numpy()))
            print('Test set: {}'.format(
                tf.data.experimental.cardinality(test_set).numpy()))

            train_set = train_set.cache()
            test_set = test_set.cache()

            train_set = train_set.shuffle(
                len(train_set)).batch(BATCH_SIZE)
            test_set = test_set.batch(BATCH_SIZE)

            # Augmentation
            train_set = train_set.map(
                lambda images:
                    tf.cond(
                        tf.random.uniform([], 0, 1) < AUGMENTATION_THRESHOLD,
                        lambda: tf_augmentation(images), lambda: images
                    ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            train_set = train_set.prefetch(buffer_size=tf.data.AUTOTUNE)
            test_set = test_set.prefetch(buffer_size=tf.data.AUTOTUNE)

            # Model
            if ARCHITECTURE == "cae":
                model = ConvolutionalAutoencoder(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS,
                    optimizer=OPTIMIZER,
                    loss=LOSS
                )

            elif ARCHITECTURE == "cvae":
                model = ConvolutionalVariationalAutoencoder(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS,
                    optimizer=OPTIMIZER,
                    loss=LOSS
                )

            elif ARCHITECTURE == "simclr":
                model = SimCLR(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS,
                    optimizer=OPTIMIZER
                )

            log_dir = os.path.join(LOGS_DIR, experiment_dir)
            summary_writer = tf.summary.create_file_writer(log_dir)

            for epoch in tqdm(range(EPOCHS)):

                # Training
                for batch, train_batch in enumerate(train_set):
                    model.train_step(train_batch)

                # Test
                for batch, test_batch in enumerate(test_set):
                    model.test_step(test_batch)

                with summary_writer.as_default():
                    model.log(
                        tf.constant(epoch, dtype=tf.int64),
                        train_batch, test_batch
                    )

            model_dir = os.path.join(MODELS_DIR, experiment_dir)
            # model_architecture = os.path.join(
            #     MODELS_DIR, experiment_dir, 'model.png')
            experiment_config = os.path.join(
                MODELS_DIR, experiment_dir, 'config.json')

            model.save(model_dir)
            # tf.keras.utils.plot_model(
            #     model, to_file=model_architecture, show_shapes=True, expand_nested=True)
            print('Model saved')

            with open(experiment_config, 'w+') as f:
                json.dump(CONFIG[exp_id], f, indent=2)
            print('Experiment config saved')

    elif args.step == 'inference':
        print('Inference...')

        if not args.experiment:
            print('--experiment argument is required')
            exit()

        experiment = args.experiment
        experiment_dir = os.path.join(INFERENCE_DIR, experiment)
        images_dir = os.path.join(experiment_dir, 'images')
        embeddings_path = os.path.join(experiment_dir, 'embeddings.json')
        os.makedirs(experiment_dir)
        os.makedirs(images_dir)

        experiment_config_file = os.path.join(
            MODELS_DIR, experiment, 'config.json')
        with open(experiment_config_file) as f:
            experiment_config = json.load(f)

        image_dim = experiment_config['image']['dim']

        pattern = os.path.join(DATA_NUMPY_DIR, '*.npy')
        dataset = tf.data.Dataset.list_files(pattern, shuffle=False)

        dataset = dataset.map(
            lambda file: (file, tf_numpy_load(file)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.map(
            lambda file, image:
                (file, tf_preprocessing(image, tf.constant(image_dim))),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        length = tf.data.experimental.cardinality(dataset).numpy()
        print(f'Dataset: {length}')

        model_path = os.path.join(MODELS_DIR, experiment)
        model = tf.keras.models.load_model(model_path)

        embeddings = []

        for file, image in tqdm(dataset):
            image_name = os.path.basename(tf.compat.as_str_any(file.numpy()))
            image_name = '{}.png'.format(os.path.splitext(image_name)[0])
            image_path = os.path.join(images_dir, image_name)

            channels = tf.unstack(image, axis=2)

            r = channels[experiment_config['image']['preview']['r']]
            g = channels[experiment_config['image']['preview']['g']]
            b = channels[experiment_config['image']['preview']['b']]

            channels = tf.stack([r, g, b], axis=2)

            tf.keras.utils.save_img(
                path=image_path,
                x=channels.numpy(),
                data_format="channels_last"
            )

            image = tf.expand_dims(image, 0)

            embedding = model.predict(image)[0]
            embeddings.append(embedding.tolist())

        with open(embeddings_path, 'w+') as f:
            json.dump(embeddings, f)
