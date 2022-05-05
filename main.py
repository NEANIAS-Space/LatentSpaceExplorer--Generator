import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pandas_profiling import ProfileReport


from utils.preparation import preparation
from utils.preprocessing import tf_numpy_load, tf_preprocessing
from utils.augmentation import tf_augmentation

from architectures.cae import CAE
from architectures.scae import SCAE
from architectures.cvae import CVAE
from architectures.simclr import SimCLR


print('TensorFlow version: ', tf.__version__)
print('Devices', tf.config.list_physical_devices())


# Structure
DATA_DIR = os.path.join(os.getcwd(), 'data')
DATA_OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
INFERENCE_DIR = os.path.join(os.getcwd(), 'inference')

# Config file
CONFIG_FILE = os.path.join(os.getcwd(), 'config.json')
if os.path.isfile(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        CONFIG = json.load(f)
else:
    print('You have to put the configuration file in the folder {}'.format(CONFIG_FILE))
    exit()


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
            IMAGE_FORMAT = CONFIG[exp_id]['image']['format']
            IMAGE_DIM = CONFIG[exp_id]['image']['dim']
            CHANNELS_MAP = CONFIG[exp_id]['image']['channels']['map']
            CHANNELS_NUM = len(set(CHANNELS_MAP.values()))
            SPLIT_THRESHOLD = CONFIG[exp_id]['dataset']['split_threshold']
            NORMALIZATION_TYPE = CONFIG[exp_id]['preprocessing']['normalization_type']
            AUGMENTATION_THRESHOLD = CONFIG[exp_id]['augmentation']['threshold']
            AUGMENTATION_FLIP_X = CONFIG[exp_id]['augmentation']['flip_x']
            AUGMENTATION_FLIP_Y = CONFIG[exp_id]['augmentation']['flip_y']
            AUGMENTATION_ROTATE = CONFIG[exp_id]['augmentation']['rotate']['enabled']
            AUGMENTATION_ROTATE_DEGREES = CONFIG[exp_id]['augmentation']['rotate']['degrees']
            AUGMENTATION_SHIFT = CONFIG[exp_id]['augmentation']['shift']['enabled']
            AUGMENTATION_SHIFT_PERCENTAGE = CONFIG[exp_id]['augmentation']['shift']['percentage']
            ARCHITECTURE = CONFIG[exp_id]['architecture']['name']
            FILTERS = CONFIG[exp_id]['architecture']['filters']
            LATENT_DIM = CONFIG[exp_id]['architecture']['latent_dim']
            EPOCHS = CONFIG[exp_id]['training']['epochs']
            BATCH_SIZE = CONFIG[exp_id]['training']['batch_size']
            OPTIMIZER = CONFIG[exp_id]['training']['optimizer']['name']
            LEARNING_RATE = CONFIG[exp_id]['training']['optimizer']['learning_rate']
            LOSS = CONFIG[exp_id]['training']['loss']

            experiment_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_dir = "{}-{}".format(experiment_dir, EXPERIMENT_NAME)

            # Data preparation
            print('Data preparation...')
            preparation(DATA_DIR, IMAGE_FORMAT, CHANNELS_MAP)

            # Dataset
            pattern = os.path.join(DATA_OUTPUT_DIR, '*.npy')
            dataset = tf.data.Dataset.list_files(pattern)

            dataset = dataset.map(
                lambda file: tf_numpy_load(file),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            # Preprocessing
            dataset = dataset.map(
                lambda image: tf_preprocessing(
                    image,
                    tf.constant(IMAGE_DIM, tf.uint16),
                    tf.constant(NORMALIZATION_TYPE, tf.string)
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            length = tf.data.experimental.cardinality(dataset).numpy()
            print(f'Dataset: {length}')

            # Split
            index = round(length * SPLIT_THRESHOLD)
            train_set = dataset.take(index)
            test_set = dataset.skip(index + 1)

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
                        lambda: tf_augmentation(
                            images,
                            tf.constant(AUGMENTATION_FLIP_X, tf.bool),
                            tf.constant(AUGMENTATION_FLIP_Y, tf.bool),
                            tf.constant(AUGMENTATION_ROTATE, tf.bool),
                            tf.constant(
                                AUGMENTATION_ROTATE_DEGREES, tf.float32),
                            tf.constant(AUGMENTATION_SHIFT, tf.bool),
                            tf.constant(
                                AUGMENTATION_SHIFT_PERCENTAGE, tf.float32)
                        ),
                        lambda: images
                    ),
                num_parallel_calls=tf.data.AUTOTUNE
            )

            train_set = train_set.prefetch(buffer_size=tf.data.AUTOTUNE)
            test_set = test_set.prefetch(buffer_size=tf.data.AUTOTUNE)

            # Model
            if ARCHITECTURE == "cae":
                model = CAE(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS
                )

                model.compile(
                    optimizer=OPTIMIZER,
                    learning_rate=LEARNING_RATE,
                    loss=LOSS
                )

            elif ARCHITECTURE == "scae":
                model = SCAE(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS
                )

                model.compile(
                    optimizer=OPTIMIZER,
                    learning_rate=LEARNING_RATE,
                    loss=LOSS
                )

            elif ARCHITECTURE == "cvae":
                model = CVAE(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS
                )

                model.compile(
                    optimizer=OPTIMIZER,
                    learning_rate=LEARNING_RATE,
                    loss=LOSS
                )

            elif ARCHITECTURE == "simclr":
                model = SimCLR(
                    image_dim=IMAGE_DIM,
                    channels_num=CHANNELS_NUM,
                    latent_dim=LATENT_DIM,
                    filters=FILTERS,
                    temperature=0.1
                )

                model.compile(
                    optimizer=OPTIMIZER,
                    learning_rate=LEARNING_RATE,
                )

            # Create model dir
            model_dir = os.path.join(MODELS_DIR, experiment_dir)
            os.makedirs(model_dir)

            # Save experiement config
            experiment_config = os.path.join(
                MODELS_DIR, experiment_dir, 'config.json')
            with open(experiment_config, 'w+') as f:
                json.dump(CONFIG[exp_id], f, indent=4)

            # Set logger
            log_dir = os.path.join(LOGS_DIR, experiment_dir)
            summary_writer = tf.summary.create_file_writer(log_dir)

            for epoch in tqdm(range(EPOCHS)):

                # Training
                for batch, train_batch in enumerate(train_set):
                    model.train_step(train_batch)

                # Test
                for batch, test_batch in enumerate(test_set):
                    model.test_step(test_batch)

                # Save best model
                if epoch > (EPOCHS / 2):
                    model.save_best_model(model_dir)

                # Log
                with summary_writer.as_default():
                    model.log(epoch, train_batch, test_batch)

                # Reset losses
                model.reset_losses_state()

    elif args.step == 'inference':
        print('Inference...')

        if not args.experiment:
            print('--experiment argument is required')
            exit()

        experiment = args.experiment
        experiment_dir = os.path.join(INFERENCE_DIR, experiment)
        images_dir = os.path.join(experiment_dir, 'images')
        generated_dir = os.path.join(experiment_dir, 'generated')
        reductions_dir = os.path.join(experiment_dir, 'reductions')
        clusters_dir = os.path.join(experiment_dir, 'clusters')
        metadata_path = os.path.join(experiment_dir, 'metadata.json')
        embeddings_path = os.path.join(experiment_dir, 'embeddings.json')
        labels_path = os.path.join(experiment_dir, 'labels.json')

        os.makedirs(experiment_dir)
        os.makedirs(images_dir)
        os.makedirs(generated_dir)
        os.makedirs(reductions_dir)
        os.makedirs(clusters_dir)

        experiment_config_path = os.path.join(
            MODELS_DIR, experiment, 'config.json')
        with open(experiment_config_path) as f:
            experiment_config = json.load(f)

        IMAGE_DIM = experiment_config['image']['dim']
        NORMALIZATION_TYPE = experiment_config['preprocessing']['normalization_type']
        SAVE_GENERATED_IMAGES = experiment_config['inference']['save_generated_images']
        GENERATE_EMBEDDINGS_REPORT = experiment_config['inference']['generate_embeddings_report']

        pattern = os.path.join(DATA_OUTPUT_DIR, '*.npy')
        dataset = tf.data.Dataset.list_files(pattern, shuffle=False)

        dataset = dataset.map(
            lambda file: (file, tf_numpy_load(file)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = dataset.map(
            lambda file, image: (
                file,
                tf_preprocessing(
                    image,
                    tf.constant(IMAGE_DIM, tf.uint16),
                    tf.constant(NORMALIZATION_TYPE, tf.string)
                )
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        length = tf.data.experimental.cardinality(dataset).numpy()
        print(f'Dataset: {length}')

        model_path = os.path.join(MODELS_DIR, experiment)
        model = tf.keras.models.load_model(model_path)

        embeddings = []
        ids = []

        for i, data in enumerate(tqdm(dataset)):
            file, image = data

            image_name = os.path.basename(tf.compat.as_str_any(file.numpy()))
            image_name = '{}.png'.format(os.path.splitext(image_name)[0])
            image_path = os.path.join(images_dir, image_name)
            generated_path = os.path.join(generated_dir, image_name)

            channels = tf.unstack(image, axis=2)

            r = channels[
                experiment_config['image']['channels']['preview']['r']
            ]
            g = channels[
                experiment_config['image']['channels']['preview']['g']
            ]
            b = channels[
                experiment_config['image']['channels']['preview']['b']
            ]

            channels = tf.stack([r, g, b], axis=2)

            tf.keras.utils.save_img(
                path=image_path,
                x=channels.numpy(),
                data_format="channels_last"
            )

            image = tf.expand_dims(image, 0)

            embedding = model.predict(image)[0]
            embeddings.append(embedding.tolist())

            ids.append(image_name)

            if SAVE_GENERATED_IMAGES:
                generated = model(image, training=True)[0]

                tf.keras.utils.save_img(
                    path=generated_path,
                    x=generated.numpy(),
                    data_format="channels_last"
                )

        with open(metadata_path, 'w+') as f:
            json.dump(experiment_config, f)

        with open(embeddings_path, 'w+') as f:
            json.dump(embeddings, f)

        labels = {"columns": [], "index": [], "data": []}

        labels_path = os.path.join(DATA_DIR, 'labels.json')
        if os.path.isfile(labels_path):
            with open(labels_path) as f:
                labels = json.load(f)

        labels['columns'] = ids

        labels_path = os.path.join(experiment_dir, 'labels.json')
        with open(labels_path, 'w+') as f:
            json.dump(labels, f)

        clusters_gitkeep_path = os.path.join(clusters_dir, ".gitkeep")
        with open(clusters_gitkeep_path, 'w+') as f:
            pass

        reductions_gitkeep_path = os.path.join(reductions_dir, ".gitkeep")
        with open(reductions_gitkeep_path, 'w+') as f:
            pass

        if GENERATE_EMBEDDINGS_REPORT:
            dataframe = pd.DataFrame(embeddings)

            pandas_profiler_config_path = os.path.join(
                os.getcwd(), "pandas_profiler_config.yaml")
            profile = ProfileReport(
                dataframe, config_file=pandas_profiler_config_path)

            embeddings_report_path = os.path.join(
                experiment_dir, "embeddings_report.html")
            profile.to_file(embeddings_report_path)
