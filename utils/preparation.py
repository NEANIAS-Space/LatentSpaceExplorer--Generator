import os
import json
import numpy as np
from tqdm import tqdm


from utils.converters.FitsDataGenerator import FitsDataGenerator
from utils.converters.ImageDataGenerator import ImageDataGenerator


def preparation(data_dir, image_format, channels_map):

    data_fits_dir = os.path.join(data_dir, 'fits')
    data_images_dir = os.path.join(data_dir, 'images')
    data_numpy_dir = os.path.join(data_dir, 'numpy')
    metadata_path = os.path.join(data_numpy_dir, 'metadata.json')

    numpy_files = os.listdir(data_numpy_dir)
    numpy_files = [file for file in numpy_files if file.endswith('.npy')]

    # Check if numpy folder is empty
    if numpy_files:
        print('Detected files inside {} directory'.format(data_numpy_dir))

        # Check if preparation metadata file exist
        if os.path.isfile(metadata_path):
            print('Detected a metadata file')

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Check if preparation variables are the same
            if metadata['format'] == image_format and metadata['channels_map'] == channels_map:
                print(
                    'Detected the same preparation config. Skipping preparation step...')
                return

    # Remove old files from the numpy folder
    for file in numpy_files:
        os.remove(os.path.join(data_numpy_dir, file))

    # Remove old preparation metadata file
    if os.path.isfile(metadata_path):
        os.remove(metadata_path)

    # Process fits files
    if image_format == "fits":
        print('Converting fits files into numpy files...')

        if not os.path.isdir(data_fits_dir):
            os.makedirs(data_fits_dir)
            print('You have to put your fits files in the folder {}'.format(
                data_fits_dir))
            exit()

        data = FitsDataGenerator(data_fits_dir, channels_map)

        for item in tqdm(data):
            image_name, image = item
            image_path = os.path.join(data_numpy_dir, image_name)
            np.save(image_path, image)

    # Process image files
    elif image_format == "image":
        print('Converting image files into numpy files...')

        if not os.path.isdir(data_images_dir):
            os.makedirs(data_images_dir)
            print('You have to put your image files in the folder {}'.format(
                data_images_dir))
            exit()

        data = ImageDataGenerator(data_images_dir)

        for item in tqdm(data):
            image_name, image = item
            image_path = os.path.join(data_numpy_dir, image_name)
            np.save(image_path, image)

    # Save preparation metadata
    metadata = {
        "format": image_format,
        "channels_map": channels_map
    }

    with open(metadata_path, 'w+') as f:
        json.dump(metadata, f)
