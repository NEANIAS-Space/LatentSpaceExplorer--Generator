import os
import json
import numpy as np
from tqdm import tqdm


from utils.converters.FitsDataGenerator import FitsDataGenerator
from utils.converters.TifDataGenerator import TifDataGenerator
from utils.converters.ImageDataGenerator import ImageDataGenerator


def preparation(data_dir, image_format, channels_map):

    data_input_dir = os.path.join(data_dir, 'input')
    data_output_dir = os.path.join(data_dir, 'output')
    metadata_path = os.path.join(data_output_dir, 'metadata.json')

    numpy_files = os.listdir(data_output_dir)
    numpy_files = [file for file in numpy_files if file.endswith('.npy')]

    # Check if output folder is empty
    if numpy_files:
        print('Detected files inside {} directory'.format(data_output_dir))

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

    # Remove old files from the output folder
    for file in numpy_files:
        os.remove(os.path.join(data_output_dir, file))

    # Remove old preparation metadata file
    if os.path.isfile(metadata_path):
        os.remove(metadata_path)

    # Check if input folder exists
    if not os.path.isdir(data_input_dir):
        os.makedirs(data_input_dir)
        print('You have to put your files in the folder {}'.format(
            data_input_dir))
        exit()

    # Process fits files
    if image_format == "fits":
        print('Converting fits files into numpy files...')

        data = FitsDataGenerator(data_input_dir, channels_map)

        for item in tqdm(data):
            image_name, image = item
            image_path = os.path.join(data_output_dir, image_name)
            np.save(image_path, image)

    # Process image files
    elif image_format == "tif":
        print('Converting tif files into numpy files...')

        data = TifDataGenerator(data_input_dir, channels_map)

        for item in tqdm(data):
            image_name, image = item
            image_path = os.path.join(data_output_dir, image_name)
            np.save(image_path, image)

    # Process image files
    elif image_format == "image":
        print('Converting image files into numpy files...')

        data = ImageDataGenerator(data_input_dir, channels_map)

        for item in tqdm(data):
            image_name, image = item
            image_path = os.path.join(data_output_dir, image_name)
            np.save(image_path, image)

    # Process numpy files
    elif image_format == "numpy":
        print('Copying numpy files...')

        data = [
            (numpy_file, np.load(os.path.join(data_input_dir, numpy_file)))
            for numpy_file in os.listdir(data_input_dir)
            if numpy_file.endswith('.npy')
        ]

        for item in tqdm(data):
            image_name, image = item
            image_path = os.path.join(data_output_dir, image_name)
            np.save(image_path, image)

    # Save preparation metadata
    metadata = {
        "format": image_format,
        "channels_map": channels_map
    }

    with open(metadata_path, 'w+') as f:
        json.dump(metadata, f)
