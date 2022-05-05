import os
from PIL import Image, ImageOps
import numpy as np


class ImageDataGenerator:
    def __init__(self, input_dir, channels_map):

        self.channels_map = channels_map
        allowed_extensions = ['.png', '.jpg', '.jpeg']

        # get all images from input folder
        self.images_paths = [
            os.path.join(input_dir, image)
            for image in os.listdir(input_dir)
            if image.endswith(tuple(allowed_extensions))
        ]
        # images files should be a file
        self.images_paths = [
            image for image in self.images_paths
            if os.path.isfile(image)
        ]

    def __iter__(self):
        self.image_id = 0
        return self

    def __len__(self):
        return len(self.images_paths)

    def __next__(self):
        if self.image_id >= len(self.images_paths):
            raise StopIteration

        image_path = self.images_paths[self.image_id]

        image = Image.open(image_path)

        if len(self.channels_map) == 3:
            image = image.convert('RGB')
            image = np.array(image, dtype=np.float32)

        elif len(self.channels_map) == 1:
            #image.convert('L')
            image = ImageOps.grayscale(image)
            image = np.array(image, dtype=np.float32)
            print(image.shape)
            image = np.expand_dims(image, axis=-1)
            print(image.shape)

        image_name = '{}.npy'.format(
            os.path.splitext(os.path.basename(image_path))[0])

        self.image_id += 1

        return image_name, image
