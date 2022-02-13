import os
import tifffile
import numpy as np


class TifDataGenerator:
    def __init__(self, input_dir, channels_map):

        self.channels_map = channels_map
        allowed_extensions = ['.tif']

        # get all images from input folder
        self.tif_paths = [
            os.path.join(input_dir, image)
            for image in os.listdir(input_dir)
            if image.endswith(tuple(allowed_extensions))
        ]
        # images files should be a file
        self.tif_paths = [
            image for image in self.tif_paths
            if os.path.isfile(image)
        ]

    def __iter__(self):
        self.image_id = 0
        return self

    def __len__(self):
        return len(self.tif_paths)

    def __next__(self):
        if self.image_id >= len(self.tif_paths):
            raise StopIteration

        image_path = self.tif_paths[self.image_id]

        image = np.empty(
            len(set(self.channels_map.values())), dtype=list)

        tif_channels = []

        with tifffile.TiffFile(image_path) as tif:
            tif_channels = np.array([page.asarray() for page in tif.pages])
            tif_channels = np.squeeze(tif_channels, axis=0)

        for channel_map in self.channels_map.keys():
            channel_index = self.channels_map[channel_map]

            image[int(channel_map)] = tif_channels[:, :,
                                                   channel_index].astype('float32')

        image = np.dstack(image)
        image_name = '{}.npy'.format(
            os.path.splitext(os.path.basename(image_path))[0])

        self.image_id += 1

        return image_name, image
