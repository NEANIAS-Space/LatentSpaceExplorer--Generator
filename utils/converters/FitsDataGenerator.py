import os
import numpy as np
from astropy.io import fits


class FitsDataGenerator:
    def __init__(self, input_dir, channels_map):
        self.channels_map = channels_map

        # get all files from input folder
        self.fits_paths = [
            os.path.join(input_dir, dir)
            for dir in os.listdir(input_dir)
        ]
        # fits files should be a folder
        self.fits_paths = [
            dir for dir in self.fits_paths
            if os.path.isdir(dir)
        ]

    def __iter__(self):
        self.fits_id = 0
        return self

    def __len__(self):
        return len(self.fits_paths)

    def __next__(self):
        if self.fits_id >= len(self.fits_paths):
            raise StopIteration

        fits_path = self.fits_paths[self.fits_id]
        # get all image channels from fits folder
        fits_channels = [
            os.path.join(fits_path, dir)
            for dir in os.listdir(fits_path)
        ]
        # fits channels should be a fits file
        fits_channels = [
            fits_channel
            for fits_channel in fits_channels
            if fits_channel.endswith('.fits')
        ]

        image = np.empty(
            len(set(self.channels_map.values())), dtype=list)

        for fits_channel in fits_channels:
            for channel_map in self.channels_map.keys():

                if channel_map in os.path.basename(fits_channel):
                    hdul = fits.open(fits_channel)
                    matrix = hdul[0].data.astype('float32')

                    channel_index = self.channels_map[channel_map]
                    image[channel_index] = matrix

        image = np.dstack(image)
        image_name = '{}.npy'.format(os.path.basename(fits_path))

        self.fits_id += 1

        return image_name, image
