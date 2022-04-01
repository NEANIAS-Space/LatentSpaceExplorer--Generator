# Data preparation
The preparation step transforms the original data into the NPY file format by constructing a multidimensional matrix W x H x C, where W and H are the original image dimensions and C is the chosen number of channels. Through the configuration file you can choose the number and the order of the channels in the output file.

The LSG script supports the following input formats: FITS, TIF, PNG, JPG, JPEG, NPY.

## FITS
The FITS format requires that the image be represented as a folder and each channel be separated into a separate FITS file.

```
data\
    image1\
        channel1.fits
        channel2.fits
        channel3.fits
        channel4.fits
        ...
    ...
```

## TIF, PNG, JPG, JPEG
The TIF, PNG, JPG, JPEG formats requires that the image be represented as a file with a variable number of channels.
The PNG, JPG, JPEG formats support 1 channel (grayscale) or 3 channels (RBG). The TIF format support a variable number of channels.

```
data\
    image1.jpg
    image2.jpg
    image3.jpg
    ...
```

## NPY
If you provide the NPY format as input we assume that it is already formatted as expected.

```
data\
    image1.npy
    image2.npy
    image3.npy
    ...
```
