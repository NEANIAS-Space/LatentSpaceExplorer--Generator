# Data preparation
The preparation step transforms the original data into the NPY file format by constructing a multidimensional matrix W x H x C, where W and H are the original image dimensions and C is the chosen number of channels. Through the configuration file you can choose the number and the order of the channels in the output file.

The script supports the following input formats: FITS, TIF, PNG, JPG, JPEG, NPY.

## FITS
In the config file set "format" to "fits"

The FITS format requires that the image be represented as a folder and each channel be separated into a separate FITS file.

```
data\
    input\
        image1\
            channel1.fits
            channel2.fits
            channel3.fits
            channel4.fits
            ...
        ...
```

## TIF
In the config file set "format" to "tif"

```
data\
    input\
        image1.tif
        image2.tif
        image3.jpg
        ...
```

## PNG, JPG, JPEG
In the config file set "format" to "image"

The TIF, PNG, JPG, JPEG formats requires that the image be represented as a file with a variable number of channels.
The FITS format requires that the image be represented as a folder and each channel be separated into a separate FITS file.

```
data\
    input\
        image1.jpg
        image2.jpg
        image3.jpg
        ...
```

## NPY
In the config file set "format" to "numpy"

If you provide the NPY format as input we assume that it is already formatted as expected.

```
data\
    input\
        image1.npy
        image2.npy
        image3.npy
        ...
```
