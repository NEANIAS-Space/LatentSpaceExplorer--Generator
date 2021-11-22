# Latent space generator

## General introduction

## Setup
1. Download or clone the code:

    `git clone git@gitlab.neanias.eu:s3-service/latent-space-explorer/generator.git`

2. Environment setup
    - pip env

        ```
        python -m venv .venv
        source /.venv/bin/activate
        pip install -r requirements.txt
        ```
    - docker image

        `docker run -it -p 6006:6006 --gpus=all -v $PWD:/workdir dr4thmos/lsg-gpu:0.1`

## Import data
1. Copy your sources into /data/input
    - For images, you could use extensions in [jpeg, jpg, png]
    - For fits, your sources need to be folders with inside a fits file per band (e.g. )
        ```    
        - /Source001
            - <survey_band1-name>.fits
            - <survey_band2-name>.fits
            - ...
        - /Source002
            - <survey_band1-name>.fits
            - <survey_band2-name>.fits
            - ...
        ```
    - For numpy, copy as they are
2. Describe your input in the config.json file image section

    ```
    "image": {
        "format": "fits",
        "dim": 32,
        "channels": {
        "map": {
            {
                "survey_band1-name": 2,
                "survey_band2-name": 0,
                "survey_band3-name": 1,
                "survey_band4-name": 3,
                "survey_band5-name": 4,
                "survey_band6-name": 4
            }
        },
            "preview": {
                "r": 4,
                "g": 0,
                "b": 2
            }
        }
    }
    ```
    - Detailed info in dedicated section

## Training
1. Fill the config file in the following parts: (more detail in specific section)
    ```
    "dataset": {
        "split_threshold": 0.95
    },
    "preprocessing": {
        "normalization_type": "snr"
    },
    "augmentation": {
        "threshold": 0.6,
        "flip_x": true,
        "flip_y": true,
        "rotate": {
            "enabled": true,
            "degrees": 45
        },
        "shift": {
            "enabled": true,
            "percentage": 10
        }
    },
    "architecture": {
        "name": "cae",
        "filters": [
            32,
            64,
            128,
            256
        ],
        "latent_dim": 64
    },
    "training": {
        "epochs": 10000,
        "batch_size": 32,
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.0003
        },
        "loss": "MeanSquaredError"
    }
    ```
2. Start the training:

    `python main.py training`

3. Start tensorboard to monitor the training:
    
    `tensorboard --logdir logs`

## Export
1. Fill the config file:
    
    ```
    "inference": {
        "save_generated_images": false
    }
    ```
2. Start the inference:
    `python main.py inference --experiment <experiment-folder>`

## Config templates
To make your life easier, we prepared some config files for each architecture available. More details in dedicated readme.

- Convolutional AutoEncoder
- Variational Convolutional AutoEncoder
- Simclr