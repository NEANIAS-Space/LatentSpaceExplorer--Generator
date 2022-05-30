# Latent space generator

This repository implements a Machine Learning pipeline that produces latent space to be explored in the projector https://lse.neanias.eu 

Here follows a quick start guide to go quick in a working example.

---

## Setup

1. Download or clone the code
    ```
    git clone git@gitlab.neanias.eu:s3-service/latent-space-explorer/generator.git
    ```

2. Navigate inside the folder
    ```
    cd generator
    ```

3. Environment setup

    - docker with GPU
        ```
        docker run -it --rm -p 6006:6006 --gpus=all -v $PWD:/workdir dr4thmos/hackthescience2022:1.0-gpu
        ```

    - docker without GPU
        ```
        docker run -it --rm -p 6006:6006 -v $PWD:/workdir dr4thmos/hackthescience2022:1.0-gpu
        ```

    - Your own environment could be setup by install requirements.txt
        ```
        pip install -r requirements.txt
        ```

---

## Import data

1. Copy your data into into `/data/input`

    - Follows [Data preparation](./docs/PREPARATION.md) for the folder structure

2. Describe your input in the config.json file image section
    - ```
        "image": {
            "format": "image",
            "dim": 28,
            "channels": {
            "map": {
                {
                    "grey": 0
                }
            },
                "preview": {
                    "r": 0,
                    "g": 0,
                    "b": 0
                }
            }
        }
        ```

    - ```
        "image": {
            "format": "fits",
            "dim": 64,
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

---

## Training

1. Rename config_template.json file to config.json or choose one of the prepared config files in order to start and copy it to the project root (near the `main.py` file)

    - [Convolutional Autoencoder](./docs/templates/cae.json) (name it config.json)
    - [SimClr](./docs/templates/simclr.json) (name it config.json)

2. Start the training

    ```
    python main.py training
    ```

3. Start tensorboard to monitor the training

    ```
    tensorboard --logdir logs
    ```

---

## Export to nextcloud

1. Start the inference on a specific model in the `/models` folder

    ```
    python main.py inference --experiment <experiment-folder>
    ```

2. Upload the folder created into nextcloud `/lse-<your-email>` shared folder (We suggest you to use the desktop application of nextcloud)


# Advanced manuals

Full detailed pipeline info will be available in specific readme:

-   [Data preparation](./docs/PREPARATION.md)
-   [Data preprocessing](./docs/PREPROCESSING.md)
-   [Data Augmentation](./docs/AUGMENTATION.md)
-   [Training](./docs/TRAINING.md)
-   [Inference](./docs/INFERENCE.md)