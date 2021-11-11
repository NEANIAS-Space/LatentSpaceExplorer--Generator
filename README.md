# Latent space generator

Use that repo to produce latent space to be explored in the projector lse.neanias.eu

## General introduction
The list of experiments to be runned need to be listed in config.json file. (detailed later)

When the config file is fullfilled run:

`python main.py training`

Training produce models and logs saved in the same name folders. Logs could be user in tensorboard to monitor the trainining by opening the browser at the link shown by:

`tensorboard --logdir logs`

To extract embeddings run:

`python main.py inference --experiment <model-name>`

The above commands refer to data stored in data/numpy. It's possible to work on preprocessing and augmentation of the data, detail follow in the data section

## How to run
To run the code install requirements listed in requirements.txt file with:

`pip install -r requirements.txt`

If you want to run it in an isolated environment follow 1 of below steps

### pipenv
```
python -m venv .venv
source /.venv/bin/activate
pip install -r requirements.txt
```

### docker container
`docker run -it -p 6006:6006 --gpus=all -v $PWD:/workdir gitlab.neanias.eu:5050/s3-service/latent-space-explorer/generator/lsg-gpu:0.1`

## Data
Data are intended to be produced by the repo "data-converter" as a folder containing a numpy file per source, in a structure like that:
- /data
    - channels.json -> probably it will be removed in a future release, because is used only to infer number of channels
    - /numpy
        - source0001.npy
        - source0002.npy
        - ...

Every source numpy file has a number of channels choosen in the previous step (fits -> numpy) referring to specific bands.

<b>Preprocessing</b> phase it's done in the pipeline, and the functions to be performed could be found in utils/images.py. Here it could be possible to add normalization, deal with NaN and so on.

In the same file it could be tuned the <b>augmentation</b> phase. (and in future could be added to the config.json)

## Experiment configuration
The config.json file have that structure:
```
[
  {
    "name": "default",
    "image": {
      "dim": 64
    },
    "dataset": {
      "split_threshold": 0.8,
      "augmentation_threshold": 0.5
    },
    "architecture": {
      "filters": [16, 32, 64],
      "latent_dim": 128
    },
    "training": {
      "epochs": 3,
      "batch_size": 32,
      "optimizer": "Adam",
      "loss": "MeanSquaredError"
    }
  }
]
```


## Architectures
In architectures folder there are the python files with proper classes for each model.

