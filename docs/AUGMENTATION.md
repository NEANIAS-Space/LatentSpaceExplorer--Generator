# Data augmentation

Fields of the config file you could manage:

- threshold: [0.0, 1.0] ; probability to perform augmentation at each epoch
- flip_x: [true, false] ; mirror image on x axis
- flip_y: [true, false] ; mirror image on y axis
- rotate: 
    - enabled: [true, false]
    - degrees: [0,365]
- shift:
    - enabled: [true, false]
    - percentage: [0,100]

# Custom augmentation

More augmentation methods could be added into the utils/augmentation.py file