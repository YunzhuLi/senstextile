# Learning human-environment interactions using scalable functional textiles

## Introduction

This is a Pytorch-based code for self-supervised sensing correction, classification and mocap prediction in the paper "Learning human-environment interactions using scalable functional textiles".

## Code organization

Our code is consists of three sub-sections and organized as the following.

### Self-supervised sensing correction
```
calibration
|--glove_withscale
|    |--scripts
|    |    |--calib.sh
|    |    |--eval.sh
|    |--calib.py
|    |--config.py
|    |--data.py
|    |--eval.py
|--sock_withscale
|--vest_withglove
|--kuka_withglove
|
|--models.py
|--utils.py
|--visualizer.py
```

### Classification
```
classification
|--letter_classification
|    |--scripts
|    |    |--train.sh
|    |--config.py
|    |--data.py
|    |--models.py
|    |--train.py
|    |--utils.py
|--object_classification
|--sock_classification
|--vest_classification
|
|--models.py
|--utils.py
```

### Mocap prediction

