# Learning human-environment interactions using conformal tactile textiles

Yiyue Luo, Yunzhu Li, Pratyusha Sharma, Wan Shou, Kui Wu, Michael Foshey, Beichen Li,
Tomás Palacios, Antonio Torralba, Wojciech Matusik

**Nature Electronics 2021**
[[website]](http://senstextile.csail.mit.edu/)

## Introduction

This is a PyTorch-based implementation for self-supervised sensing correction, classification and human pose prediction in the paper "Learning human-environment interactions using conformal tactile textiles".

## Self-Supervised Sensing Correction

#### Generate demos using pretrained weights

#### Training

#### Testing

## Human Pose Estimation

#### Generate demos using pretrained weights

#### Training

#### Testing

## Classification

#### Data preparation
1. You will need to download the data from the link: [[DropBox]](https://www.dropbox.com/s/vp5q6v85w14844v/data_classification.zip?dl=0) (451.2 MB)
2. Uncompress the data and place them according to the following structure
```
classification/
|--data_classification/
|    |--glove_objclaassification_26obj/
|    |--sock_classification/
|    |--vest_classification/
|    |--vest_letter/
|--letter_classification/
...
```

#### Letter classification using the vest
```
cd classification/letter_classification
bash scripts/train.sh
```
Results in the form of confusion matrix are stored in `classification/letter_classification/dump*`.

#### Action classification using the sock
```
cd classification/sock_classification
bash scripts/train.sh
```
Results in the form of confusion matrix are stored in `classification/sock_classification/dump*`.

#### Action classification using the vest
```
cd classification/vest_classification
bash scripts/train.sh
```
Results in the form of confusion matrix are stored in `classification/vest_classification/dump*`.

#### Object classification using the glove
```
cd classification/object_classification
bash scripts/train_26obj.sh
```
Results in the form of confusion matrix are stored in `classification/object_classification/dump*`.




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

### Pose Prediction
```
Pose Prediction
|--smpl
|    |--verts.py
|    |--serialization.py
|    |--render_smpl.py
|    |--posemapper.py
|    |--lbs.py
|    |--models
|--chkpts
|    |-model_in_paper
|    |    |--checkpoint.pth.tar
|--train.py
|--models.py
|--dataloader.py
|--data_preprocessing.py
|--test_visualize.py
|--utils
|    |--rotation_matrix.py
|    |--transformations.py
```

