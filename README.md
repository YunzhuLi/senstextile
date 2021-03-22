# Learning human-environment interactions using conformal tactile textiles

Yiyue Luo, Yunzhu Li, Pratyusha Sharma, Wan Shou, Kui Wu, Michael Foshey, Beichen Li,
Tomás Palacios, Antonio Torralba, Wojciech Matusik

**Nature Electronics 2021**
[[website]](http://senstextile.csail.mit.edu/)

## Introduction

This is a PyTorch-based implementation for self-supervised sensing correction, classification and human pose prediction in the paper "Learning human-environment interactions using conformal tactile textiles".

## Contents

- [Self-Supervised Sensing Correction](#self-supervised-sensing-correction)
- [Human Pose Estimation](#human-pose-estimation)
- [Classification](#classification)

## Self-Supervised Sensing Correction

#### Data and environment preparation
1. You will need to download the data from the link: [[DropBox]](https://www.dropbox.com/s/vp5q6v85w14844v/data_classification.zip?dl=0) (451.2 MB)
2. Uncompress the data and place them according to the following structure
```
sensing_correction/
|--data_sensing_correction/
|    |--vest_calibration/
|    |--visualization/
|--glove_withscale/
...
```
3. Setup the environmental variable
```
cd sensing_correction
export PYTHONPATH=${PYTHONPATH}:${PWD}
```

#### Calibrate the glove using the scale


#### Calibrate the sock using the scale


#### Calibrate the vest using a calibrated glove

1. Generate demo visualizations using pretrained models
```
cd sensing_correction/vest_withglove
bash scripts/eval.sh
```
Visualizations showing the side-by-side comparison between the raw signal and the calibrated results are stored in `sensing_correction/vest_withglove/dump_vest_calibration/vis*`.

2. Training the calibration model for the vest using a pretrained calibrated glove
```
cd sensing_correction/vest_withglove
bash scripts/calib.sh
```

#### Calibrate the kuka sleeve using a calibrated glove




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

