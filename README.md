# PointNet

## Overview
Implementation of PointNet based architecture for classification and segmentation with point clouds.

`models.py` is where the model structures are defined. `train.py` loads data, trains models, logs trajectories and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain script to evaluate model accuracy and visualize segmentation result.


## Data Preparation
Download zip file (~2GB) from [here](https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing). Put the unzipped `data` folder under root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

## Classification Model
- Input: points clouds from across 3 classes (chairs, vases and lamps objects)

- Output: probability distribution indicating predicted classification (Dimension: Batch * Number of Classes)

Run `python train.py --task cls` to train the model, and `python eva_cls.py` for evaluation.

#### Test Accuracy : __97.58%__

#### Results

- Correct Classications/Predictions

- Incorrect Classications/Predictions

#### Interpretation
The overall performance of the classification model is commendable, especially in accurately identifying chairs where it exhibits a high level of precision. 
Nevertheless, there are instances where the model tends to misclassify certain samples, particularly confusing between lamps and vases. 
This suggests that the model encounters challenges distinguishing between classes with similar structural characteristics.


## Segmentation Model
- Input: points of chair objects (6 semantic segmentation classes) 

- Output: segmentation of points (Dimension: Batch * Number of Points per Object * Number of Segmentation Classes)

Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gif's, one for ground truth and the other for model prediction.

#### Test Accuracy : __88.52%__


#### Results

- Good Segmentations/Predictions

- Bad Segmentations/Predictions

#### Interpretation

The model excels in accurately segmenting the various components of a well defined conventional chairs, especially where the different segments like chair legs, backrests, armrests, etc are distinguishably defined. 
However, its performance experiences a decline when faced with unconventional chair designs, particularly those that integrate the legs and base. 
In such instances, the model encounters challenges, often misidentifying a substantial part of the legs as seat.
