# PointNet

## Overview
Implementation of PointNet-based architecture for classification and segmentation with point clouds.

`models.py` is where the model structures are defined. `train.py` loads data, trains models, logs trajectories, and saves checkpoints. `eval_cls.py` and `eval_seg.py` contain scripts to evaluate model accuracy and visualize segmentation results.


## Data Preparation
Download the zip file (~2GB) from [here](https://drive.google.com/file/d/1wXOgwM_rrEYJfelzuuCkRfMmR0J7vLq_/view?usp=sharing). Put the unzipped `data` folder under the root directory. There are two folders (`cls` and `seg`) corresponding to two tasks, each of which contains `.npy` files for training and testing.

## Classification Model
- Input: points clouds from across 3 classes (chairs, vases, and lamps objects)

- Output: probability distribution indicating predicted classification (Dimension: Batch * Number of Classes)

Run `python train.py --task cls` to train the model, and `python eva_cls.py` for evaluation.

#### Test Accuracy : __97.58%__

#### Results

- Correct Classifications/Predictions
  
| Class | Ground Truth | Prediction | Ground Truth | Prediction | Ground Truth | Prediction |
|-------|--------------|------------|--------------|------------|--------------|------------|
| Chair | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/75_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/75_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/125_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/125_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/435_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/435_chair.gif" alt="Input RGB" width="200"/> |
| Vase  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/640_vase.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/655_vase.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/655_vase.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/705_vase.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/705_vase.gif" alt="Input RGB" width="200"/>  |
| Lamp  |  <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/790_lamp.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/790_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/880_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/880_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/910_lamp.gif" alt="Input RGB" width="200"/>  |  <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/910_lamp.gif" alt="Input RGB" width="200"/>  |

- Incorrect Classifications/Predictions

| Class | Ground Truth | Prediction |
|-------|--------------|------------|
| Chair | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/595_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/595_lamp.gif" alt="Input RGB" width="200"/> |
| Vase  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/620_vase.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/620_lamp.gif" alt="Input RGB" width="200"/>  | 
| Lamp  |  <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/gt/750_lamp.gif" alt="Input RGB" width="200"/>  | <img src="https://github.com/kirangit27/PointNet/blob/master/output/cls/pred/750_vase.gif" alt="Input RGB" width="200"/>  |  |


#### Interpretation
The overall performance of the classification model is commendable, especially in accurately identifying chairs where it exhibits a high level of precision. 
Nevertheless, there are instances where the model tends to misclassify certain samples, particularly confusing between lamps and vases. 
This suggests the model encounters challenges distinguishing between classes with similar structural characteristics.


## Segmentation Model
- Input: points of chair objects (6 semantic segmentation classes)
- Output: segmentation of points (Dimension: Batch * Number of Points per Object * Number of Segmentation Classes)

Run `python train.py --task seg` to train the model. Running `python eval_seg.py` will save two gifs, one for ground truth and the other for model prediction.

#### Test Accuracy : __88.52%__


#### Results

- Good Segmentations/Predictions

| Ground Truth | Prediction | Ground Truth | Prediction |
|--------------|------------|--------------|------------|
| <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/75_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/75_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/125_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/125_chair.gif" alt="Input RGB" width="200"/> |
| <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/435_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/435_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/55_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/55_chair.gif" alt="Input RGB" width="200"/> |
| <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/260_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/260_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/10_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/10_chair.gif" alt="Input RGB" width="200"/> |


- Bad Segmentations/Predictions

| Ground Truth | Prediction |
|--------------|------------|
| <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/605_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/605_chair.gif" alt="Input RGB" width="200"/> |
| <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/140_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/140_chair.gif" alt="Input RGB" width="200"/> |
| <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/gt/335_chair.gif" alt="Input RGB" width="200"/> | <img src="https://github.com/kirangit27/PointNet/blob/master/output/seg/pred/335_chair.gif" alt="Input RGB" width="200"/> |

#### Interpretation

The model excels in accurately segmenting the various components of a well-defined conventional chair, especially where the different segments like chair legs, backrests, armrests, etc are distinguishably defined. 
However, its performance experiences a decline when faced with unconventional chair designs, particularly those that integrate the legs and base. 
In such instances, the model encounters challenges, often misidentifying a substantial part of the legs as the seat.


## Robustness Analysis

###  Rotating Pointclouds

Evaluating the classification and segmentation models on rotated point-clouds. Point-clouds are either rotated along all x,y, and z axes combined or individually along a single axis.

   ```
          def rotate_test_data(point_cloud, angles):
          # Convert angles to radians 
          alpha, beta, gamma = np.deg2rad(angles)

          # Define rotation matrices
          Rx = torch.Tensor([[1,         0,           0       ],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha),  np.cos(alpha)]]).to(args.device) 
          
          Ry = torch.Tensor([[ np.cos(beta), 0, np.sin(beta)], 
                            [     0,        1,       0     ], 
                            [-np.sin(beta), 0, np.cos(beta)]]).to(args.device)
          
          Rz = torch.Tensor([[np.cos(gamma), -np.sin(gamma), 0],
                            [np.sin(gamma),  np.cos(gamma), 0],
                            [      0,             0,        1]]).to(args.device)

          # Combine rotations
          R_xyz = torch.matmul(Rz, torch.matmul(Ry, Rx))

          rotated_point_cloud = torch.matmul(point_cloud, R_xyz)
          return rotated_point_cloud
