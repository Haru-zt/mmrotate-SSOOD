# mmrotate-SSOOD: Simplified Framework for Semi-Supervised Oriented Object Detection

## Introduction
**mmrotate-SSOOD** is a simplified, modular, and flexible framework designed specifically for **Semi-Supervised Oriented Object Detection (SSOOD)** tasks. 

## Key Features
Our framework offers the following advantages:

- **Simplified Implementation**: Implementing custom semi-supervised detection methods is straightforward. You only need to modify **two key functions**, allowing faster experimentation and development.
- **Flexible Data Augmentation**: Built upon MMCV, our framework supports seamless integration of custom data augmentation techniques. We also provide ready-to-use augmentation configs for fair comparisons with prior works.
- **Dataset Splitting Tools**: Easily split your dataset into labeled and unlabeled subsets using our user-friendly tools, saving time on data preparation for semi-supervised learning.
- **Extensible Method Support**: Currently, we support **SOOD(CVPR 2023)**(https://arxiv.org/abs/2304.04515) and **Dense Teacher(ECCV 2022)**(https://arxiv.org/abs/2207.02541), with plans to add more state-of-the-art semi-supervised learning methods in future updates.

## Future Methods Support
We plan to continually update this framework to include more state-of-the-art semi-supervised learning methods. Here are some of the methods we aim to support in future updates:
1. **Soft Teacher(ICCV 2021) (End-to-End Semi-Supervised Object Detection with Soft Teacher)**(https://arxiv.org/abs/2106.09018)
   A famous semi-supervised object detection method that uses teacher-student architecture with pseudo-label refinement for better performance.

2. **ARSL(CVPR 2023) (Ambiguity-Resistant Semi-Supervised Learning for Dense Object Detection)**(https://arxiv.org/abs/2303.14960)
   Focuses on ambiguities in semi-supervised object detection.

## Requirements

To ensure compatibility, please install the following dependencies:

### 1. PyTorch
- **PyTorch**: `1.13.x`  
  We recommend PyTorch `1.13.x` as all modules have been tested with this version. Installation guide: [PyTorch.org](https://pytorch.org/get-started/locally/)

### 2. MMDetection
- **MMDetection**: `3.0.0`  
  MMDetection serves as the base object detection framework. Refer to the [MMDetection documentation](https://mmdetection.readthedocs.io/en/latest/) for installation instructions.

### 3. MMPretrain
- **MMPretrain**: `1.1.0`  
  MMPretrain is used for pretraining models. Please follow the [MMPretrain installation guide](https://mmpretrain.readthedocs.io/en/latest/get_started.html).

### Notes:
- **CUDA Compatibility**: Make sure all dependencies match your system's CUDA version for proper GPU acceleration. Check the [PyTorch documentation](https://pytorch.org/get-started/locally/) for compatibility.
- **Virtual Environment**: For a cleaner setup, we highly recommend using a virtual environment like `conda` or `venv`.

### Installation Example:
Here’s a quick guide to set up the environment:
```bash
# Create a virtual environment
conda create -n ssood python=3.10
conda activate ssood

# Install PyTorch

# Install mmdet
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet==3.0.0

# Install mmpretrain
mim install mmpretrain==1.1.0
pip install future tensorboard
pip install -v -e .
```

## Data Preparation

Please refer to [data_preparation.md](tools/data/dota/README.md) to prepare the original data. After that, the data folder should be organized as follows:

```
├── data
│   ├── split_ss_dota1_5
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

For partial labeled setting, we split the DOTA-v1.5's train set via the author released [split data list](tools/SSOD/split_dota1.5_lists) and [split tool](tools/SSOD/split_dota1.5_via_lists.py):

```angular2html
python tools/SOOD/split_dota1.5_via_lists.py
```

For fully labeled setting, we use DOTA-V1.5 train as labeled set and DOTA-V1.5 test as unlabeled set.

After that, the data folder should be organized as follows:

```
├── data
│   ├── split_ss_dota1_5
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_10_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_10_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_20_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_20_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_30_labeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── train_30_unlabeled
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── val
│   │   │   ├── images
│   │   │   ├── annfiles
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── annfiles
```

## Results

DOTA1.5

SOOD

|         Backbone         | Setting | mAP50 | mAP50 in Paper | Mem (GB) |                              Config                              |
| :----------------------: | :-----: | :---: | :-------------: | :------: | :-------------------------------------------------------------: |
| ResNet50 (1024,1024,200) |   10%   | 47.93 |      48.63      |   8.45   | [config](configs/sood/sood_fcos_2xb3-180000k_semi-0.1-dotav1.5.py) |
| ResNet50 (1024,1024,200) |   20%   |       |      55.58      |          | [config](configs/sood/sood_fcos_2xb3-180000k_semi-0.2-dotav1.5.py) |
| ResNet50 (1024,1024,200) |   30%   |       |      59.23      |          | [config](configs/sood/sood_fcos_2xb3-180000k_semi-0.3-dotav1.5.py) |

Dense Teacher
|         Backbone         | Setting | mAP50 | mAP50 in Paper | Mem (GB) |                              Config                              |
| :----------------------: | :-----: | :---: | :-------------: | :------: | :-------------------------------------------------------------: |
| ResNet50 (1024,1024,200) |   10%   | 47.10 |                 |          | [config](configs/rotated_dense_teacher/rotated-dense-teacher_2xb3-180000k_semi-0.1-dotav1.5.py) |
| ResNet50 (1024,1024,200) |   20%   |       |                 |          | [config](configs/rotated_dense_teacher/rotated-dense-teacher_2xb3-180000k_semi-0.2-dotav1.5.py) |
| ResNet50 (1024,1024,200) |   30%   |       |                 |          | [config](configs/rotated_dense_teacher/rotated-dense-teacher_2xb3-180000k_semi-0.3-dotav1.5.py) |



## Acknowledgement
This repo is built upon [mmrotate](https://github.com/open-mmlab/mmrotate).
The implementation of SOOD is based on [SOOD](https://github.com/HamPerdredes/SOOD).
The implementation of Dense Teacher is based on [Dense Teacher](https://github.com/Megvii-BaseDetection/DenseTeacher).
Thanks for their open source code.
