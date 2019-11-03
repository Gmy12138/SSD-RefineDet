# SSD
### pytorch 1.1 and python 3.6 is supported
A PyTorch implementation of SSD, with support for training, inference and evaluation.

## Introduction
The method of SSD and RefineDet was used to perform defect detection on NEU surface defect database, and We adopted data enhancement methods such as random clipping, flipping and color enhancement. Finally achieved a satisfactory result.

## Support Architecture

    SSD: Single Shot Multibox Detector
    RefineDet: Single-Shot Refinement Neural Network for Object Detection

## Installation
##### Clone and install requirements
    $ git clone https://github.com/Gmy12138/SSD-RefineDet
    $ cd SSD-RefineDet/
    $ sudo pip install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ Download address   https://download.csdn.net/download/qq_34374211/10712378

##### Download NEU-DET dataset
    $ Download address    http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    $ cd data/
    $ Put the dataset in the data folder
    
## Test
Evaluates the model on NEU-DET test.

| Model      |Image Size|   Data Enhancement      | mAP (min. 50 IoU) |
| -------    |:--------:|:-----------------------:|:-----------------:|
| SSD        |300       |      NO                 | 37.8              |
| SSD        |300       |      YES                | 64.0              |
| RefineDet  |320       |      YES                | 68.1              |


## Inference
Uses pretrained weights to make predictions on images. The VGG16 measurement marked shows the inference time of this implementation on GPU 2080ti.

| Model      |Backbone    |  Image Size     | GPU      | FPS      |
| -----------|:----------:|:---------------:|:--------:|:--------:|
| SSD        | VGG16      |     300         | 2080ti   |   10     |
| RefineDet  | VGG16      |     320         | 2080ti   |   13     |






