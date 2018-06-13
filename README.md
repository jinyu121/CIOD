# CIOD: Class Incremental Object Detector

## Introduction

Based on Faster R-CNN, CIOD can add new class to a well-trained object detector, only use data of new classes.

## Preparation

First of all, clone the code
```
git clone https://github.com/jinyu121/CIOD-FasterRCNN
```

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

And setup the environment:

* Python3
* PyTorch
* CUDA 8.0 or higher


Install all the python dependencies using pip:

```
pip3 install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib && ./make.sh
```

## Train

### Split the dataset

We split the classes of the dataset into `class group`s. For example, Pascal VOC dataset have 20 classes, and we split the classes into 4 class groups equally.

Then, we select pictures for each group in the whole dataset, forming `image group`s. Each image group will only contains pictures who only contains objects in the corresponding class group.

We can do it using `select_pic.py`.

### Train

```
python3 ./trainval_net.py --use_tfboard
```

## Test

```
python3 ./test_net.py --no_repr
```

## Authorship

+ [Yu Hao](https://haoyu.love)
+ The code is based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch)

## Citation

    @article{jinyu121-ciod-fasterrcnn,
        Author = {Yu Hao},
        Title = {CIOD: Class Incremental Object Detector},
        Journal = {https://github.com/jinyu121/CIOD-FasterRCNN},
        Year = {2018}
    } 
    
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }
