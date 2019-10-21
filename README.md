# CIOD: Class Incremental Object Detector

## Introduction

Based on Faster R-CNN, CIOD can add new class to a well-trained object detector, only use data of new classes.

## Preparation

First of all, clone the code
```
git clone https://github.com/jinyu121/CIOD
```

Then, create a folder:
```
cd CIOD && mkdir data
```

And setup the environment:

* Python3
* PyTorch
* CUDA 8.0 or higher


Install all the python dependencies using conda:

```
conda env create -f environment.yml --name ciod-faster
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
python3 ./trainval_net.py --session 10000
```

## Test

```
python3 ./test_net.py --session 10000 --no_repr
```

## Citation

    @InProceedings{ICME2019:FCIOD,
        author = {Yu Hao and Yanwei Fu and Yu-Gang, Jiang and Qi, Tian,
        title = {An End-to-End Architecture for Class-Incremental Object Detection With Knowledge Distillation},
        booktitle = {IEEE International Conference on Multimedia and Expo (ICME '19)},
        year = {2019}
    }

    @InProceedings{ICMR2019:TGFS, 
        author = {Yu, Hao and Yanwei, Fu and Yu-Gang, Jiang}, 
        title = {Take Goods from Shelves: A Dataset for Class-Incremental Object Detection},
        booktitle = {ACM International Conference on Multimedia Retrieval (ICMR '19)},
        year = {2019}
    }
    
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

## Note

It's toooo busy to re-write the code using [Detectron](https://github.com/facebookresearch/Detectron) or pure [TorchVision](https://github.com/pytorch/vision), which have been planned for long.

And you can also check out `d58255605e37e362ee746637fe93efb669a7e7ab`. Most of the results are produced based on this commit.
