# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import argparse
import os
import pprint
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from tqdm import tqdm, trange

from datasets.samplers.rcnnsampler import RcnnSampler
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.net_utils import change_require_gradient, heat_exp
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # Config the session ID for identify
    parser.add_argument('--s', dest='session', default=1, type=int, help='training session ID')
    # Config the session
    parser.add_argument('--dataset', dest='dataset', default='2007', type=str, help='training dataset, in VOC format')
    # Config the net
    parser.add_argument('--net', dest='net', default='res101', type=str, help='vgg16, res101')
    parser.add_argument('--ls', dest='large_scale', action='store_true', help='whether use large imag scale')
    parser.add_argument('--cag', dest='class_agnostic', action='store_true',
                        help='whether perform class_agnostic bbox regression')
    # Config optimization
    parser.add_argument('--o', dest='optimizer', default="sgd", type=str, help='training optimizer')
    # Logging, displaying and saving
    parser.add_argument('--use_tfboard', dest='use_tfboard', default=False, type=bool,
                        help='whether use tensorflow tensorboard')
    parser.add_argument('--save_dir', dest='save_dir', nargs=argparse.REMAINDER, default="results",
                        help='directory to save models')
    parser.add_argument('--nw', dest='num_workers', default=16, type=int, help='number of worker to load data')
    # Other config override
    parser.add_argument('--conf', dest='config_file', type=str, help='Other config(s) to override')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # torch.backends.cudnn.benchmark = True

    if args.use_tfboard:
        from model.utils.logger import Logger

        logger = Logger('./logs')  # Set the logger

    args.imdb_name = "voc_{}_trainval".format(args.dataset)
    args.imdbval_name = "voc_{}_test".format(args.dataset)
    cfg_from_file("cfgs/{}{}.yml".format(args.net, "_ls" if args.large_scale else ""))
    if args.config_file:
        cfg_from_file(args.config_file)

    cfg.CUDA = torch.cuda.is_available()
    cfg.MGPU = cfg.CUDA and torch.cuda.device_count() > 1

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    output_dir = os.path.join(args.save_dir, args.net, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if cfg.CUDA:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    b_fasterRCNN = None  # The backup net

    for group in trange(cfg.CIOD.GROUPS, desc="Group"):
        now_cls_low = cfg.CIOD.TOTAL_CLS * group // cfg.CIOD.GROUPS + 1
        now_cls_high = cfg.CIOD.TOTAL_CLS * (group + 1) // cfg.CIOD.GROUPS + 1

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.dataset, "trainvalStep{}".format(group))
        train_size = len(roidb)
        tqdm.write('{:d} roidb entries'.format(len(roidb)))

        lr = cfg.TRAIN.LEARNING_RATE

        sampler_batch = RcnnSampler(train_size, cfg.TRAIN.BATCH_SIZE)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, cfg.TRAIN.BATCH_SIZE, imdb.num_classes, training=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=sampler_batch, num_workers=args.num_workers)

        if 0 == group:  # Foe the first group, initialize the network here.
            if args.net == 'vgg16':
                fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
            elif args.net.startswith('res'):
                fasterRCNN = resnet(imdb.classes, int(args.net[3:]),
                                    pretrained=True, class_agnostic=args.class_agnostic)
            else:
                raise KeyError("network is not defined")

            fasterRCNN.create_architecture()

            if cfg.CUDA:
                if cfg.MGPU:
                    fasterRCNN = nn.DataParallel(fasterRCNN)
                fasterRCNN.cuda()
            b_fasterRCNN = deepcopy(fasterRCNN)  # Fake here

            params = []
            for key, value in dict(fasterRCNN.named_parameters()).items():
                if value.requires_grad:
                    if 'bias' in key:
                        params += [{'params': [value],
                                    'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
                    else:
                        params += [{'params': [value],
                                    'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            if args.optimizer == "adam":
                lr = lr * 0.1
                optimizer = torch.optim.Adam(params)
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
        else:
            # b_fasterRCNN.load_state_dict((fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict())
            b_fasterRCNN = deepcopy(fasterRCNN)
            change_require_gradient(b_fasterRCNN, False)

        iters_per_epoch = int(train_size / cfg.TRAIN.BATCH_SIZE)

        for epoch in trange(cfg.TRAIN.MAX_EPOCH, desc="Epoch"):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0
            start = time.time()

            if epoch % (cfg.TRAIN.LEARNING_RATE_DECAY_STEP + 1) == 0:
                adjust_learning_rate(optimizer, cfg.TRAIN.LEARNING_RATE_DECAY_GAMMA)
                lr *= cfg.TRAIN.LEARNING_RATE_DECAY_GAMMA

            data_iter = iter(dataloader)
            for step in trange(iters_per_epoch, desc="Iter"):
                data = next(data_iter)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, rois_label, \
                (rpn_loss_cls, rpn_loss_box, RCNN_loss_bbox, cls_score) \
                    = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                if 0 == group:
                    RCNN_loss_cls = F.cross_entropy(cls_score[:, :now_cls_high], rois_label)
                else:
                    # Backup the old net
                    b_rois, b_cls_prob, b_bbox_pred, b_rois_label, \
                    (b_rpn_loss_cls, b_rpn_loss_box, b_RCNN_loss_bbox, b_cls_score) \
                        = b_fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                    new_label_mask = rois_label >= now_cls_low
                    # label_one_hot = make_one_hot(rois_label, now_cls_high)

                    label_old = heat_exp(b_cls_score[:, :now_cls_low], cfg.CIOD.TEMPERATURE)
                    pred_old = heat_exp(cls_score[:, :now_cls_low], cfg.CIOD.TEMPERATURE)

                    loss_cls_old = F.kl_div(pred_old, label_old)

                    label_new = rois_label.index_select(0, new_label_mask.nonzero().squeeze()) - now_cls_low
                    pred_new = cls_score[:, now_cls_low:now_cls_high] \
                        .index_select(0, new_label_mask.nonzero().squeeze())

                    loss_cls_new = F.cross_entropy(pred_new, label_new)

                    RCNN_loss_cls = loss_cls_old + cfg.CIOD.NEW_CLS_LOSS_SCALE * loss_cls_new

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() + \
                       RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                loss_temp += loss.data[0]

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                if step % cfg.TRAIN.DISPLAY == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= cfg.TRAIN.DISPLAY

                    if cfg.MGPU:
                        loss_rpn_cls = rpn_loss_cls.mean().data[0]
                        loss_rpn_box = rpn_loss_box.mean().data[0]
                        loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                        loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.data[0]
                        loss_rpn_box = rpn_loss_box.data[0]
                        loss_rcnn_cls = RCNN_loss_cls.data[0]
                        loss_rcnn_box = RCNN_loss_bbox.data[0]
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    tqdm.write("[session {}] lr: {:.2}, loss: {:.4}, fg/bg=({}/{})\n"
                               "\t\t\trpn_cls: {:.4}, rpn_box: {:.4}, rcnn_cls: {:.4}, rcnn_box {:.4}".format(
                        args.session, lr, loss_temp, fg_cnt, bg_cnt,
                        loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                    if args.use_tfboard:
                        info = {
                            'loss': loss_temp,
                            'loss_rpn_cls': loss_rpn_cls,
                            'loss_rpn_box': loss_rpn_box,
                            'loss_rcnn_cls': loss_rcnn_cls,
                            'loss_rcnn_box': loss_rcnn_box
                        }
                        for tag, value in info.items():
                            logger.scalar_summary(tag, value, step)

                    loss_temp = 0
                    start = time.time()

            if (epoch + 1) % cfg.TRAIN.SAVE_INTERVAL == 0:
                save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, group, epoch + 1))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': (fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                tqdm.write('save model: {}'.format(save_name))

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}.pth'.format(args.session, group))
        save_checkpoint({
            'session': args.session,
            'epoch': cfg.TRAIN.MAX_EPOCH,
            'model': (fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        tqdm.write('save model: {}'.format(save_name))
