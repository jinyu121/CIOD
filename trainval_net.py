# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from tqdm import tqdm, trange

import _init_paths
from datasets.samplers.rcnnsampler import RcnnSampler
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_from_list
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
    parser.add_argument('--bs', dest='batch_size', default=0, type=int, help='Batch size')
    # Logging, displaying and saving
    parser.add_argument('--use_tfboard', dest='use_tfboard', action="store_true",
                        help='whether use tensorflow tensorboard')
    parser.add_argument('--save_dir', dest='save_dir', nargs=argparse.REMAINDER, default="results",
                        help='directory to save models')
    parser.add_argument('--nw', dest='num_workers', default=16, type=int, help='number of worker to load data')
    # Other config override
    parser.add_argument('--conf', dest='config_file', type=str, help='Other config(s) to override')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(_init_paths.lib_path)
    args = parse_args()

    print('Called with args:')
    print(args)

    # torch.backends.cudnn.benchmark = True

    if args.use_tfboard:
        from model.utils.logger import Logger

        logger = Logger(os.path.join('logs', '{}_{}'.format(args.session, args.dataset)))  # Set the logger

    args.imdb_name = "voc_{}_trainval".format(args.dataset)
    args.imdbval_name = "voc_{}_test".format(args.dataset)
    cfg_from_file("cfgs/{}{}.yml".format(args.net, "_ls" if args.large_scale else ""))
    if args.config_file:
        cfg_from_file(args.config_file)
    if args.batch_size:
        cfg_from_list(["TRAIN.BATCH_SIZE", args.batch_size])

    cfg.CUDA = torch.cuda.is_available()
    cfg.MGPU = cfg.CUDA and torch.cuda.device_count() > 1

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    output_dir = os.path.join(args.save_dir, str(args.session), args.net, args.dataset)
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

    for group in trange(cfg.CIOD.GROUPS, desc="Group", leave=False):
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
            b_fasterRCNN = deepcopy(fasterRCNN)

            if cfg.CUDA:
                if cfg.MGPU:
                    fasterRCNN = nn.DataParallel(fasterRCNN)
                fasterRCNN.cuda()
                b_fasterRCNN.cuda()

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

        # Get the weights from the previous group
        b_fasterRCNN.load_state_dict((fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict())
        change_require_gradient(b_fasterRCNN, False)

        iters_per_epoch = int(train_size / cfg.TRAIN.BATCH_SIZE)

        tot_step = 0

        for epoch in trange(cfg.TRAIN.MAX_EPOCH, desc="Epoch", leave=False):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0

            if epoch % (cfg.TRAIN.LEARNING_RATE_DECAY_STEP) == 0 and epoch > 0:
                adjust_learning_rate(optimizer, cfg.TRAIN.LEARNING_RATE_DECAY_GAMMA)
                lr *= cfg.TRAIN.LEARNING_RATE_DECAY_GAMMA

            data_iter = iter(dataloader)
            for _ in trange(iters_per_epoch, desc="Iter", leave=False):
                tot_step += 1
                data = next(data_iter)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

                fasterRCNN.zero_grad()
                rois, cls_score, bbox_pred, pooled_feat, \
                rpn_cls_score, rpn_label, rpn_feature, \
                rpn_loss_bbox, \
                rois_label, \
                RCNN_loss_bbox = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                if 0 == group:
                    # RPN binary classification loss
                    rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)

                    # Classification loss
                    cls_prob = F.softmax(cls_score[..., :now_cls_high])
                    RCNN_loss_cls = F.cross_entropy(cls_prob, rois_label)
                else:
                    b_rois, b_cls_score, b_bbox_pred, b_pooled_feat, \
                    b_rpn_cls_score, b_rpn_label, b_rpn_feature, \
                    b_rpn_loss_bbox, \
                    b_rois_label, \
                    b_RCNN_loss_bbox = b_fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                    # RPN binary classification loss
                    rpn_loss_cls_old = F.mse_loss(rpn_cls_score, b_rpn_cls_score)  # To make change small?
                    rpn_loss_cls_new = F.cross_entropy(rpn_cls_score, rpn_label)
                    rpn_loss_cls = rpn_loss_cls_old + cfg.CIOD.NEW_CLS_LOSS_SCALE * rpn_loss_cls_new

                    # Classification loss
                    new_label_mask = (rois_label >= now_cls_low).nonzero().squeeze()
                    zero_label_mask = (rois_label == 0).nonzero().squeeze()

                    # For old class, use knowledge distillation with KLDivLoss
                    label_old = heat_exp(b_cls_score[:, :now_cls_low], cfg.CIOD.TEMPERATURE)
                    pred_old = heat_exp(cls_score[:, :now_cls_low], cfg.CIOD.TEMPERATURE)
                    loss_cls_old = F.kl_div(torch.log(pred_old), label_old)

                    # For new classes, use cross entropy loss
                    label_new = rois_label.index_select(0, new_label_mask) - now_cls_low
                    pred_new = cls_score[:, now_cls_low:now_cls_high].index_select(0, new_label_mask)
                    loss_cls_new = F.cross_entropy(pred_new, label_new)

                    # Process class 0 (__background__)
                    label_zero = rois_label.index_select(0, zero_label_mask)
                    pred_zero = cls_score[:, now_cls_low:now_cls_high].index_select(0, zero_label_mask)
                    loss_cls_zero = F.cross_entropy(pred_zero, label_zero)

                    # Total classification loss
                    RCNN_loss_cls = loss_cls_old + cfg.CIOD.NEW_CLS_LOSS_SCALE * (loss_cls_new + loss_cls_zero)

                loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() + \
                       RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

                loss_temp += loss.data[0]

                # backward
                optimizer.zero_grad()
                loss.backward()
                if args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                if tot_step % cfg.TRAIN.DISPLAY == 0:
                    if tot_step > 0:
                        loss_temp /= cfg.TRAIN.DISPLAY

                    loss_rpn_cls = rpn_loss_cls.mean().data[0]
                    loss_rpn_box = rpn_loss_bbox.mean().data[0]
                    loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                    loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
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
                            'loss_rcnn_box': loss_rcnn_box,
                            'learning_rate': lr
                        }
                        for tag, value in info.items():
                            logger.scalar_summary("Group{}/{}".format(group, tag), value, tot_step)

                    loss_temp = 0

            if (epoch + 1) % cfg.TRAIN.SAVE_INTERVAL == 0:
                save_name = os.path.join(
                    output_dir,
                    'faster_rcnn_{}_{}_{}_{}_{}.pth'.format(args.session, args.net, args.dataset, group, epoch + 1))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': (fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                    'cls_means': 0,
                }, save_name)
                tqdm.write('save model: {}'.format(save_name))

        # ===== Representation learning =====
        repr_labels = []
        repr_features = []
        class_means = np.zeros([pooled_feat.shape[-1], imdb.num_classes], dtype=np.float)
        # Make dataset (Notice the `a` here, means "[A]ll previous examples")
        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.dataset, "trainvalStep{}a".format(group))
        train_size = len(roidb)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=True, shuffle=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
        # Walk all examples
        data_iter = iter(dataloader)
        for _ in trange(iters_per_epoch, desc="Iter", leave=False):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])
            fasterRCNN.zero_grad()
            rois, cls_score, bbox_pred, pooled_feat, \
            rpn_cls_score, rpn_label, rpn_feature, \
            rpn_loss_bbox, \
            rois_label, \
            RCNN_loss_bbox = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            fasterRCNN.zero_grad()
            Dtmp = torch.t(pooled_feat)
            Dtot = Dtmp / torch.norm(Dtmp)
            repr_features.append(Dtot.data.cpu().numpy())
            repr_labels.append(rois_label.data.cpu().numpy())

        # Make representation of each class
        Dtot = np.concatenate(repr_features, axis=1)
        labels = np.concatenate(repr_labels, axis=0).ravel()

        for ith in range(now_cls_high):
            ind_cl = np.where(labels == ith)[0]
            D = Dtot[:, ind_cl]
            tmp_mean = np.mean(D, axis=1)
            class_means[:, ith] = tmp_mean / np.linalg.norm(tmp_mean)
        assert not (np.any(np.isnan(class_means)) or np.any(np.isinf(class_means))), "Nan or Inf occurred!"

        # Save the model
        save_name = os.path.join(
            output_dir,
            'faster_rcnn_{}_{}_{}_{}.pth'.format(args.session, args.net, args.dataset, group))
        save_checkpoint({
            'session': args.session,
            'epoch': cfg.TRAIN.MAX_EPOCH,
            'model': (fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
            'cls_means': class_means,
        }, save_name)
        tqdm.write('save model: {}'.format(save_name))

    print("All Done")
