# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import pickle
import pprint
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from tqdm import tqdm, trange

import _init_paths
from datasets.samplers.rcnnsampler import RcnnSampler
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_fix
from model.utils.net_utils import adjust_learning_rate, set_learning_rate, save_checkpoint, clip_gradient
from model.utils.net_utils import change_require_gradient, heat_exp, tensor_holder, ciod_old_and_new, flatten
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    # Config the session ID for identify
    parser.add_argument('--session', dest='session', default=1, type=int, help='Training session ID')
    parser.add_argument('--group', dest='group', type=int, default=-1, help='Train certain group, or all (-1) groups')

    # Config the session
    parser.add_argument('--dataset', dest='dataset', default='2007', type=str, help='Training dataset, in VOC format')

    # Config the net
    parser.add_argument('--net', dest='net', default='res101', type=str, help='vgg16, res101')
    parser.add_argument('--ls', dest='large_scale', action='store_true', help='Whether use large image scale')
    parser.add_argument('--cag', dest='class_agnostic', action='store_true',
                        help='Whether perform class_agnostic bbox regression')

    # Logging, displaying and saving
    parser.add_argument('--use_tfboard', dest='use_tfboard', action="store_true",
                        help='Whether use tensorflow tensorboard')
    parser.add_argument('--save_dir', dest='save_dir', nargs=argparse.REMAINDER, default="results",
                        help='Directory to save models')
    parser.add_argument('--save_without_repr', dest='save_without_repr', action="store_true",
                        help='Save the model before representation learning')
    # Other config to override
    parser.add_argument('--conf', dest='config_file', type=str, help='Other config(s) to override')

    return parser.parse_args()


if __name__ == '__main__':
    print(_init_paths.lib_path)
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter(os.path.join('logs', '{}_{}'.format(args.session, args.dataset)))

    args.imdb_name = "voc_{}_trainval".format(args.dataset)
    args.imdbval_name = "voc_{}_test".format(args.dataset)
    cfg_from_file("cfgs/{}{}.yml".format(args.net, "_ls" if args.large_scale else ""))
    if args.config_file:
        cfg_from_file(args.config_file)

    cfg_fix()

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    output_dir = os.path.join(args.save_dir, str(args.session), args.net, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    cfg_to_json = deepcopy(cfg)
    del cfg_to_json["PIXEL_MEANS"]
    json.dump(cfg_to_json, open(os.path.join(output_dir, 'config.json'), "w"), indent=4)

    # initialize the tensor holders here.
    im_data = tensor_holder(torch.FloatTensor(1), cfg.CUDA, True)
    im_info = tensor_holder(torch.FloatTensor(1), cfg.CUDA, True)
    num_boxes = tensor_holder(torch.LongTensor(1), cfg.CUDA, True)
    gt_boxes = tensor_holder(torch.FloatTensor(1), cfg.CUDA, True)

    # The representation classifier
    class_means = torch.zeros(2048, cfg.NUM_CLASSES + 1)
    # The iCaRL-like training procedure
    class_proto = [[] for _ in range(cfg.NUM_CLASSES + 1)]

    # Get the net
    if args.net == 'vgg16':
        fasterRCNN = vgg16(cfg.CLASSES, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net.startswith('res'):
        fasterRCNN = resnet(cfg.CLASSES, int(args.net[3:]),
                            pretrained=True, class_agnostic=args.class_agnostic)
    else:
        raise KeyError("Unknown Network")

    fasterRCNN.create_architecture()
    b_fasterRCNN = deepcopy(fasterRCNN)  # The backup net

    if args.group > 0:  # If not train from the first group, We should load the weights here
        load_name = os.path.join(
            output_dir,
            'faster_rcnn_{}_{}_{}_{}.pth'.format(args.session, args.net, args.dataset, args.group - 1))
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])

    if cfg.CUDA:  # Send to GPU
        if cfg.MGPU:
            fasterRCNN = nn.DataParallel(fasterRCNN)
            # b_fasterRCNN = nn.DataParallel(b_fasterRCNN)
        fasterRCNN.cuda()
        b_fasterRCNN.cuda()

    # How to optimize
    params = []
    rpn_cls_params_index = []
    base_net_params_index = []
    lr = cfg.TRAIN.LEARNING_RATE

    for key, value in dict(fasterRCNN.named_parameters()).items():  # since we froze some layers
        if value.requires_grad:
            ith = len(params)
            if 'RCNN_rpn.RPN_cls_score' in key:  # Record the parameter position of RPN_cls_score
                rpn_cls_params_index.append(ith)
            if 'RCNN_base' in key:
                base_net_params_index.append(ith)

            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if cfg.TRAIN.OPTIMIZER == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif cfg.TRAIN.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    else:
        raise KeyError("Unknown Optimizer")

    if args.group > 0:  # And load the status here
        optimizer.load_state_dict(checkpoint['optimizer'])
        cfg.POOLING_MODE = checkpoint['pooling_mode']
        class_means = checkpoint['cls_means'].float()
        class_proto = checkpoint['cls_proto']
        tqdm.write("Resume from {}".format(load_name))

    group_cls, group_cls_arr, group_merged_arr = ciod_old_and_new(
        cfg.NUM_CLASSES, cfg.CIOD.GROUPS, cfg.CIOD.DISTILL_GROUP)

    # Train ALL groups, or just ONE group
    start_group, end_group = (0, cfg.CIOD.GROUPS) if args.group == -1 else (args.group, args.group + 1)

    # Now we enter the group loop
    for group in trange(start_group, end_group, desc="Group", leave=True):
        now_cls_low, now_cls_high = group_cls[group], group_cls[group + 1]
        max_proto = max(1, cfg.CIOD.TOTAL_PROTO // (now_cls_high - int(not cfg.CIOD.REMEMBER_BG)))
        # For one class, we at least preserve 1 proto
        # And sometimes we do not want to preserve proto for background class

        lr = cfg.TRAIN.LEARNING_RATE  # Reverse the Learning Rate
        if group:
            lr *= cfg.CIOD.LEARNING_RATE_INIT_DISTILL
        if cfg.TRAIN.OPTIMIZER == 'adam':
            lr = lr * 0.1
        set_learning_rate(optimizer, lr)
        if group:
            if cfg.CIOD.SWITCH_DO_IN_RPN and cfg.CIOD.SWITCH_FREEZE_RPN_CLASSIFIER:
                set_learning_rate(optimizer, 0.0, rpn_cls_params_index)
            if cfg.CIOD.SWITCH_FREEZE_BASE_NET:
                set_learning_rate(optimizer, 1e-6, base_net_params_index)
        fasterRCNN.train()

        # Get database, and merge the class proto
        imdb, roidb, ratio_list, ratio_index = combined_roidb(
            args.dataset, "trainvalStep{}".format(group), classes=cfg.CLASSES[:now_cls_high], ext=cfg.EXT,
            data_extra=flatten(class_proto[:now_cls_low], distinct=True))

        train_size = len(roidb)
        sampler_batch = RcnnSampler(train_size, cfg.TRAIN.BATCH_SIZE)
        dataset = roibatchLoader(roidb, ratio_list, ratio_index, cfg.TRAIN.BATCH_SIZE, now_cls_high, training=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN.BATCH_SIZE, sampler=sampler_batch,
            num_workers=min(cfg.TRAIN.BATCH_SIZE * 2, os.cpu_count()))
        tqdm.write('{:d} roidb entries'.format(len(roidb)))

        # Get weights from the previous group
        b_fasterRCNN.load_state_dict((fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict())
        change_require_gradient(b_fasterRCNN, False)

        iters_per_epoch = train_size // cfg.TRAIN.BATCH_SIZE

        tot_step = 0

        # Here is the training loop
        for epoch in trange(cfg.TRAIN.MAX_EPOCH, desc="Epoch", leave=True):
            loss_temp = 0

            if epoch % cfg.TRAIN.LEARNING_RATE_DECAY_STEP == 0 and epoch > 0:
                adjust_learning_rate(optimizer, cfg.TRAIN.LEARNING_RATE_DECAY_GAMMA)
                lr *= cfg.TRAIN.LEARNING_RATE_DECAY_GAMMA

            data_iter = iter(dataloader)
            for _ in trange(iters_per_epoch, desc="Iter", leave=True):
                tot_step += 1
                data = next(data_iter)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])
                im_path = list(data[4])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, bbox_raw, \
                rpn_label, rpn_feature, rpn_cls_score, \
                rois_label, pooled_feat, cls_score, \
                rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox \
                    = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                RCNN_loss_bbox_distill = 0

                if (0 != group) and (cfg.CIOD.SWITCH_DO_IN_RPN or cfg.CIOD.SWITCH_DO_IN_FRCN):
                    # Get result from the backup net
                    b_rois, b_cls_prob, b_bbox_pred, b_bbox_raw, \
                    b_rpn_label, b_rpn_feature, b_rpn_cls_score, \
                    b_rois_label, b_pooled_feat, b_cls_score, \
                    b_rpn_loss_cls, b_rpn_loss_bbox, b_RCNN_loss_cls, b_RCNN_loss_bbox \
                        = b_fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                    if cfg.CIOD.SWITCH_DO_IN_RPN:
                        # RPN binary classification loss
                        # Less-forgetting Learning in Deep Neural Networks (Equ 1)
                        rpn_loss_cls_old = F.mse_loss(rpn_feature, b_rpn_feature)  # To make change small?
                        rpn_loss_cls_new = F.cross_entropy(rpn_cls_score, rpn_label)
                        rpn_loss_cls = cfg.CIOD.RPN_CLS_LOSS_SCALE_FEATURE * rpn_loss_cls_old + rpn_loss_cls_new

                    if cfg.CIOD.SWITCH_DO_IN_FRCN:
                        # Classification loss in Fast R-CNN
                        # For old class, use knowledge distillation with KLDivLoss
                        loss_frcn_cls_old = 0
                        for index_old in group_merged_arr[group]:
                            label_old = heat_exp(b_cls_score.index_select(1, index_old), cfg.CIOD.TEMPERATURE)
                            pred_old = heat_exp(cls_score.index_select(1, index_old), cfg.CIOD.TEMPERATURE)
                            if cfg.CIOD.DISTILL_METHOD == 'kldiv':
                                loss_frcn_cls_old += F.kl_div(torch.log(pred_old), label_old)
                            elif cfg.CIOD.DISTILL_METHOD == 'mse':
                                loss_frcn_cls_old += F.mse_loss(pred_old, label_old)
                            else:
                                raise KeyError("Unknown distill method")

                        # For new classes, use cross entropy loss
                        label_new = torch.max(torch.zeros_like(rois_label), rois_label - now_cls_low + 1)
                        pred_new = cls_score.index_select(1, group_cls_arr[group]).contiguous()
                        loss_frcn_cls_new = F.cross_entropy(pred_new, label_new)

                        # Process class 0 (__background__)
                        # If it is background class, we do not want to change it too much
                        if cfg.CIOD.DISTILL_BACKGROUND:
                            zero_label_mask = (rois_label == 0).nonzero().squeeze()
                            label_zero_f = cls_score.index_select(0, zero_label_mask)
                            pred_zero_f = cls_score.index_select(0, zero_label_mask)
                            loss_cls_zero = F.mse_loss(pred_zero_f, label_zero_f)

                        # Total classification loss
                        RCNN_loss_cls = cfg.CIOD.LOSS_SCALE_DISTILL * loss_frcn_cls_old + loss_frcn_cls_new
                        if cfg.CIOD.DISTILL_BACKGROUND:
                            RCNN_loss_cls += loss_cls_zero

                        if cfg.CIOD.DISTILL_BOUNDINGBOX and not args.class_agnostic:
                            real_shape = [cls_prob.shape[0], cls_prob.shape[1], cfg.NUM_CLASSES + 1, 4]
                            bbox_raw = bbox_raw.view(real_shape)[:, :, :now_cls_low, :]
                            b_bbox_raw = b_bbox_raw.view(real_shape)[:, :, :now_cls_low, :]
                            RCNN_loss_bbox_distill = cfg.CIOD.LOSS_SCALE_DISTILL * F.mse_loss(bbox_raw, b_bbox_raw)

                else:
                    RCNN_loss_cls = F.cross_entropy(cls_score[..., :now_cls_high], rois_label)

                loss = rpn_loss_cls.mean() + rpn_loss_bbox.mean() \
                       + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + RCNN_loss_bbox_distill

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

                    tqdm.write("[S{} G{}] lr: {:.2}, loss: {:.4}, fg/bg=({}/{})\n"
                               "\t\t\trpn_cls: {:.4}, rpn_box: {:.4}, rcnn_cls: {:.4}, rcnn_box {:.4}".format(
                        args.session, group, lr, loss_temp, fg_cnt, bg_cnt,
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
                            logger.add_scalar("Group{}/{}".format(group, tag), value, tot_step)

                    loss_temp = 0

        if cfg.CIOD.REPRESENTATION:
            if args.save_without_repr:  # We can save weights before representation learning
                save_name = os.path.join(
                    output_dir,
                    'faster_rcnn_{}_{}_{}_{}_norepr.pth'.format(args.session, args.net, args.dataset, group))
                save_checkpoint({
                    'session': args.session,
                    'epoch': cfg.TRAIN.MAX_EPOCH,
                    'model': (fasterRCNN.module if cfg.MGPU else fasterRCNN).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                    'cls_means': 0,
                    'cls_proto': class_proto
                }, save_name)
                tqdm.write('save model before representation learning: {}'.format(save_name))

            tqdm.write("===== Representation learning {} =====".format(group))
            repr_labels = []
            repr_features = []
            repr_images = []
            repr_score = []

            # Walk through all examples
            data_iter = iter(dataloader)
            # fasterRCNN.eval()
            for _ in trange(iters_per_epoch, desc="Repr", leave=True):
                data = next(data_iter)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])
                im_path = list(data[4])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, bbox_raw, \
                rpn_label, rpn_feature, rpn_cls_score, \
                rois_label, pooled_feat, cls_score, \
                rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox \
                    = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
                fasterRCNN.zero_grad()

                Dtmp = torch.t(pooled_feat)
                Dtot = Dtmp / torch.norm(Dtmp)
                repr_features.append(Dtot.data.cpu().numpy().copy())
                repr_labels.append(rois_label.data.cpu().numpy().copy())
                repr_images.extend(flatten([[x] * rois.shape[1] for x in im_path]))

            # Make representation of each class, and manage the examples
            Dtot = np.concatenate(repr_features, axis=1)
            labels = np.concatenate(repr_labels, axis=0)
            labels = labels.ravel()

            # If not use exemplar manage, we should start at now_cls_low, or it will cause error.
            re_sta = 0 if (cfg.CIOD.REMEMBER_PROTO or group == 0) else now_cls_low
            for ith in trange(re_sta, now_cls_high, desc="ClsMean"):
                ind_cl = np.where(labels == ith)[0]
                D = Dtot[:, ind_cl]
                # Make class mean
                tmp_mean = np.mean(D, axis=1)
                cls_mean = tmp_mean / np.linalg.norm(tmp_mean)
                class_means[:, ith] = torch.from_numpy(cls_mean)
                # Example manage
                dis = cdist(D.T, np.expand_dims(cls_mean, 0), 'euclidean').squeeze()
                if cfg.CIOD.REMEMBER_PROTO and (ith or cfg.CIOD.REMEMBER_BG):
                    # Sometimes we do not want to remember proto for all classes, so there is `and`
                    # Sometimes, we want to remember proto for background (0) class, so there is `or`
                    sorted_index = dis.argsort()
                    cls_set = set()
                    for idx in sorted_index:
                        cls_set.add(repr_images[ind_cl[idx]])
                        if len(cls_set) >= max_proto:
                            break
                    class_proto[ith] = list(cls_set)

            if np.any(np.isnan(class_means)) or np.any(np.isinf(class_means)):
                save_name = os.path.join(
                    output_dir,
                    'faster_rcnn_{}_{}_{}_{}_FAIL.pkl'.format(args.session, args.net, args.dataset, group))
                pickle.dump(class_means, open("foo.pkl", "wb"))
                assert False, "Nan or Inf occurred! Dumped ar {} for check".format(save_name)

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
            'cls_proto': class_proto
        }, save_name)
        tqdm.write('save model: {}'.format(save_name))
        print("{0} Group {1} Done {0}".format('=' * 10, group), end="\n" * 5)

    print("{0} All Done {0}".format('=' * 10), end="\n" * 5)
