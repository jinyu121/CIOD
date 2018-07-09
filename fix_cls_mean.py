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
import pickle
import pprint
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from tqdm import tqdm, trange

import _init_paths
from datasets.samplers.rcnnsampler import RcnnSampler
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_fix
from model.utils.net_utils import save_checkpoint
from model.utils.net_utils import tensor_holder, ciod_old_and_new, flatten
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
    # json.dump(cfg_to_json, open(os.path.join(output_dir, 'config.json'), "w"), indent=4)

    # initialize the tensor holders here.
    im_data = tensor_holder(torch.FloatTensor(1), cfg.CUDA, True)
    im_info = tensor_holder(torch.FloatTensor(1), cfg.CUDA, True)
    num_boxes = tensor_holder(torch.LongTensor(1), cfg.CUDA, True)
    gt_boxes = tensor_holder(torch.FloatTensor(1), cfg.CUDA, True)

    # The representation classifier
    class_means = torch.zeros(2048, cfg.NUM_CLASSES + 1)
    # The iCaRL-like training procedure
    class_proto = [[] for _ in range(cfg.NUM_CLASSES + 1)]

    group_cls, group_cls_arr, group_merged_arr = ciod_old_and_new(
        cfg.NUM_CLASSES, cfg.CIOD.GROUPS, cfg.CIOD.DISTILL_GROUP)

    # Train ALL groups, or just ONE group
    start_group, end_group = (0, cfg.CIOD.GROUPS) if args.group == -1 else (args.group, args.group + 1)

    # Now we enter the group loop
    for group in trange(start_group, end_group, desc="Group", leave=True):
        # Get the net
        if args.net == 'vgg16':
            fasterRCNN = vgg16(cfg.CLASSES, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net.startswith('res'):
            fasterRCNN = resnet(cfg.CLASSES, int(args.net[3:]),
                                pretrained=True, class_agnostic=args.class_agnostic)
        else:
            raise KeyError("Unknown Network")

        fasterRCNN.create_architecture()

        load_name = os.path.join(
            output_dir,
            'faster_rcnn_{}_{}_{}_{}.pth'.format(args.session, args.net, args.dataset, group))
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])

        if cfg.CUDA:  # Send to GPU
            if cfg.MGPU:
                fasterRCNN = nn.DataParallel(fasterRCNN)
            fasterRCNN.cuda()

        tqdm.write("Resume from {}".format(load_name))

        now_cls_low, now_cls_high = group_cls[group], group_cls[group + 1]
        max_proto = max(1, cfg.CIOD.TOTAL_PROTO // (now_cls_high - 1))

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

        iters_per_epoch = train_size // cfg.TRAIN.BATCH_SIZE

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
            rois, cls_prob, bbox_pred, \
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

        for ith in trange(0 if group == 0 else now_cls_low, now_cls_high, desc="ClsMean"):
            ind_cl = np.where(labels == ith)[0]
            D = Dtot[:, ind_cl]
            # Make class mean
            tmp_mean = np.mean(D, axis=1)
            cls_mean = tmp_mean / np.linalg.norm(tmp_mean)
            class_means[:, ith] = torch.from_numpy(cls_mean)
            # Example manage
            dis = np.sum((D - np.expand_dims(cls_mean, -1)) ** 2, axis=0)

        if np.any(np.isnan(class_means)) or np.any(np.isinf(class_means)):
            save_name = os.path.join(
                output_dir,
                'faster_rcnn_{}_{}_{}_{}_FAIL.pkl'.format(args.session, args.net, args.dataset, group))
            pickle.dump(class_means, open("foo.pkl", "wb"))
            assert False, "Nan or Inf occurred! Dumped ar {} for check".format(save_name)

        # Save the model
        checkpoint['cls_means'] = class_means
        save_checkpoint(checkpoint, load_name)
        tqdm.write('Re-save model: {}'.format(load_name))
        print("{0} Group {1} Done {0}".format('=' * 10, group), end="\n" * 5)

    print("{0} All Done {0}".format('=' * 10), end="\n" * 5)
