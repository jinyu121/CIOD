# --------------------------------------------------------
# Tensorflow Faster R-CNN
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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm, trange

import _init_paths
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.vgg16 import vgg16
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.utils.config import cfg, cfg_from_file, get_output_dir
from model.utils.net_utils import vis_detections, cdist
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.roidb import combined_roidb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # Config the session ID for identify
    parser.add_argument('--s', dest='session', default=1, type=int, help='training session ID')
    # Config the dataset
    parser.add_argument('--dataset', dest='dataset', type=str, default='2007', help='training dataset')
    # Config the net
    parser.add_argument('--net', dest='net', default='res101', type=str, help='vgg16, res101')
    parser.add_argument('--ls', dest='large_scale', action='store_true', help='whether use large imag scale')
    parser.add_argument('--cag', dest='class_agnostic', action='store_true',
                        help='whether perform class_agnostic bbox regression')
    parser.add_argument('--no_repr', dest='no_repr', action='store_true',
                        help='Do not use representation classification')
    # Logging, displaying and saving
    parser.add_argument('--load_dir', dest='load_dir', type=str, help='directory to load models', default="results")
    parser.add_argument('--vis', dest='vis', action='store_true', help='visualization mode')
    # Other config override
    parser.add_argument('--conf', dest='config_file', type=str, help='Other config(s) to override')
    parser.add_argument('--parallel_type', dest='parallel_type', default=0, type=int,
                        help='which part of model to parallel, 0: all, 1: model before roi pooling')
    parser.add_argument('--ck', dest='self_check', action='store_true', help='Self check: test on training set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print(_init_paths.lib_path)
    args = parse_args()

    print('Called with args:')
    print(args)

    cfg.CUDA = torch.cuda.is_available()

    np.random.seed(cfg.RNG_SEED)
    args.imdb_name = "voc_{}_trainval".format(args.dataset)
    args.imdbval_name = "voc_{}_test".format(args.dataset)
    cfg_from_file("cfgs/{}{}.yml".format(args.net, "_ls" if args.large_scale else ""))
    if args.config_file:
        cfg_from_file(args.config_file)

    cfg.TRAIN.USE_FLIPPED = False

    print('Using config:')
    pprint.pprint(cfg)

    load_dir = os.path.join(args.load_dir, str(args.session), args.net, args.dataset)

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

    max_per_image = 100
    thresh = 0.05 if args.vis else 0.0
    aps = []
    for group in trange(cfg.CIOD.GROUPS, desc="Group", leave=False):
        now_cls_low = cfg.CIOD.TOTAL_CLS * group // cfg.CIOD.GROUPS + 1
        now_cls_high = cfg.CIOD.TOTAL_CLS * (group + 1) // cfg.CIOD.GROUPS + 1

        imdb, roidb, ratio_list, ratio_index = combined_roidb(
            args.dataset,
            "{}Step{}a".format("trainval" if args.self_check else "test", group),
            training=False)
        imdb.competition_mode(on=True)

        tqdm.write('{:d} roidb entries'.format(len(roidb)))

        if 0 == group:
            # initilize the network here.
            if args.net == 'vgg16':
                fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
            elif args.net.startswith('res'):
                fasterRCNN = resnet(imdb.classes, int(args.net[3:]),
                                    pretrained=True, class_agnostic=args.class_agnostic)
            else:
                raise KeyError("network is not defined")
            fasterRCNN.create_architecture()

            if cfg.CUDA:
                fasterRCNN.cuda()

        load_name = os.path.join(
            load_dir, 'faster_rcnn_{}_{}_{}_{}.pth'.format(args.session, args.net, args.dataset, group))
        tqdm.write("load checkpoint {}".format(load_name))
        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])
        class_means = torch.from_numpy(checkpoint['cls_means'][:, :now_cls_high]).float()
        cfg.POOLING_MODE = checkpoint['pooling_mode']
        tqdm.write('load model successfully!')
        if cfg.CUDA:
            class_means = class_means.cuda()

        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in range(num_images)] for _ in range(imdb.num_classes)]

        dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=False, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

        fasterRCNN.eval()
        data_iter = iter(dataloader)
        for i in trange(num_images, desc="Iter", leave=False):
            data = next(data_iter)
            im_data.data.resize_(data[0].size()).copy_(data[0])
            im_info.data.resize_(data[1].size()).copy_(data[1])
            gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            num_boxes.data.resize_(data[3].size()).copy_(data[3])

            rois, cls_prob, bbox_pred, \
            rpn_label, rpn_feature, rpn_cls_score, \
            rois_label, pooled_feat, cls_score, \
            rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox \
                = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            if args.no_repr:
                scores = F.softmax(cls_score)
                cls_prob = scores.view(im_data.size(0), rois.size(1), -1).data
            else:
                # Representation classification
                scores = torch.zeros_like(cls_score.data)
                if cfg.CUDA:
                    scores = scores.cuda()
                features = torch.t(pooled_feat)
                features = features / torch.norm(features)
                scores[:, :now_cls_high] = -torch.log(torch.t(cdist(torch.t(class_means), torch.t(features.data))))
                scores = F.softmax(Variable(scores), dim=-1).data

            if group != cfg.CIOD.GROUPS - 1:  # Tiny fix
                scores[:, now_cls_high:] = 0

            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
                    bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    if cfg.CUDA:
                        bbox_normalize_stds = bbox_normalize_stds.cuda()
                        bbox_normalize_means = bbox_normalize_means.cuda()
                    if args.class_agnostic:
                        box_deltas = box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means
                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        box_deltas = box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means
                        box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            if args.vis:
                im2show = np.copy(cv2.imread(imdb.image_path_at(i)))
            for j in range(1, imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if args.vis:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # Limit to max_per_image detections *over all classes*
            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            if args.vis:
                cv2.imwrite('result.png', im2show)
                # cv2.imshow('test', im2show)
                # cv2.waitKey(0)

        save_name = 'faster_rcnn_10'
        output_dir = get_output_dir(imdb, save_name)
        with open(os.path.join(output_dir, 'detections.pkl'), 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        tqdm.write('Evaluating detections')
        ap = imdb.evaluate_detections(all_boxes, output_dir)
        aps.append(ap)

    print("{0} RCNN {1} set Summary (mAP, AP, %) {0}".format(
        "=" * 10, "Training" if args.self_check else "Test"))
    for now_group, x in enumerate(aps):
        now_classes_low = cfg.CIOD.TOTAL_CLS * now_group // cfg.CIOD.GROUPS
        now_classes_high = cfg.CIOD.TOTAL_CLS * (now_group + 1) // cfg.CIOD.GROUPS
        print("{:.2f} :".format(np.mean(x) * 100), end="\t")
        for y in x[:now_classes_high]:
            print("{:.2f}\t".format(y * 100), end="")
        print()

    print("{0} RCNN {1} set Group Summary (mAP, AP, %) {0}".format(
        "=" * 10, "Training" if args.self_check else "Test"))
    for now_group in range(cfg.CIOD.GROUPS):
        now_classes_low = cfg.CIOD.TOTAL_CLS * now_group // cfg.CIOD.GROUPS
        now_classes_high = cfg.CIOD.TOTAL_CLS * (now_group + 1) // cfg.CIOD.GROUPS
        ans = []
        for x in range(now_group, cfg.CIOD.GROUPS):
            ans.append(np.mean(aps[x][now_classes_low:now_classes_high]) * 100.)
        print("Group {:>2} :".format(now_group), "\t->\t".join(map("{:.2f}".format, ans)))
