# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import os
import sys
import os.path as osp
import argparse
import cPickle
import numpy as np

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# add lib
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, '..', 'lib')
add_path(lib_path)
path_to_faster_rcnn = osp.join(this_dir, '..', 'fpn') # change to faster_rcnn
cfg_path = osp.join(path_to_faster_rcnn, 'config')
add_path(cfg_path)

from config import config, update_config
from dataset import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Label each detections with corresponding class')
    # parser.add_argument('--thresh', dest='thresh',
    #                     help='IOU threshold for gt',
    #                     default=0.5, type=float)
    parser.add_argument('--score_thresh', dest='score_thresh',
                        help='score threshold for interests',
                        default=0.0, type=float)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset with gt label',
                        default='2007_test', type=str)
    parser.add_argument('--cfg', dest='cfg',
                        help='path to config file',
                        required=True, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def iou(bbgt, bb):
    # intersection
    iymin = np.maximum(bbgt[:, 0], bb[0])
    ixmin = np.maximum(bbgt[:, 1], bb[1])
    iymax = np.minimum(bbgt[:, 2], bb[2])
    ixmax = np.minimum(bbgt[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (bbgt[:, 2] - bbgt[:, 0] + 1.) *
           (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

DEBUG = False

def label_dets(args, cfg):
    root_output_path = cfg.output_path
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
    final_output_path = os.path.join(config_output_path, '{}'.format(args.imdb_name))
    dataset = cfg.dataset.dataset
    image_set = args.imdb_name
    root_path = cfg.dataset.root_path
    dataset_path = cfg.dataset.dataset_path
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=final_output_path)
    roidb = imdb.gt_roidb()

    det_file = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    assert os.path.exists(det_file), 'Please generate detections first'
    with open(det_file, 'rb') as fid:
        all_boxes = cPickle.load(fid)
    if DEBUG:
        for cls_ind, cls in enumerate(imdb.classes):
            all_boxes[cls_ind] = all_boxes[cls_ind][:2]

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    # target output label
    #    labels[image] = N x 8 array of detections in
    #    (x1, y1, x2, y2, score, pred_label, gt_label, gt_iou)

    num_images = imdb.num_images
    if DEBUG:
        num_images = 2
    labels = [np.zeros((0, 8)) for _ in range(num_images)]

    for n in range(num_images):
        gt_boxes = roidb[n]['boxes']
        if gt_boxes.shape[0] ==  0:
            labels[n] = np.zeros((0, 8))
            continue
        for cls_ind, cls in enumerate(imdb.classes):
            if cls == '__background__':
                continue
            dets = all_boxes[cls_ind][n]
            if len(dets) == 0:
                continue
            keep = np.where(dets[:, 4] > args.score_thresh)[0]
            dets = dets[keep]
            # keep = np.where((dets[:, 0] >= 0) & (dets[:, 1] >= 0) & (dets[:, 2] >= 0) & (dets[:, 3] >= 0))[0]
            # dets = dets[keep]
            keep = np.where((dets[:, 2] > dets[:, 0]) & (dets[:, 3] > dets[:, 1]))[0]
            dets = dets[keep]
            assert (dets[:, 2] >= dets[:, 0]).all()
            assert (dets[:, 3] >= dets[:, 1]).all()
            temp_labels = np.hstack((dets, cls_ind*np.ones((dets.shape[0], 1)), np.zeros((dets.shape[0], 1)), np.zeros((dets.shape[0], 1))))
            labels[n] = np.vstack((labels[n], temp_labels))
        for nbox in range(labels[n].shape[0]):
            box = labels[n][nbox, :4]
            overlaps = iou(gt_boxes, box)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            labels[n][nbox, -2] = roidb[n]['gt_classes'][jmax]
            labels[n][nbox, -1] = ovmax

        print 'Processing {}/{} total box {} gt box {}'.format(n+1, num_images, labels[n].shape[0], gt_boxes.shape[0])

    label_file = os.path.join(imdb.result_path, imdb.name + '_labels.pkl')
    with open(label_file, 'wb') as f:
        cPickle.dump(labels, f, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    args = parse_args()
    update_config(args.cfg)
    label_dets(args, config)