# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2018 University of Illinois
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------
"""
Fast R-CNN:
data =
    {'data': [num_images, c, h, w],
    'rois': [num_rois, 5]}
label =
    {'label': [num_rois],
    'bbox_target': [num_rois, 4 * num_classes],
    'bbox_weight': [num_rois, 4 * num_classes]}
roidb extended format [image_index]
    ['image', 'height', 'width', 'flipped',
     'boxes', 'gt_classes', 'gt_overlaps', 'max_classes', 'max_overlaps', 'bbox_targets']
"""
import cv2
import os
import numpy as np
import numpy.random as npr
from utils.image import tensor_vstack


def sample_rcnn(roi_rec, cfg, sample_per_img):

    # pred_classes = roi_rec['pred_classes']
    # pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # guard against the case when an image has fewer than fg_rois_per_image foreground RoIs
    fg_rois_per_image = int(0.25*sample_per_img)
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_indexes.size)
    # Sample foreground regions without replacement
    if len(fg_indexes) > fg_rois_per_this_image:
        fg_indexes = npr.choice(fg_indexes, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_indexes = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding against there being fewer than desired)
    bg_rois_per_this_image = sample_per_img - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_indexes.size)
    # Sample foreground regions without replacement
    if len(bg_indexes) > bg_rois_per_this_image:
        bg_indexes = npr.choice(bg_indexes, size=bg_rois_per_this_image, replace=False)

    # indexes selected
    keep_indexes = np.append(fg_indexes, bg_indexes)

    missing_sample = sample_per_img - keep_indexes.shape[0]
    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.choice(np.arange(len(max_classes)), missing_sample, replace=False)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_hard_fg(roi_rec, cfg, sample_per_img):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # gt
    gt_idx = np.where(max_overlaps == 1)[0]
    if len(gt_idx) > sample_per_img:
        gt_idx = npr.choice(gt_idx, size=sample_per_img, replace=False)
    keep_indexes = gt_idx
    # hard false positive + fg
    missing_sample = sample_per_img - len(keep_indexes)
    if missing_sample > 0:
        hard_fp_score = cfg.DCR.hard_fp_score
        hard_fg_score = cfg.DCR.hard_fg_score
        # define hard fp and fg
        hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
        hard_fp_2 = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes != max_classes)
        hard_fp_3 = (max_overlaps < cfg.train.FG_THRESH) & (pred_scores >= hard_fp_score)
        fg = (max_overlaps >= cfg.train.FG_THRESH) & (max_overlaps < 1) & (pred_classes == max_classes) & (pred_scores < hard_fg_score)
        hard_fp_idx = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3 | fg)[0]
        if len(hard_fp_idx) > missing_sample:
            hard_fp_idx = npr.choice(hard_fp_idx, size=missing_sample, replace=False)
        keep_indexes = np.append(keep_indexes, hard_fp_idx)
        missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.choice(np.arange(len(max_classes)), missing_sample, replace=False)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_random(roi_rec, cfg, sample_per_img):

    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # random sample
    keep_indexes = npr.choice(np.arange(len(max_overlaps)), size=sample_per_img, replace=False)
    pad_indexes = []
    return keep_indexes, pad_indexes


def sample_rois_fp_only(roi_rec, cfg, sample_per_img):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # hard false positive + bg
    missing_sample = sample_per_img
    hard_fp_score = cfg.DCR.hard_fp_score
    # define hard fp and bg
    hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
    hard_fp_2 = (max_overlaps >= cfg.TRAIN.FG_THRESH) & (pred_classes != max_classes)
    hard_fp_3 = (max_overlaps < cfg.TRAIN.FG_THRESH) & (pred_scores >= hard_fp_score)
    keep_indexes = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3)[0]
    if len(keep_indexes) > missing_sample:
        keep_indexes = npr.choice(keep_indexes, size=missing_sample, replace=False)
    missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.randint(len(max_overlaps), size=missing_sample)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_bg(roi_rec, cfg, sample_per_img):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # hard false positive + bg
    missing_sample = sample_per_img
    hard_fp_score = cfg.TRAIN.hard_fp_score
    # define hard fp and bg
    hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
    hard_fp_2 = (max_overlaps >= cfg.TRAIN.FG_THRESH) & (pred_classes != max_classes)
    hard_fp_3 = (max_overlaps < cfg.TRAIN.FG_THRESH) & (pred_scores >= hard_fp_score)
    bg = (max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
    keep_indexes = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3 | bg)[0]
    if len(keep_indexes) > missing_sample:
        keep_indexes = npr.choice(keep_indexes, size=missing_sample, replace=False)
    missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.randint(len(max_overlaps), size=missing_sample)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_fg(roi_rec, cfg, sample_per_img):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # gt
    gt_idx = np.where(max_overlaps == 1)[0]
    if len(gt_idx) > sample_per_img:
        gt_idx = npr.choice(gt_idx, size=sample_per_img, replace=False)
    keep_indexes = gt_idx
    # hard false positive + fg
    missing_sample = sample_per_img - len(keep_indexes)
    if missing_sample > 0:
        hard_fp_score = cfg.DCR.hard_fp_score
        # define hard fp and fg
        hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
        hard_fp_2 = (max_overlaps >= cfg.TRAIN.FG_THRESH) & (pred_classes != max_classes)
        hard_fp_3 = (max_overlaps < cfg.TRAIN.FG_THRESH) & (pred_scores >= hard_fp_score)
        fg = (max_overlaps >= cfg.TRAIN.FG_THRESH) & (max_overlaps < 1) & (pred_classes == max_classes)
        hard_fp_idx = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3 | fg)[0]
        if len(hard_fp_idx) > missing_sample:
            hard_fp_idx = npr.choice(hard_fp_idx, size=missing_sample, replace=False)
        keep_indexes = np.append(keep_indexes, hard_fp_idx)
        missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.choice(np.arange(len(max_classes)), missing_sample, replace=False)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_fg_bg(roi_rec, cfg, sample_per_img):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    # sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # gt
    gt_idx = np.where(max_overlaps == 1)[0]
    if len(gt_idx) > sample_per_img:
        gt_idx = npr.choice(gt_idx, size=sample_per_img, replace=False)
    keep_indexes = gt_idx
    # hard false positive + fg
    missing_sample = sample_per_img - len(keep_indexes)
    if missing_sample > 0:
        hard_fp_score = cfg.DCR.hard_fp_score
        # define hard fp and fg
        hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
        hard_fp_2 = (max_overlaps >= cfg.TRAIN.FG_THRESH) & (pred_classes != max_classes)
        hard_fp_3 = (max_overlaps < cfg.TRAIN.FG_THRESH) & (pred_scores >= hard_fp_score)
        fg = (max_overlaps >= cfg.TRAIN.FG_THRESH) & (max_overlaps < 1) & (pred_classes == max_classes)
        bg = (max_overlaps < cfg.TRAIN.BG_THRESH_HI) & (max_overlaps >= cfg.TRAIN.BG_THRESH_LO)
        hard_fp_idx = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3 | fg | bg)[0]
        if len(hard_fp_idx) > missing_sample:
            hard_fp_idx = npr.choice(hard_fp_idx, size=missing_sample, replace=False)
        keep_indexes = np.append(keep_indexes, hard_fp_idx)
        missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.choice(np.arange(len(max_classes)), missing_sample, replace=False)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes
