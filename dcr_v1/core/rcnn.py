# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
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

def get_rcnn_test_batch(roidb, boxes_all, cfg, pad):
    num_images = len(roidb)
    imgs_array = list()
    for i in range(num_images):
        roi_rec = roidb[i]
        boxes = boxes_all[i][:, 4:8]
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        height, width = im.shape[0], im.shape[1]
        scale = cfg.train.roi_enlarge_scale
        if scale > 1.0:
            boxes = resize_box(boxes, scale)
        boxes = clip_boxes(boxes, (height, width))
        for n in range(boxes.shape[0]):
            imgs_array.append(crop_transform(im, boxes[n], cfg))
    if pad > 0:
        batch_size_per_gpu = boxes_all[0][:, 4:8].shape[0]
        image_shape = [int(l) for l in cfg.dataset.image_shape.split(',')]
        (nchannel, height, width) = image_shape
        for _ in range(pad * batch_size_per_gpu):
            imgs_array.append(np.zeros((1, 3, height, width)))

    imgs_array = tensor_vstack(imgs_array)
    data = {'data': imgs_array}
    return data


def get_rcnn_batch(roidb, cfg):
    """
    return a dict of multiple images
    :param roidb: a list of dict, whose length controls batch size
    ['images', 'flipped'] + ['gt_boxes', 'boxes', 'gt_overlap'] => ['bbox_targets']
    :return: data, label
    """

    # load image
    num_images = len(roidb)
    imgs_array = list()
    labels_array = list()
    for i in range(num_images):
        roi_rec = roidb[i]
        assert os.path.exists(roi_rec['image']), '%s does not exist'.format(roi_rec['image'])
        im = cv2.imread(roi_rec['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if roi_rec['flipped']:
            im = im[:, ::-1, :]
        height, width = im.shape[0], im.shape[1]
        max_classes = generate_labels(roi_rec, cfg)
        # use default sampling method
        if cfg.train.SAMPLE_RCNN:
            rois_idx, pad_idx = sample_rois_hard_fg(roi_rec, cfg)
        elif cfg.train.SAMPLE_DEFAULT:
            rois_idx, pad_idx = sample_rois(roi_rec, cfg)
        # random sample
        elif cfg.train.SAMPLE_RANDOM:
            rois_idx, pad_idx = sample_rois_random(roi_rec, cfg)
        # sample fp only
        elif not cfg.train.SAMPLE_BG and not cfg.train.SAMPLE_FG:
            rois_idx, pad_idx = sample_rois_fp_only(roi_rec, cfg)
        # sample fp + bg
        elif cfg.train.SAMPLE_BG and not cfg.train.SAMPLE_FG:
            rois_idx, pad_idx = sample_rois_bg(roi_rec, cfg)
        # sample fp + fg
        elif not cfg.train.SAMPLE_BG and cfg.train.SAMPLE_FG:
            rois_idx, pad_idx = sample_rois_fg(roi_rec, cfg)
        # sample fp + bg + fg
        elif cfg.train.SAMPLE_BG and cfg.train.SAMPLE_FG:
            rois_idx, pad_idx = sample_rois_fg_bg(roi_rec, cfg)
        else:
            raise NotImplemented
            # rois_idx, pad_idx = sample_rois(roi_rec, cfg)
        boxes = roi_rec['boxes']
        scale = cfg.train.roi_enlarge_scale
        if scale > 1.0:
            boxes = resize_box(boxes, scale)
        boxes = clip_boxes(boxes, (height, width))
        for idx in rois_idx:
            imgs_array.append(crop_transform(im, boxes[idx], cfg))
            labels_array.append(max_classes[idx])
        if len(pad_idx) > 0:
            for idx in pad_idx:
                imgs_array.append(crop_transform(im, boxes[idx], cfg))
                labels_array.append(-1)

    imgs_array = tensor_vstack(imgs_array)
    labels_array = np.array(labels_array)
    data = {'data': imgs_array}
    label = {'softmax_label': labels_array}

    return data, label


def generate_labels(roi_rec, cfg):
    gt_classes = roi_rec['gt_classes']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']
    max_classes_new = max_classes.copy()

    for idx in range(len(gt_classes)):
        max_overlap = max_overlaps[idx]
        if max_overlap < cfg.train.FG_THRESH:
            max_classes_new[idx] = 0

    return max_classes_new

def sample_rcnn(roi_rec, cfg):

    # pred_classes = roi_rec['pred_classes']
    # pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # foreground RoI with FG_THRESH overlap
    fg_indexes = np.where(max_overlaps >= cfg.train.FG_THRESH)[0]
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


def sample_rois_hard_fg(roi_rec, cfg):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
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
        hard_fp_score = cfg.train.hard_fp_score
        hard_fg_score = cfg.train.hard_fg_score
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


def sample_rois_random(roi_rec, cfg):

    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # random sample
    keep_indexes = npr.choice(np.arange(len(max_overlaps)), size=sample_per_img, replace=False)
    pad_indexes = []
    return keep_indexes, pad_indexes


def sample_rois_fp_only(roi_rec, cfg):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # hard false positive + bg
    missing_sample = sample_per_img
    hard_fp_score = cfg.train.hard_fp_score
    # define hard fp and bg
    hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
    hard_fp_2 = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes != max_classes)
    hard_fp_3 = (max_overlaps < cfg.train.FG_THRESH) & (pred_scores >= hard_fp_score)
    keep_indexes = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3)[0]
    if len(keep_indexes) > missing_sample:
        keep_indexes = npr.choice(keep_indexes, size=missing_sample, replace=False)
    missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.randint(len(max_overlaps), size=missing_sample)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_bg(roi_rec, cfg):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # hard false positive + bg
    missing_sample = sample_per_img
    hard_fp_score = cfg.train.hard_fp_score
    # define hard fp and bg
    hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
    hard_fp_2 = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes != max_classes)
    hard_fp_3 = (max_overlaps < cfg.train.FG_THRESH) & (pred_scores >= hard_fp_score)
    bg = (max_overlaps < cfg.train.BG_THRESH_HI) & (max_overlaps >= cfg.train.BG_THRESH_LO)
    keep_indexes = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3 | bg)[0]
    if len(keep_indexes) > missing_sample:
        keep_indexes = npr.choice(keep_indexes, size=missing_sample, replace=False)
    missing_sample = sample_per_img - len(keep_indexes)

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.randint(len(max_overlaps), size=missing_sample)

    assert len(keep_indexes) + len(pad_indexes) == sample_per_img
    return keep_indexes, pad_indexes


def sample_rois_fg(roi_rec, cfg):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
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
        hard_fp_score = cfg.train.hard_fp_score
        # define hard fp and fg
        hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
        hard_fp_2 = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes != max_classes)
        hard_fp_3 = (max_overlaps < cfg.train.FG_THRESH) & (pred_scores >= hard_fp_score)
        fg = (max_overlaps >= cfg.train.FG_THRESH) & (max_overlaps < 1) & (pred_classes == max_classes)
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


def sample_rois_fg_bg(roi_rec, cfg):

    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
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
        hard_fp_score = cfg.train.hard_fp_score
        # define hard fp and fg
        hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
        hard_fp_2 = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes != max_classes)
        hard_fp_3 = (max_overlaps < cfg.train.FG_THRESH) & (pred_scores >= hard_fp_score)
        fg = (max_overlaps >= cfg.train.FG_THRESH) & (max_overlaps < 1) & (pred_classes == max_classes)
        bg = (max_overlaps < cfg.train.BG_THRESH_HI) & (max_overlaps >= cfg.train.BG_THRESH_LO)
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


def sample_rois(roi_rec, cfg):

    # gt_classes = roi_rec['gt_classes']
    pred_classes = roi_rec['pred_classes']
    pred_scores = roi_rec['scores']
    max_classes = roi_rec['max_classes']
    # overlaps = roi_rec['gt_overlaps']
    max_overlaps = roi_rec['max_overlaps']

    sample_per_img = cfg.train.sample_per_image
    # assert len(max_classes) >= sample_per_img
    if sample_per_img > len(max_classes):
        keep_indexes = np.arange(len(max_overlaps))
        # pad_indexes = npr.choice(np.arange(len(max_overlaps)), size=sample_per_img - len(max_overlaps), replace=False)
        pad_indexes = npr.randint(len(max_overlaps), size=sample_per_img - len(max_overlaps))
        return keep_indexes, pad_indexes

    # debug purpose
    add_gt = False
    add_fp_bg = False
    add_fg = False
    add_pad = False
    # gt
    gt_idx = np.where(max_overlaps == 1)[0]
    if len(gt_idx) > sample_per_img:
        gt_idx = npr.choice(gt_idx, size=sample_per_img, replace=False)
    keep_indexes = gt_idx
    # hard false positive
    missing_sample = sample_per_img - len(keep_indexes)
    add_gt = True
    hard_fp_idx = []
    if missing_sample > 0:
        hard_fp_score = cfg.train.hard_fp_score
        # define hard fp
        hard_fp_1 = (pred_scores >= hard_fp_score) & (pred_classes != max_classes)
        hard_fp_2 = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes != max_classes)
        hard_fp_3 = (max_overlaps < cfg.train.FG_THRESH) & (pred_scores >= hard_fp_score)
        bg = (max_overlaps < cfg.train.BG_THRESH_HI) & (max_overlaps >= cfg.train.BG_THRESH_LO)
        hard_fp_idx = np.where(hard_fp_1 | hard_fp_2 | hard_fp_3 | bg)[0]
        if len(hard_fp_idx) > missing_sample:
            hard_fp_idx = npr.choice(hard_fp_idx, size=missing_sample, replace=False)
        keep_indexes = np.append(keep_indexes, hard_fp_idx)
        missing_sample = sample_per_img - len(keep_indexes)
        add_fp_bg = True

    fg_idx = []
    if missing_sample > 0:
        fg = (max_overlaps >= cfg.train.FG_THRESH) & (pred_classes == max_classes)
        fg_idx = np.where(fg)[0]
        if len(fg_idx) > missing_sample:
            fg_idx = npr.choice(fg_idx, size=missing_sample, replace=False)
        keep_indexes = np.append(keep_indexes, fg_idx)
        missing_sample = sample_per_img - len(keep_indexes)
        add_fg = True

    pad_indexes = []
    if missing_sample > 0:
        pad_indexes = npr.choice(np.arange(len(max_classes)), missing_sample, replace=False)
        add_pad = True

    # assert len(keep_indexes) + len(pad_indexes) == sample_per_img,\
    #     'keep: {} (gt: {} fp_bg: {} fg: {}) pad: {} gt: {} fp_bg: {} fg: {} pad: {}'.format(len(keep_indexes), len(pad_indexes),
    #                                                                                  len(gt_idx), len(hard_fp_idx), len(fg_idx),
    #                                                                                  add_gt, add_fp_bg, add_fg, add_pad)
    assert len(keep_indexes) + len(pad_indexes) == sample_per_img

    return keep_indexes, pad_indexes


def crop_transform(img, box, cfg):
    box = box.astype(np.int32)
    # crop
    cropped_img = img[box[1]:box[3]+1, box[0]:box[2]+1, :]
    # resize
    image_shape = [int(l) for l in cfg.dataset.image_shape.split(',')]
    (nchannel, height, width) = image_shape
    cropped_img = cv2.resize(cropped_img, (height, width), interpolation=cv2.INTER_LINEAR)
    # transform
    cropped_img_tensor = transform(cropped_img, cfg.network.PIXEL_MEANS)
    return cropped_img_tensor


def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [B, G, R pixel means]
    :return: [batch, channel, height, width]
    """
    im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]))
    for i in range(3):
        im_tensor[0, i, :, :] = im[:, :, 2 - i] - pixel_means[2 - i]
    return im_tensor

def resize_box(boxes, scale):

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

    new_widths = scale * widths
    new_heights = scale * heights

    new_boxes = np.zeros(boxes.shape)
    # x1
    new_boxes[:, 0] = ctr_x - 0.5 * (new_widths - 1.0)
    # y1
    new_boxes[:, 1] = ctr_y - 0.5 * (new_heights - 1.0)
    # x2
    new_boxes[:, 2] = ctr_x + 0.5 * (new_widths - 1.0)
    # y2
    new_boxes[:, 3] = ctr_y + 0.5 * (new_heights - 1.0)

    return new_boxes.astype(np.int32)

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    :param boxes: [N, 4* num_classes]
    :param im_shape: tuple of 2
    :return: [N, 4* num_classes]
    """
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


