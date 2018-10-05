# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Written by Bowen Cheng
# --------------------------------------------------------

"""
Proposal Target Operator for DCR selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np

from bbox.bbox_transform import bbox_pred, clip_boxes, bbox_overlaps
from core.dcr import sample_rois_fg_bg, sample_rcnn, sample_rois_random
import cPickle

DEBUG = False


class DCRTargetOperator(mx.operator.CustomOp):
    def __init__(self, cfg, resample):
        super(DCRTargetOperator, self).__init__()
        self._cfg = cfg
        self._resample = resample

    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()
        cls_prob = in_data[1].asnumpy()
        assert self._cfg.CLASS_AGNOSTIC, 'Currently only support class agnostic'
        if self._cfg.CLASS_AGNOSTIC:
            bbox_deltas = in_data[2].asnumpy()[:, 4:8]
        else:
            fg_cls_prob = cls_prob[:, 1:]
            fg_cls_idx = np.argmax(fg_cls_prob, axis=1).astype(np.int)
            batch_idx_array = np.arange(fg_cls_idx.shape[0], dtype=np.int)
            # bbox_deltas = in_data[2].asnumpy()[batch_idx_array, fg_cls_idx * 4 : (fg_cls_idx+1) * 4]
            in_data2 = in_data[2].asnumpy()
            bbox_deltas = np.hstack((in_data2[batch_idx_array, fg_cls_idx * 4].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4+1].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4+2].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4+3].reshape(-1, 1)))
        im_info = in_data[3].asnumpy()[0, :]
        gt_boxes = in_data[4].asnumpy()

        # post processing
        if self._cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
            bbox_deltas = bbox_deltas * np.array(self._cfg.TRAIN.BBOX_STDS) + np.array(self._cfg.TRAIN.BBOX_MEANS)

        proposals = bbox_pred(rois[:, 1:], bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])

        # only support single batch
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        # reassign label
        gt_classes = gt_boxes[:, -1].astype(np.int)
        overlaps = np.zeros((blob.shape[0], self._cfg.dataset.NUM_CLASSES), dtype=np.float32)
        # n boxes and k gt_boxes => n * k overlap
        gt_overlaps = bbox_overlaps(blob[:, 1:].astype(np.float), gt_boxes[:, :-1].astype(np.float))
        # for each box in n boxes, select only maximum overlap (must be greater than zero)
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

        roi_max_classes = overlaps.argmax(axis=1)
        roi_max_overlaps = overlaps.max(axis=1)
        # assign bg labels
        roi_max_classes[np.where(roi_max_overlaps < self._cfg.TRAIN.FG_THRESH)] = 0
        assert (roi_max_classes[np.where(roi_max_overlaps < self._cfg.TRAIN.FG_THRESH)] == 0).all()

        if self._resample == -1:
            self.assign(out_data[0], req[0], blob)
            self.assign(out_data[1], req[1], roi_max_classes)
        else:
            # Include ground-truth boxes in the set of candidate rois
            batch_inds = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
            all_rois = np.vstack((np.hstack((batch_inds, gt_boxes[:, :-1])), blob))

            # gt boxes
            pred_classes = gt_boxes[:, -1]
            pred_scores = np.ones_like(pred_classes)
            max_classes = pred_classes.copy()
            max_overlaps = np.ones_like(max_classes)
            # predicted boxes
            roi_pred_classes = cls_prob.argmax(axis=1)
            roi_pred_scores = cls_prob.max(axis=1)

            roi_rec = {}
            # roi_rec['pred_classes'] = np.vstack((pred_classes, roi_pred_classes))
            # roi_rec['scores'] = np.vstack((pred_scores, roi_pred_scores))
            # roi_rec['max_classes'] = np.vstack((max_classes, roi_max_classes))
            # roi_rec['max_overlaps'] = np.vstack((max_overlaps, roi_max_overlaps))
            roi_rec['pred_classes'] = np.append(pred_classes, roi_pred_classes)
            roi_rec['scores'] = np.append(pred_scores, roi_pred_scores)
            roi_rec['max_classes'] = np.append(max_classes, roi_max_classes)
            roi_rec['max_overlaps'] = np.append(max_overlaps, roi_max_overlaps)

            if self._cfg.DCR.sample == 'DCRV1':
                keep_indexes, pad_indexes = sample_rois_fg_bg(roi_rec, self._cfg, self._resample)
            elif self._cfg.DCR.sample == 'RANDOM':
                keep_indexes, pad_indexes = sample_rois_random(roi_rec, self._cfg, self._resample)
            else:
                raise ValueError('Undefined sampling method: %s' % self._cfg.DCR.sample)

            resampled_blob = np.vstack((all_rois[keep_indexes, :], all_rois[pad_indexes, :]))
            # assign bg classes
            assert (roi_rec['max_classes'][np.where(roi_rec['max_overlaps'] < self._cfg.TRAIN.FG_THRESH)] == 0).all()
            resampled_label = np.append(roi_rec['max_classes'][keep_indexes], -1*np.ones(len(pad_indexes)))

            self.assign(out_data[0], req[0], resampled_blob)
            self.assign(out_data[1], req[1], resampled_label)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)
        self.assign(in_grad[4], req[4], 0)


@mx.operator.register("dcr_target")
class DCRTargetProp(mx.operator.CustomOpProp):
    def __init__(self, cfg):
        super(DCRTargetProp, self).__init__(need_top_grad=False)
        self._cfg = cPickle.loads(cfg)
        self._resample = self._cfg.DCR.sample_per_img

    def list_arguments(self):
        return ['rois', 'cls_prob', 'bbox_pred', 'im_info', 'gt_boxes']

    def list_outputs(self):
        return ['output', 'label']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        cls_prob_shape = in_shape[1]
        bbox_pred_shape = in_shape[2]
        gt_boxes_shape = in_shape[4]
        assert rois_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in reg'

        rois = rois_shape[0] if self._resample == -1 else self._resample
        batch_size = 1
        im_info_shape = (batch_size, 3)
        output_shape = (rois, 5)
        label_shape = (rois, )

        return [rois_shape, cls_prob_shape, bbox_pred_shape, im_info_shape, gt_boxes_shape], \
               [output_shape, label_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DCRTargetOperator(self._cfg, self._resample)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
