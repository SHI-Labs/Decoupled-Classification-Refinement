# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Written by Bowen Cheng
# --------------------------------------------------------

"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import mxnet as mx
import numpy as np

from bbox.bbox_transform import bbox_pred, clip_boxes
import cPickle

DEBUG = False


class DCRProposalOperator(mx.operator.CustomOp):
    def __init__(self, cfg):
        super(DCRProposalOperator, self).__init__()
        self._cfg = cfg

    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()[:, 1:]
        cls_prob = in_data[1].asnumpy()
        assert self._cfg.CLASS_AGNOSTIC, 'Currently only support class agnostic'
        if self._cfg.CLASS_AGNOSTIC:
            bbox_deltas = in_data[2].asnumpy()[:, 4:8]
        else:
            fg_cls_prob = cls_prob[:, 1:]
            fg_cls_idx = np.argmax(fg_cls_prob, axis=1).astype(np.int)
            batch_idx_array = np.arange(fg_cls_idx.shape[0], dtype=np.int)
            # bbox_deltas = in_data[2].asnumpy()[batch_idx_array, fg_cls_idx * 4: (fg_cls_idx + 1) * 4]
            in_data2 = in_data[2].asnumpy()
            bbox_deltas = np.hstack((in_data2[batch_idx_array, fg_cls_idx * 4].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4 + 1].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4 + 2].reshape(-1, 1),
                                     in_data2[batch_idx_array, fg_cls_idx * 4 + 3].reshape(-1, 1)))
        im_info = in_data[3].asnumpy()[0, :]

        # post processing
        # if self._is_train:
        #     if self._cfg.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED:
        #         bbox_deltas = bbox_deltas * np.array(self._cfg.TRAIN.BBOX_STDS) + np.array(self._cfg.TRAIN.BBOX_MEANS)

        proposals = bbox_pred(rois, bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        self.assign(out_data[0], req[0], blob)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)


@mx.operator.register("dcr_proposal")
class DCRProposalProp(mx.operator.CustomOpProp):
    def __init__(self, cfg):
        super(DCRProposalProp, self).__init__(need_top_grad=False)
        self._cfg = cPickle.loads(cfg)

    def list_arguments(self):
        return ['rois', 'cls_prob', 'bbox_pred', 'im_info']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        cls_prob_shape = in_shape[1]
        bbox_pred_shape = in_shape[2]
        assert rois_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in reg'

        batch_size = 1
        im_info_shape = (batch_size, 3)
        output_shape = rois_shape

        return [rois_shape, cls_prob_shape, bbox_pred_shape, im_info_shape], [output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DCRProposalOperator(self._cfg)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
