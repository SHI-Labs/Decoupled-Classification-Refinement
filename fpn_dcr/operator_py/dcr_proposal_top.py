# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Written by Bowen Cheng
# --------------------------------------------------------

"""
Proposal Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
With our proposed "top-sampling" strategy
"""

import mxnet as mx
import numpy as np

from bbox.bbox_transform import bbox_pred, clip_boxes
import cPickle

DEBUG = False


class DCRProposalTopOperator(mx.operator.CustomOp):
    def __init__(self, top):
        super(DCRProposalTopOperator, self).__init__()
        self._top = top

    def forward(self, is_train, req, in_data, out_data, aux):

        rois = in_data[0].asnumpy()[:, 1:]
        bbox_deltas = in_data[1].asnumpy()[:, 4:8]
        im_info = in_data[2].asnumpy()[0, :]
        cls_prob = in_data[3].asnumpy()[:, 1:]  # ignore bg

        num_keep_index = int(rois.shape[0] * self._top)
        # sort scores
        max_scores = np.amax(cls_prob, axis=1)
        # keep top scores
        keep_index = np.argsort(-max_scores)[:num_keep_index]

        proposals = bbox_pred(rois, bbox_deltas)
        proposals = clip_boxes(proposals, im_info[:2])

        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

        self.assign(out_data[0], req[0], blob[keep_index, :])
        self.assign(out_data[1], req[1], keep_index)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)
        self.assign(in_grad[3], req[3], 0)


@mx.operator.register("dcr_proposal_top")
class DCRProposalTopProp(mx.operator.CustomOpProp):
    def __init__(self, top):
        super(DCRProposalTopProp, self).__init__(need_top_grad=False)
        self._top = float(top)

    def list_arguments(self):
        return ['rois', 'bbox_pred', 'im_info', 'cls_prob']

    def list_outputs(self):
        return ['output', 'keep_index']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]
        bbox_pred_shape = in_shape[1]
        cls_prob_shape = in_shape[3]
        assert rois_shape[0] == bbox_pred_shape[0], 'ROI number does not equal in reg'

        roi_batch_size = int(rois_shape[0] * self._top)
        batch_size = 1
        im_info_shape = (batch_size, 3)

        output_shape = (roi_batch_size, rois_shape[1])
        keep_index_shape = (roi_batch_size, )

        return [rois_shape, bbox_pred_shape, im_info_shape, cls_prob_shape],\
               [output_shape, keep_index_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return DCRProposalTopOperator(self._top)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
