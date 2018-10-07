# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import mxnet as mx
import numpy as np

class RCNNFGAccuracy(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNFGAccuracy, self).__init__('R-CNN FG Accuracy')

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        pred_label = mx.ndarray.argmax(pred, axis=1).asnumpy().astype('int32')
        # selection of ground truth label is different from softmax or sigmoid classifier
        label = label.asnumpy().astype('int32')
        keep_inds = np.where(label > 0)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(np.equal(pred_label.flat, label.flat))
        self.num_inst += pred_label.shape[0]

class RCNNAccMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNAccMetric, self).__init__('R-CNN Acc')

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        # pred (b, c)
        pred_label = mx.ndarray.argmax(pred, axis=1).asnumpy().astype('int32')
        # pred_label = pred_label.reshape((pred_label.shape[0], -1))
        # label (b, p)
        label = label.asnumpy().astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)
        pred_label = pred_label[keep_inds]
        label = label[keep_inds]

        self.sum_metric += np.sum(pred_label.flat == label.flat)
        self.num_inst += len(pred_label.flat)


class RCNNLogLossMetric(mx.metric.EvalMetric):
    def __init__(self):
        super(RCNNLogLossMetric, self).__init__('R-CNN LogLoss')

    def update(self, labels, preds):
        pred = preds[0]
        label = labels[0]

        last_dim = pred.shape[-1]
        pred = pred.asnumpy().reshape(-1, last_dim)
        label = label.asnumpy().reshape(-1,).astype('int32')

        # filter with keep_inds
        keep_inds = np.where(label != -1)[0]
        label = label[keep_inds]
        cls = pred[keep_inds, label]

        cls += 1e-14
        cls_loss = -1 * np.log(cls)
        cls_loss = np.sum(cls_loss)
        self.sum_metric += cls_loss
        self.num_inst += label.shape[0]
