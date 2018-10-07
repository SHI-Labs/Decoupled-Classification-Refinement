# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import _init_paths

import argparse
import os
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
import numpy as np
import cPickle
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='Test a RCNN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--thresh', help='filter classification score', default=0.0, type=float)
    parser.add_argument('--gpu', help='suppress gpu setting in config files',
                        default=None, type=str)
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')

    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from dataset import *
from utils.create_logger import create_logger
from core.loader import TestLoader, PrefetchingIter
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
# from utils.PrefetchingIter import PrefetchingIter
# from mxnet.io import PrefetchingIter
import time


def main():
    if args.gpu is not None:
        config.gpus = args.gpu
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    print args

    if args.thresh > 0.0:
        config.test.thresh = args.thresh

    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set + '_combined')

    pprint.pprint(config)
    logger.info('testing cfg:{}\n'.format(pprint.pformat(config)))

    # get imdb
    dataset = config.dataset.dataset
    image_set = config.dataset.test_image_set
    root_path = config.dataset.root_path
    dataset_path = config.dataset.dataset_path
    imdb = eval(dataset)(image_set, root_path, dataset_path, result_path=final_output_path)
    roidb = imdb.gt_roidb()

    det_file_new = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    if os.path.exists(det_file_new) and not args.ignore_cache:
        with open(det_file_new, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        info_str = imdb.evaluate_detections(all_boxes)
        if logger:
            logger.info('evaluate detections: \n{}'.format(info_str))
        return

    # load symbol and params
    prefix = os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]),
                          config.TRAIN.model_prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, config.test.test_epoch)

    # get detections
    det_file_name = config.train.base_detector
    det_file = os.path.join(imdb.root_path, 'rcnn_detection_data', det_file_name, imdb.name + '_detections_all.pkl')
    assert os.path.exists(det_file), 'Please generate detections first'
    with open(det_file, 'rb') as fid:
        all_boxes = cPickle.load(fid)
    scores_all = all_boxes['scores_all']
    boxes_all = all_boxes['boxes_all']
    scores_all_new = list()

    # get test data iter
    batch_size = len(ctx)
    input_batch_size = int(config.test.BATCH_IMAGES * batch_size)
    test_data = TestLoader(roidb, boxes_all, config, batch_size=input_batch_size, shuffle=False)

    # get module
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=['softmax_label', ])
    mod.bind(for_training=False,
             data_shapes=test_data.provide_data,)
    mod.set_params(arg_params, aux_params)

    nms = py_nms_wrapper(config.test.NMS)
    max_per_image = config.test.max_per_image
    num_images = imdb.num_images
    thresh = config.test.thresh
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # if not isinstance(test_data, PrefetchingIter):
    #     test_data = PrefetchingIter(test_data)
    # else:
    #     prefetch_test_data = test_data

    idx = 0
    data_time, net_time, post_time = 0.0, 0.0, 0.0
    t = time.time()

    print (test_data.batch_size, test_data.provide_data)

    for data_batch in test_data:
        t1 = time.time() - t
        t = time.time()

        mod.forward(data_batch, is_train=False)
        # outputs = mod.get_outputs(merge_multi_context=False)[0]
        outputs = mod.get_outputs(merge_multi_context=True)[0].asnumpy()
        np_outputs = [outputs[i*config.test.NUM_PROPOSALS : (i+1)*config.test.NUM_PROPOSALS, :] for i in range(input_batch_size)]

        t2 = time.time() - t
        t = time.time()

        for delta, output in enumerate(np_outputs):
            if data_batch.pad > 0:
                if delta >= len(ctx) - data_batch.pad:
                    continue

            scores = output
            scores_new = scores * scores_all[idx+delta]
            scores_all_new.append(scores_new)
            boxes_new = boxes_all[idx+delta]

            for j in range(1, imdb.num_classes):
                indexes = np.where(scores_new[:, j] > thresh)[0]
                cls_scores = scores_new[indexes, j, np.newaxis]
                cls_boxes = boxes_new[indexes, 4:8]
                cls_dets = np.hstack((cls_boxes, cls_scores))
                keep = nms(cls_dets)
                all_boxes[j][idx+delta] = cls_dets[keep, :]

            if max_per_image > 0:
                image_scores = np.hstack([all_boxes[j][idx+delta][:, -1]
                                          for j in range(1, imdb.num_classes)])
                if len(image_scores) > max_per_image:
                    image_thresh = np.sort(image_scores)[-max_per_image]
                    for j in range(1, imdb.num_classes):
                        keep = np.where(all_boxes[j][idx+delta][:, -1] >= image_thresh)[0]
                        all_boxes[j][idx+delta] = all_boxes[j][idx+delta][keep, :]

        idx += test_data.batch_size
        t3 = time.time() - t
        t = time.time()
        data_time += t1
        net_time += t2
        post_time += t3
        print 'testing {}/{} data {:.4f}s net {:.4f}s post {:.4f}s'.format(idx, imdb.num_images,
                                                                           data_time / idx * test_data.batch_size,
                                                                           net_time / idx * test_data.batch_size,
                                                                           post_time / idx * test_data.batch_size)

    det_file_new = os.path.join(imdb.result_path, imdb.name + '_detections.pkl')
    with open(det_file_new, 'wb') as f:
        cPickle.dump(all_boxes, f, protocol=cPickle.HIGHEST_PROTOCOL)

    # save all scores and boxes for rcnn
    results_all = {'scores_all': scores_all_new, 'boxes_all': boxes_all}
    det_file_all = os.path.join(imdb.result_path, imdb.name + '_detections_all.pkl')
    with open(det_file_all, 'wb') as f:
        cPickle.dump(results_all, f, cPickle.HIGHEST_PROTOCOL)

    info_str = imdb.evaluate_detections(all_boxes)
    if logger:
        logger.info('evaluate detections: \n{}'.format(info_str))

    print 'done'

if __name__ == '__main__':
    main()
