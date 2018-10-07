# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import _init_paths
import os
import sys
import argparse
import logging
import pprint

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

from config.config import config, update_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Resnet')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    parser.add_argument('--gpu', help='suppress gpu setting in config files',
                        default=None, type=str)

    args = parser.parse_args()
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

from utils.create_logger import create_logger
from utils.load_data import load_rcnn_roidb, merge_roidb, filter_roidb
from utils.load_model import load_param
# from utils.PrefetchingIter import PrefetchingIter
# from mxnet.io import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler
import mxnet as mx
from core.loader import ROIIter, PreFetchROIIter, PrefetchingIter
from core.fit import _save_model
from core import callback, metric

def train_net(args, ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # load symbol
    from importlib import import_module
    net = import_module('symbols.' + config.network.network)
    kargs = {'num_classes': config.dataset.num_classes,
             'num_layers': config.network.num_layers,
             'image_shape': config.dataset.image_shape,
             'conv_workspace': config.train.workspace,
             'dtype': config.train.dtype,
             'fc_lr': config.train.fc_lr}

    sym = net.get_symbol(**kargs)

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.train.BATCH_IMAGES * batch_size

    # train
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    image_sets = [iset for iset in config.dataset.image_set.split('+')]

    # add config name
    det_file_name = config.train.base_detector
    roidbs = [load_rcnn_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                              append_gt=True, flip=config.train.FLIP, det_file_name=det_file_name)
              for image_set in image_sets]
    roidb = merge_roidb(roidbs)
    roidb = filter_roidb(roidb, config)

    # load training data
    # train_data = ROIIter(roidb, config, batch_size=input_batch_size, shuffle=config.train.SHUFFLE)
    train_data = PreFetchROIIter(roidb, config, batch_size=input_batch_size, shuffle=config.train.SHUFFLE)

    if args.test_io:
        from core.test_io import test_io
        test_io(train_data, end_epoch, config.train.disp_batches, input_batch_size, logger)

        return

    # load and initialize params
    if config.train.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        data_shape = (1, 3, 224, 224)
        arg_names = sym.list_arguments()
        arg_shapes, _, _ = sym.infer_shape(data=data_shape)
        for x in zip(arg_names, arg_shapes):
            # resnet/resnext
            if 'fc1' in x[0]:
                if 'weight' in x[0]:
                    if config.train.fc_init == 'xavier':
                        try:
                            initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
                            initializer(mx.init.InitDesc('fc1_weight'), arg_params['fc1_weight'])
                        except:
                            arg_params[x[0]] = mx.random.normal(0, 0.01, shape=x[1], ctx=mx.cpu())
                    else:
                        arg_params[x[0]] = mx.random.normal(0, 0.01, shape=x[1], ctx=mx.cpu())
                elif 'bias' in x[0]:
                    arg_params[x[0]] = mx.nd.zeros(x[1], ctx=mx.cpu())
            # dpn
            if 'fc6' in x[0]:
                if 'weight' in x[0]:
                    if config.train.fc_init == 'xavier':
                        try:
                            initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
                            initializer(mx.init.InitDesc('fc6_weight'), arg_params['fc6_weight'])
                        except:
                            arg_params[x[0]] = mx.random.normal(0, 0.01, shape=x[1], ctx=mx.cpu())
                    else:
                        arg_params[x[0]] = mx.random.normal(0, 0.01, shape=x[1], ctx=mx.cpu())
                elif 'bias' in x[0]:
                    arg_params[x[0]] = mx.nd.zeros(x[1], ctx=mx.cpu())

    fixed_param_prefix = config.network.FIXED_PARAMS
    fixed_param_names = list()
    if fixed_param_prefix is not None:
        for name in sym.list_arguments():
            for fixed_prefix in fixed_param_prefix:
                if fixed_prefix in name:
                    fixed_param_names.append(name)

    # kvstore
    kv = mx.kvstore.create(config.train.kv_store)
    # save model
    checkpoint = _save_model(prefix, kv.rank)

    # devices for training
    devs = mx.cpu() if config.gpus is None or config.gpus is '' else [
        mx.gpu(int(i)) for i in config.gpus.split(',')]

    # decide learning rate
    base_lr = lr
    lr_factor = config.train.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(roidb) / input_batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.train.warmup, config.train.warmup_lr,
                                              config.train.warmup_step)

    # create model
    model = mx.mod.Module(
        context=devs,
        symbol=sym,
        logger=logger,
        fixed_param_names=fixed_param_names
    )
    # create solver

    optimizer_params = {
        'learning_rate': lr,
        'wd': config.train.wd,
        'lr_scheduler': lr_scheduler,
        'rescale_grad': 1.0,
        'clip_gradient': None}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag'}
    if config.train.optimizer in has_momentum:
        optimizer_params['momentum'] = config.train.mom

    monitor = mx.mon.Monitor(config.train.monitor, pattern=".*") if config.train.monitor > 0 else None

    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = mx.metric.CompositeEvalMetric()
    eval_metrics.add(metric.RCNNFGAccuracy())
    eval_metrics.add(metric.RCNNAccMetric())
    eval_metrics.add(metric.RCNNLogLossMetric())

    # callbacks that run after each batch
    batch_end_callbacks = callback.Speedometer(train_data.batch_size, frequent=config.train.disp_batches)

    # run
    if not isinstance(train_data, PrefetchingIter):
        train_data = PrefetchingIter(train_data)

    print (train_data.batch_size, train_data.provide_data, train_data.provide_label)

    model.fit(train_data,
              eval_metric = eval_metrics,
              begin_epoch=begin_epoch if config.TRAIN.RESUME else 0,
              num_epoch=end_epoch,
              kvstore=kv,
              optimizer=config.train.optimizer,
              optimizer_params=optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              batch_end_callback=batch_end_callbacks,
              epoch_end_callback=checkpoint,
              allow_missing=False,
              monitor=monitor)


def main():
    print('Called with argument:', args)
    if args.gpu is not None:
        config.gpus = args.gpu
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_epoch, config.train.model_prefix,
              config.train.begin_epoch, config.train.end_epoch, config.train.lr, config.train.lr_step)

if __name__ == '__main__':
    main()
