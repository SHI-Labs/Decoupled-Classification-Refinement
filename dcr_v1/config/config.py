# --------------------------------------------------------
# Decoupled Classification Refinement
# Copyright (c) 2018 University of Illinois
# Licensed under The MIT License [see LICENSE for details]
# Modified by Bowen Cheng
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.MXNET_VERSION = ''
config.output_path = './output/dcr_v1'
config.gpus = ''

# default network
config.network = edict()
config.network.network = ''
config.network.num_layers = 0
config.network.pretrained = ''
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([0, 0, 0])
config.network.FIXED_PARAMS = ['gamma', 'beta']

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'PascalVOC'
config.dataset.image_set = '2007_trainval'
config.dataset.test_image_set = '2007_test'
config.dataset.root_path = './data'
config.dataset.dataset_path = './data/VOCdevkit'
config.dataset.num_classes = 21
config.dataset.image_shape = '3,224,224'

# default train
config.train = edict()
config.train.kv_store = 'device'
config.train.begin_epoch = 0
config.train.end_epoch = 0
config.train.lr = 0
config.train.lr_factor = 0.1
config.train.lr_step = ''

config.train.warmup = False
config.train.warmup_lr = 0.01
config.train.warmup_step = 0

config.train.optimizer = 'sgd'
config.train.mom = 0.9
config.train.wd = 0.0001

config.train.BATCH_IMAGES = 1
config.train.FLIP = False
config.train.SHUFFLE = False
config.train.RESUME = False
config.train.sample_per_image = 32
config.train.roi_enlarge_scale = 1.0

config.train.hard_fp_score = 0.3
config.train.hard_fg_score = 0.7

config.train.FG_THRESH = 0.5
config.train.BG_THRESH_HI = 0.5
config.train.BG_THRESH_LO = 0.0

config.train.SAMPLE_RCNN = False
config.train.SAMPLE_HARD_FG = False
config.train.SAMPLE_DEFAULT = False
config.train.SAMPLE_RANDOM = False
config.train.SAMPLE_BG = True
config.train.SAMPLE_FG = True

config.train.model_prefix = ''

config.train.gc_type = 'none'
config.train.gc_threshold = 0.5
config.train.workspace = 1024
config.train.dtype = 'float32'
config.train.monitor = 0

config.train.disp_batches = 20
config.train.base_detector = ''

config.train.fc_lr = 10
config.train.fc_init = 'normal'

config.test = edict()
config.test.test_epoch = 0
config.test.NMS = 0
config.test.max_per_image = 300
config.test.thresh = float(1e-3)
config.test.RPN_POST_NMS_TOP_N = 300
config.test.BATCH_IMAGES = 1
config.test.NUM_PROPOSALS = 300

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                raise ValueError("key must exist in config.py")
    config.TRAIN = edict()
    config.TRAIN = config.train.copy()
