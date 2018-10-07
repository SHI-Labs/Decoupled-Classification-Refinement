#!/bin/bash

CFG=$1

python dcr_v1/train_rcnn.py --cfg ${CFG}
python dcr_v1/rcnn_rescore_combined_fast.py --cfg ${CFG}
