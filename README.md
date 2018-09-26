# Decoupled Classification Refinement
The is an official implementation of our ECCV2018 paper "Revisiting RCNN: On Awakening the Classification Power of Faster RCNN (https://arxiv.org/abs/1803.06799)"

## Introduction

**Decoupled Classification Refinement** is initially described in an [ECCV 2018 paper](https://arxiv.org/abs/1803.06799) (we call it DCR V1). 
It is further extended (we call it DCR V2) in a recent [tech report](https://arxiv.org/) (we will release it very soon). 
In this extension, we speed the original DCR V1 up by 3x with same accuracy. 
Unlike DCR V1 which requires a complicated two-stage training, DCR V2 is simpler and can be trained end-to-end. 

## News
* \[2018/09/26\] Added all COCO results. Code will be released very soon with a new tech report. Stay tuned!

## License

© University of Illinois at Urbnana-Champaign, 2018. Licensed under an MIT license.

## Citing DCR

If you find Decoupled Classification Refinement module useful in your research, please consider citing:
```
@inproceedings{cheng18revisiting,
author = {Cheng, Bowen and Wei, Yunchao and Shi, Honghui and Feris, Rogerio and Xiong, Jinjun and Huang, Thomas},
title = {Revisiting RCNN: On Awakening the Classification Power of Faster RCNN},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

## Main Results

For simplicity, all train/val/test-dev refer to COCO2017 train/val and COCO test-dev.  
Notes:
* all FPN models are trained with OHEM following [Deformable ConvNets](https://github.com/msracver/Deformable-ConvNets).
* Prefix _D-_ means adding Deformable Convolutions and replacing ROIPooling with Deformable ROIPooling.
* **NO** multi-scale train/test, **NO** soft-NMS, **NO** ensemble! These are purely single model results without any test-time tricks!

#### COCO test-dev
|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>AP</sub>  | <sub>AP@0.5</sub> | <sub>AP@0.75</sub>| <sub>AP@S</sub> | <sub>AP@M</sub> | <sub>AP@L</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|
| <sub>Faster R-CNN (2fc), ResNet-v1-101 </sub>           | <sub>trainval</sub> | <sub>test-dev</sub> | 30.5 | 52.2    |   31.8  | 9.7  | 32.3  | 48.3  | 
| <sub> + DCR V1, ResNet-v1-101/152 </sub>           | <sub>trainval</sub> | <sub>test-dev</sub> | 33.9 | 57.9    |   35.3  | 14.0  | 36.1  | 50.8  | 
| <sub> + DCR V2, ResNet-v1-101 </sub>           | <sub>trainval</sub> | <sub>test-dev</sub> | 34.3 | 57.7    |   35.8  | 13.8  | 36.7  | 51.1  | 
| <sub>D-Faster R-CNN (2fc), ResNet-v1-101</sub> | <sub>trainval</sub> | <sub>test-dev</sub> | 35.2 | 55.1    | 38.2    | 14.6  | 37.4  | 52.6  |
| <sub> + DCR V1, ResNet-v1-101/152</sub> | <sub>trainval</sub> | <sub>test-dev</sub> | 38.1 | 59.7    | 41.1    | 17.9  | 41.2  | 54.7  |
| <sub> + DCR V2, ResNet-v1-101</sub> | <sub>trainval</sub> | <sub>test-dev</sub> | 38.2 | 59.7    | 41.2    | 17.3  | 41.7  | 54.6  |
| <sub>FPN, ResNet-v1-101</sub>            | <sub>trainval</sub> | <sub>test-dev</sub> | 38.8 | 61.7 | 42.6 | 21.9  | 42.1  | 49.7  | 
| <sub> + DCR V1, ResNet-v1-101/152</sub>            | <sub>trainval</sub> | <sub>test-dev</sub> | 40.7 | 64.4 | 44.6 | 24.3  | 43.7  | 51.9  | 
| <sub> + DCR V2, ResNet-v1-101</sub>            | <sub>trainval</sub> | <sub>test-dev</sub> | 40.8 | 63.6 | 44.5 | 24.3  | 44.3  | 52.0  | 
| <sub>D-FPN, ResNet-v1-101</sub> | <sub>trainval</sub> | <sub>test-dev</sub> | 41.7 | 64.0 | 45.9 | 23.7  | 44.7  | 53.4  |
| <sub> + DCR V1, ResNet-v1-101/152</sub> | <sub>trainval</sub> | <sub>test-dev</sub> | 43.1 | 66.1 | 47.3 | 25.8  | 45.9  | 55.3  |
| <sub> + DCR V2, ResNet-v1-101</sub> | <sub>trainval</sub> | <sub>test-dev</sub> | 43.5 | 65.9 | 47.6 | 25.8  | 46.6  | 55.9  |

#### COCO validation
|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>AP</sub>  | <sub>AP@0.5</sub> | <sub>AP@0.75</sub>| <sub>AP@S</sub> | <sub>AP@M</sub> | <sub>AP@L</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|
| <sub>Faster R-CNN (2fc), ResNet-v1-101 </sub>           | <sub>train</sub> | <sub>val</sub> | 30.0 | 50.9    |   30.9  | 9.9  | 33.0  | 49.1  | 
| <sub> + DCR V1, ResNet-v1-101/152 </sub>           | <sub>train</sub> | <sub>val</sub> | 33.1 | 56.3    |   34.2  | 13.8  | 36.2  | 51.5  | 
| <sub> + DCR V2, ResNet-v1-101 </sub>           | <sub>train</sub> | <sub>val</sub> | 33.6 | 56.7    |   34.7  | 13.5  | 37.1  | 52.2  | 
| <sub>D-Faster R-CNN (2fc), ResNet-v1-101</sub> | <sub>train</sub> | <sub>val</sub> | 34.4 | 53.8    | 37.2    | 14.4  | 37.7  | 53.1  |
| <sub> + DCR V1, ResNet-v1-101/152</sub> | <sub>train</sub> | <sub>val</sub> | 37.2 | 58.6    | 39.9    | 17.3  | 41.2  | 55.5  |
| <sub> + DCR V2, ResNet-v1-101</sub> | <sub>train</sub> | <sub>val</sub> | 37.5 | 58.6    | 40.1    | 17.2  | 42.0  | 55.5  |
| <sub>FPN, ResNet-v1-101</sub>            | <sub>train</sub> | <sub>val</sub> | 38.2 | 61.1 | 41.9 | 21.8  | 42.3  | 50.3  | 
| <sub> + DCR V1, ResNet-v1-101/152</sub>            | <sub>train</sub> | <sub>val</sub> | 40.2 | 63.8 | 44.0 | 24.3  | 43.9  | 52.6  | 
| <sub> + DCR V2, ResNet-v1-101</sub> | <sub>train</sub> | <sub>val</sub> | 40.3 | 62.9 | 43.7 | 24.3  | 44.6  | 52.7  |
| <sub>D-FPN + OHEM, ResNet-v1-101</sub> | <sub>train</sub> | <sub>val</sub> | 41.4 | 63.5 | 45.3 | 24.4  | 45.0  | 55.1  |
| <sub> + DCR V1, ResNet-v1-101/152</sub> | <sub>train</sub> | <sub>val</sub> | 42.6 | 65.3 | 46.5 | 26.4  | 46.1  | 56.4  |
| <sub> + DCR V2, ResNet-v1-101</sub> | <sub>train</sub> | <sub>val</sub> | 42.8 | 65.1 | 46.8 | 27.1  | 46.6  | 56.1  |

## Contact
Bowen Cheng (bcheng9 AT illinois DOT edu)
