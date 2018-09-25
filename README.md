# Decoupled Classification Refinement
The is an official implementation of our ECCV2018 paper "Revisiting RCNN: On Awakening the Classification Power of Faster RCNN (https://arxiv.org/abs/1803.06799)"

## Introduction

**Decoupled Classification Refinement** is initially described in an [ECCV 2018 paper](https://arxiv.org/abs/1803.06799) (we call it DCR V1). 
It is further extended (we call it DCR V2) in a recent [tech report](https://arxiv.org/) (we will release it very soon). 
In this extension, we speed the original DCR V1 up by 3x with same accuracy. 
Unlike DCR V1 which requires a complicated two-stage training, DCR V2 is simpler and can be trained end-to-end. 

## License

© University of Illinois, 2018. Licensed under an MIT license.

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

|                                 | <sub>training data</sub> | <sub>testing data</sub>  | <sub>mAP</sub>  | <sub>mAP@0.5</sub> | <sub>mAP@0.75</sub>| <sub>mAP@S</sub> | <sub>mAP@M</sub> | <sub>mAP@L</sub> |
|---------------------------------|---------------|---------------|------|---------|---------|-------|-------|-------|
| <sub>Faster R-CNN (2fc), ResNet-v1-101 </sub>           | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 30.0 | 50.9    |   30.9  | 9.9  | 33.0  | 49.1  | 
| <sub>Faster R-CNN (2fc) DCR V1, ResNet-v1-101/152 </sub>           | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 33.1 | 56.3    |   34.2  | 13.8  | 36.2  | 51.5  | 
| <sub>Faster R-CNN (2fc) DCR V2, ResNet-v1-101 </sub>           | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 33.6 | 56.7    |   34.7  | 13.5  | 37.1  | 52.2  | 
| <sub>Deformable Faster R-CNN (2fc), </br>ResNet-v1-101</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 34.4 | 53.8    | 37.2    | 14.4  | 37.7  | 53.1  |
| <sub>Deformable Faster R-CNN (2fc) DCR V1, </br>ResNet-v1-101/152</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 37.2 | 58.6    | 39.9    | 17.3  | 41.2  | 55.5  |
| <sub>Deformable Faster R-CNN (2fc) DCR V2, </br>ResNet-v1-101</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 37.5 | 58.6    | 40.1    | 17.2  | 42.0  | 55.5  |
| <sub>FPN+OHEM, ResNet-v1-101</sub>            | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 38.2 | 61.1 | 41.9 | 21.8  | 42.3  | 50.3  | 
| <sub>FPN+OHEM DCR V1, ResNet-v1-101/152</sub>            | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 40.2 | 63.8 | 44.0 | 24.3  | 43.9  | 52.6  | 
| <sub>FPN+OHEM DCR V2, ResNet-v1-101/</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 40.3 | 62.9 | 43.7 | 24.3  | 44.6  | 52.7  |
| <sub>Deformable FPN + OHEM, ResNet-v1-101</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 41.4 | 63.5 | 45.3 | 24.4  | 45.0  | 55.1  |
| <sub>Deformable FPN + OHEM DCR V1, ResNet-v1-101/152</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 42.6 | 65.3 | 46.5 | 26.4  | 46.1  | 56.4  |
| <sub>Deformable FPN + OHEM DCR V2, ResNet-v1-101</sub> | <sub>coco2017 train</sub> | <sub>coco2017 val</sub> | 42.8 | 65.1 | 46.8 | 27.1  | 46.6  | 56.1  |
