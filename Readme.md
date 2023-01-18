# Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model 
### Di Wang, Qiming Zhang, Yufei Xu, Jing Zhang, Bo Du, Dacheng Tao and Liangpei Zhang

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#results-and-models">Results & Models</a> |
  <a href="#usage">Usage</a> |
</p >

<p align="left">
<a href="https://arxiv.org/abs/2208.03987"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
<a href="https://ieeexplore.ieee.org/document/9956816"><img src="https://img.shields.io/badge/TGRS-Paper-blue"></a>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/object-detection-in-aerial-images-on-dota-1)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dota-1?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/object-detection-in-aerial-images-on-dior-r)](https://paperswithcode.com/sota/object-detection-in-aerial-images-on-dior-r?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/aerial-scene-classification-on-ucm-50-as)](https://paperswithcode.com/sota/aerial-scene-classification-on-ucm-50-as?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/aerial-scene-classification-on-aid-20-as)](https://paperswithcode.com/sota/aerial-scene-classification-on-aid-20-as?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/aerial-scene-classification-on-aid-50-as)](https://paperswithcode.com/sota/aerial-scene-classification-on-aid-50-as?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/aerial-scene-classification-on-nwpu-10-as)](https://paperswithcode.com/sota/aerial-scene-classification-on-nwpu-10-as?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/aerial-scene-classification-on-nwpu-20-as)](https://paperswithcode.com/sota/aerial-scene-classification-on-nwpu-20-as?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/semantic-segmentation-on-isprs-potsdam)](https://paperswithcode.com/sota/semantic-segmentation-on-isprs-potsdam?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/semantic-segmentation-on-isaid)](https://paperswithcode.com/sota/semantic-segmentation-on-isaid?p=advancing-plain-vision-transformer-towards)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/advancing-plain-vision-transformer-towards/semantic-segmentation-on-loveda)](https://paperswithcode.com/sota/semantic-segmentation-on-loveda?p=advancing-plain-vision-transformer-towards)

## Current applications

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **VSA: Please see [ViTAE-VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;

> **Remote Sensing Pretraining: Please see [ViTAE-Transformer-Remote-Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)**;

## Updates

### 2023.01.18

Our models have been supported by [LuoJiaNET](https://github.com/WHULuoJiaTeam/luojianet), please refer to [RS-Vision-Foundation-Models](https://github.com/WHULuoJiaTeam/Model_Zoo/tree/main/RS_Vision_Foundation_Models) for more details.

### 2022.11.21

The early access is available! [TGRS link](https://ieeexplore.ieee.org/document/9956816)

### 2022.11.15

The arXiv has been updated! [arXiv link](https://arxiv.org/abs/2208.03987)

### 2022.11.06

The paper has been accepted by IEEE TGRS!

### 2022.10.11 

The codes, configs and training logs of segmentation in fintuning are released!

### 2022.10.09 

The codes, configs and training logs of detection in fintuning are released!

### 2022.10.08 

The codes of pretraining and classification in fintuning are released!

### 2022.09.19 

The codes and training logs of the [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA) have been released, which is the foundation of our RVSA.

## Introduction

This repository contains codes, models and test results for the paper "[Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model](https://arxiv.org/abs/2208.03987)".

We resort to plain vision transformers with about 100M and make the first attempt to propose large vision models customized for RS tasks and propose a new rotated varied-size window attention (RVSA) to substitute the original full attention to handle the large image size and objects of various orientations in RS images. The RVSA could significantly reduce the computational cost and memory footprint while learn better object representation by extracting rich context from the generated diverse windows.

<figure>
<img src=Figs/framework.png>
<figcaption align = "center"><b>Fig.1 - The pipeline of pretraining and finetuning. </b></figcaption>
</figure>

&emsp;

<figure>
<img src=Figs/vit_rvsa.png>
<figcaption align = "center"><b>Fig.2 - The structure and block of the adopted plain vision transformer, and the proposed RVSA. </b></figcaption>
</figure>


## Results and Models

### Pretraining 

#### MillionAID
|Pretrain|Backbone | Input size | Params (M) | Pretrained model|
|-------|-------- | ----------  | ----- | ----- |
| MAE | ViT-B | 224 × 224 | 86| [Weights](https://1drv.ms/u/s!AimBgYV7JjTlgUPBC6cvpo4oZDSR?e=kNCAhO) |
| MAE | ViTAE-B | 224 × 224 | 89 | [Weights](https://1drv.ms/u/s!AimBgYV7JjTlgUIde2jzcjrrWasP?e=gyLn29) |

### Object Detection

#### DOTA-V1.0 Single-Scale
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | :---: | :------: | :---: |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 78.75 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVJM4Znng50US8KD?e=o4MRMQ) |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 78.61 |[Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.py)| [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVOYOQ3d_QS1Vkco?e=aOrzfJ) |  |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 78.96  | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVAG59CJWkxA7f5U?e=hjR3Bx) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 78.99 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVEPLn_D1ph3EQ5Q?e=MXLorq) |


#### DOTA-V1.0 Multi-Scale
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | :---: | :------: | :---: |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 81.01 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.log)| [Model](https://1drv.ms/u/s!AimBgYV7JjTlgU6SB-_d-xk5Fh5N?e=KSkQqA) |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 80.80 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgccA-3Vs3J5SZqB0lg?e=0Wrast) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 81.24 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.log) |  [Model](https://1drv.ms/u/s!AimBgYV7JjTlgccBdC967NGcR_FcvA?e=6q2Vd2) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 81.18 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgccCSURJPWl1jPdeIA?e=HtYQgD) |

#### DIOR-R
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | :---: | :------: | :---: |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 70.67 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUjzxMV17pmGV-md?e=b0z0Gn) |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 70.85 |  [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr15.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr15.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUv2gHKLWOwa0_AD?e=gpXnKG) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 70.95 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10.py) |[Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUqfEy2J7BTmKKlK?e=nsoWwM) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 71.05 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10.py)| [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Object%20Detection/logs/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10.log) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUkvsiV0YJkWWyuY?e=IMHGAR) |

### Scene Classification

|Pretrain | Backbone | UCM-55 | AID-28 | AID-55 | NWPU-19 | NWPU-28 |
|----------|-------- | --------- | ------- | --- | ------ | --- | 
| MAE | ViT-B + RVSA | 99.70 | 96.92 | 98.33 | 93.79 | 95.49 |
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWwwNEk-zZN8Zddb?e=RIfcn9) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXaE3XBegpd9Awqb?e=snF1gg) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXvqgHU6iJ4aJ0L1?e=XXLbqy) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgW84yjf3C-RUNmOc?e=LYJLpB) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXPYuJWRpRRzu9B_?e=NJjcXB) |
| MAE | ViT-B + RVSA $^ \Diamond$ | 99.58 | 96.86 | 98.44 | 93.74 | 95.45 |
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWnicUuVKBIGZAW0?e=xjfV8z) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXhlvwoQP1Sbb9RG?e=gnMNTQ) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXzoV-yx80lxcwR7?e=CL4k33) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXBJyGduYfF5SCB7?e=LgdIXb) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXROC5R_8d3CViay?e=UdJPVM) |
| MAE | ViTAE-B + RVSA | 99.56 | 97.03 | 98.48 | 93.93 | 95.69 |
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWv-ct-hZwKbSqdg?e=tRgL4J) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXWc6RWse1WxjuI2?e=UtGAm3) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXkt_Lv6GljM4AkG?e=1wk8K3) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgW0vDOaTsch8Spqo?e=km6axJ) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXLPtT4hVj4diQax?e=7c9k0U) |
| MAE | ViTAE-B + RVSA $^ \Diamond$ | 99.50 | 97.01 | 98.50 | 93.92 | 95.66|
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWqe8N6wagHiPiFe?e=7gIVAn) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXch-D4GJ1Gutg3J?e=U3W9AJ) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXocg8ael_cdmdhK?e=pPJGRm) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgW7bR8HfcpK4wig4?e=ZYOmdS) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXFEJ_T2wZRVhiow?e=CQ6son) |

### Semantic Segmentation

#### ISPRS Potsdam

| Method | Pretrain | Backbone | Crop size | Lr schd | OA | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | :------: | :---: | :-----: |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 90.60 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vit_base_win_rvsa_v3_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vit_base_win_rvsa_v3_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggsyQuSY6EYcOj2s?e=0k1MBH) |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 90.77 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vit_base_win_rvsa_v3_kvdiff_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vit_base_win_rvsa_v3_kvdiff_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggyehoVuz4A7i9A6?e=WehMl8) |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 91.22 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_wsz7_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vitae_nc_base_rvsa_v3_wsz7_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggmAnAEe7hb5p8id?e=16oojH) |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 91.15 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggqk2eQmyCF2E2Y2?e=hYxZQv) |

#### iSAID

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | :------: | :---: | :-----: |
| UperNet| MAE | ViT-B + RVSA | 896 × 896 | 160k | 63.76 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vit_base_win_rvsa_v3_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vit_base_win_rvsa_v3_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggHtXbZ3T7hC-oqL?e=AGrYbz) |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 63.85 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vit_base_win_rvsa_v3_kvdiff_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vit_base_win_rvsa_v3_kvdiff_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggQOfObKs86ZD-fd?e=Wz3MUe) |
| UperNet| MAE | ViTAE-B + RVSA | 896 × 896 | 160k | 63.48 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vitae_nc_base_rvsa_v3_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggLl2cfm1Ro-ekra?e=EFCghU) |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 64.49 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_896x896_160k_isaid_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggPWSalF-CBKETHD?e=BRe1HQ) |

#### LoveDA

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | :------: | :---: | :-----: |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 51.95 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vit_base_win_rvsa_v3_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vit_base_win_rvsa_v3_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggcH-UQUNSDg8AMh?e=h1fiW0) |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 51.95 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vit_base_win_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vit_base_win_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgghifTHhpfc5kPZW?e=HjV8Ib) |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 52.26 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vitae_nc_base_rvsa_v3_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggX66jVrAoWdZrK8?e=8hZsse) |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 52.44 | [Config](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.py) | [Log](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA/blob/main/Semantic%20Segmentation/logs/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.log.json) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggaDnTijEqVrAHgg?e=CJHcTc) |

## Usage

Environment:

- Python 3.8.5
- Pytorch 1.9.0+cu111
- torchvision 0.10.0+cu111
- timm 0.4.12
- mmcv-full 1.4.1

### Pretraining & Finetuning-Classification

#### Pretraining (8 × A100 GPUs, 3~5 days)

1. Preparing the MillionAID: Download the [MillionAID](https://captain-whu.github.io/DiRS/). Here, we use previous `train_labels.txt` and `valid_labels.txt` of the [ViTAE-Transformer-Remote-Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing), which contain labels. However, since we conduct the ***unsupervised pretraining***, the labels are not necessary. It is easy for users to record image names and revise corresponding codes `MAEPretrain_SceneClassification/util/datasets.py/class MillionAIDDataset`.

2. Pretraining: take ViT-B as an example (batchsize: 2048=8*256)

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 10000 main_pretrain.py \
--dataset 'millionAID' --model 'mae_vit_base_patch16' \
--batch_size 256 --epochs 1600 --warmup_epochs 40 \
--input_size 224 --mask_ratio 0.75 \
--blr 1.5e-4  --weight_decay 0.05 --gpu_num 8 \
--output_dir '../mae-main/output/'
```
*Note: Padding the convolutional kernel of PCM in the pretrained ViTAE-B with `convertK1toK3.py` for finetuning.*

3. Linear probe: an example of evaluating the pretrained ViT-B on UCM-55

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 10000 main_linprobe.py \
--dataset 'ucm' --model 'vit_base_patch16' \
--batch_size 256 --epochs 100 --warmup_epochs 10 \
--blr 1e-1  --weight_decay 0 --tag 0 \
--finetune '../mae-main/output/millionAID_224/1600_0.75_0.00015_0.05_2048/checkpoint-1599.pth'
```

#### Finetuning evaluation for pretraining & Finetuning-Classification

For instance, finetuning ViTAE-B + RVSA on NWPU-28

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 20000 main_finetune.py \
--dataset 'nwpu' --model 'vitae_nc_base_win_rvsa' --input_size 224 --postfix 'sota' \
--batch_size 64 --epochs 200 --warmup_epochs 5 \
--blr 1e-3  --weight_decay 0.05 --split 28 --tag 0 --exp_num 1 \
--finetune '../mae-main/output/mae_vitae_base_pretrn/millionAID_224/1600_0.75_0.00015_0.05_2048/checkpoint-1599-transform-no-average.pth'
```

### Finetuning-Detection & Finetuning-Segmentation

Since we use OBBDetection and MMSegmenation to implement corresponding detection or segmentation models, we only provide necessary config and backbone files. The main frameworks are both in [ViTAE-Transformer-Remote-Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)

```
git clone https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing.git
```

The installation and dataset preparation can separately refer [OBBDetection-installation](https://github.com/jbwang1997/OBBDetection/blob/master/docs/install.md) and 
[MMSegmentation-installation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)

Then put these files into corresponding folders.

For convenience, we preserve the relative path for users to find files.

For example, put `./Object Detection/mmdet/models/backbones/vit_win_rvsa_v3_wsz7.py` into `ViTAE-Transformer-Remote-Sensing/Object Detection/mmdet/models/backbones`

#### Training-Detection

First, `cd ./Object Detection` 

Then, we provide several examples. For instance, 

Training the Oriented-RCNN with ViT-B + RVSA on DOTA-V1.0 multi-scale detection dataset with 2 GPUs

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=40000 tools/train.py \
configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_wsz7_fpn_1x_dota10_ms_lr1e-4_ldr75_dpr15.py \
--launcher 'pytorch' --options 'find_unused_parameters'=True
```

Training the Oriented-RCNN with ViTAE-B + RVSA $^ \Diamond$ backbone on DIOR-R detection dataset with 1 GPU

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=40001 tools/train.py \
configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10.py \
--launcher 'pytorch' --options 'find_unused_parameters'=True
```

#### Inference-Detection

Predicting the saving detection map using ViT-B + RVSA $^ \Diamond$ on DOTA-V1.0 scale-scale detection dataset

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15.py \
../OBBDetection/work_dirs/faster/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15/latest.pth \
--format-only --show-dir work_dirs/save/faster/display/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15 \
--options save_dir='work_dirs/save/faster/full_det/faster_rcnn_orpn_our_rsp_vit-base-win-rvsa_v3_kvdiff_wsz7_fpn_1x_dota10_lr1e-4_ldr75_dpr15' nproc=1
```

Evaluating the detection maps predicted by ViTAE-B + RVSA on DIOR-R dataset

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10.py \
../OBBDetection/work_dirs/faster/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10/latest.pth \
--out work_dirs/save/faster/full_det/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10/det_result.pkl --eval 'mAP' \
--show-dir work_dirs/save/faster/display/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_1x_dior_lr1e-4_ldr75_dpr10
```

*Note: the pathes of saved maps and outputs should be constructed before evaluating the DIOR-R testing set.*

#### Training & Evaluation-Segmentation

`cd ./Semantic Segmentation` 

Training and evaluation the UperNet with ViT-B + RVSA backbone on Potsdam dataset:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=30000 tools/train.py \
configs/vit_base_win/upernet_vit_base_win_rvsa_v3_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.py \
--launcher 'pytorch' --cfg-options 'find_unused_parameters'=True
```

*Note: when training on the LoveDA, please add `--no-validate`*

Inference the LoveDA dataset for online evaluation using the UperNet with ViTAE-B + RVSA $^ \Diamond$ backbone

```
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/vit_base_win/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16.py \
../mmsegmentation-master/work_dirs/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16/latest.pth \
--format-only --eval-options imgfile_prefix="work_dirs/display/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16/result" \
--show-dir work_dirs/display/upernet_vitae_nc_base_rvsa_v3_kvdiff_wsz7_512x512_160k_loveda_dpr10_lr6e5_lrd90_ps16/rgb
```

### ***When finetuning with more than one GPU for detection or segmentation, please use `nn.SyncBatchNorm` in the NormalCell of ViTAE models.***

## Citation

If this repo is useful for your research, please consider citation

```
@ARTICLE{wang_vitrvsa_2022,
  author={Wang, Di and Zhang, Qiming and Xu, Yufei and Zhang, Jing and Du, Bo and Tao, Dacheng and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2022.3222818}
  }
  
@ARTICLE{wang_rsp_2022,  
author={Wang, Di and Zhang, Jing and Du, Bo and Xia, Gui-Song and Tao, Dacheng},  
journal={IEEE Transactions on Geoscience and Remote Sensing},   
title={An Empirical Study of Remote Sensing Pretraining},   
year={2022},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/TGRS.2022.3176603}}
```

## Statement

This project is for research purpose only. For any other questions please contact [di.wang at gmail.com](mailto:wd74108520@gmail.com) .

## References

The codes of Pretraining & Scene Classification part mainly from [MAE](https://github.com/facebookresearch/mae).

## Relevant Projects
[1] <strong>An Empirical Study of Remote Sensing Pretraining, IEEE TGRS, 2022</strong> | [Paper](https://ieeexplore.ieee.org/document/9782149) | [Github](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)
<br><em>&ensp; &ensp; &ensp;Di Wang<sup>&#8727;</sup>, Jing Zhang<sup>&#8727;</sup>, Bo Du, Gui-Song Xia and Dacheng Tao</em>


