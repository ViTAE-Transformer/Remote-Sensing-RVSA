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

# Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model

<p align="left">
<a href="https://arxiv.org/abs/2208.03987"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a>
</p>

## Current applications

> **ViTAE: Please see [ViTAE-Transformer](https://github.com/ViTAE-Transformer/ViTAE-Transformer)**;

> **VSA: Please see [ViTAE-VSA](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing/tree/main/Scene%20Recognition)**;

> **Matting: Please see [ViTAE-Transformer for matting](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Matting)**;

> **Remote Sensing Pretraining: Please see [ViTAE-Transformer-Remote-Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)**;



## Introduction

This repository contains codes, models and test results for the paper "[Advancing Plain Vision Transformer Towards Remote Sensing Foundation Model](https://arxiv.org/abs/2208.03987)".

We resort to plain vision transformers with about 100M and make the first attempt to propose large vision models customized for RS tasks and propose a new rotated varied-size window attention (RVSA) to substitute the original full attention to handle the large image size and objects of various orientations in RS images. The RVSA could significantly reduce the computational cost and memory footprint while learn better object representation by extracting rich context from the generated diverse windows.

<figure>
<div align="center">
<img src=Figs/vit_rvsa.png width="70%">
</div>
<figcaption align = "center"><b>Fig. - The structure and block of the adopted plain vision transformer, and the proposed RVSA. </b></figcaption>
</figure>

## Results and Models

### Pretraining 

#### MillionAID
|Pretrain|Backbone | Input size | Params (M) | Pretrained model|
|-------|-------- | ----------  | ----- | ----- |
| MAE | ViT-B | 224 × 224 | 86| Coming Soon |
| MAE | ViTAE-B | 224 × 224 | 89 | Coming Soon |

### Object Detection

#### DOTA-V1.0
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | --- | ------ | --- |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 81.01 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 80.51 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 81.16 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 80.97 | Coming Soon | Coming Soon | Coming Soon |

#### DIOR-R
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | --- | ------ | --- |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 70.67 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 70.85 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 70.95 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 71.05 | Coming Soon | Coming Soon | Coming Soon |

### Semantic Segmentation

### ISPRS Potsdam

| Method | Pretrain | Backbone | Crop size | Lr schd | OA | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 90.60 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 90.77 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 91.22 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 91.15 | Coming Soon | Coming Soon | Coming Soon |

### iSAID

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 896 × 896 | 160k | 63.76 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 63.85 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA | 896 × 896 | 160k | 63.48 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 64.49 | Coming Soon | Coming Soon | Coming Soon |

### LoveDA

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 51.95 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 51.95 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 52.26 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 52.44 | Coming Soon | Coming Soon | Coming Soon |

