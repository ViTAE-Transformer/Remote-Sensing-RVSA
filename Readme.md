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

> **VSA: Please see [ViTAE-VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA)**;

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

#### DOTA-V1.0 Single-Scale
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | --- | ------ | --- |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 78.75 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 78.61 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 78.96 | Coming Soon | Coming Soon | Coming Soon |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 78.99 | Coming Soon | Coming Soon | Coming Soon |


#### DOTA-V1.0 Multi-Scale
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

### Scene Classification

|Pretrain | Backbone | UCM-55 | AID-28 | AID-55 | NWPU-19 | NWPU-28 |
|----------|-------- | --------- | ------- | --- | ------ | --- | 
| MAE | ViT-B + RVSA | 99.70 | 96.92 | 98.33 | 93.79 | 95.49 |
|     |              |Coming Soon | Coming Soon | Coming Soon | Coming Soon | Coming Soon |
| MAE | ViT-B + RVSA $^ \Diamond$ | 99.58 | 96.86 | 98.44 | 93.74 | 95.45 |
|     |              |Coming Soon | Coming Soon | Coming Soon | Coming Soon | Coming Soon |
| MAE | ViTAE-B + RVSA | 99.56 | 97.03 | 98.48 | 93.93 | 95.69 |
|     |              |Coming Soon | Coming Soon | Coming Soon | Coming Soon | Coming Soon |
| MAE | ViTAE-B + RVSA $^ \Diamond$ | 99.50 | 97.01 | 98.50 | 93.92 | 95.66|
|     |              |Coming Soon | Coming Soon | Coming Soon | Coming Soon | Coming Soon |

### Semantic Segmentation

#### ISPRS Potsdam

| Method | Pretrain | Backbone | Crop size | Lr schd | OA | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 90.60 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 90.77 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 91.22 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 91.15 | Coming Soon | Coming Soon | Coming Soon |

#### iSAID

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 896 × 896 | 160k | 63.76 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 63.85 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA | 896 × 896 | 160k | 63.48 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 64.49 | Coming Soon | Coming Soon | Coming Soon |

#### LoveDA

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 51.95 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 51.95 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 52.26 | Coming Soon | Coming Soon | Coming Soon |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 52.44 | Coming Soon | Coming Soon | Coming Soon |

## Usage

### Pretraining & Finetuning-Classification

To be continued。。

### Finetuning-Detection & Finetuning-Segmentation

Since we use OBBDetection and MMSegmenation to implement corresponding detection or segmentation models, we only provide necessary config and backbone files. The main frameworks are both in [ViTAE-Transformer-Remote-Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)

```
git clone https://github.com/DotWang/Remote-Sensing-Pretraining.git
```

The installation and dataset preparation can separately refer [OBBDetection-installation](https://github.com/jbwang1997/OBBDetection/blob/master/docs/install.md) and 
[MMSegmentation-installation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)

Then put these files into corresponding folders.

For convenience, we preserve the relative path for users to find files.

For example, put `./Object Detection/mmdet/models/backbones/vit_win_rvsa_v3_wsz7.py` into `ViTAE-Transformer-Remote-Sensing/mmdet/models/backbones`

*Note: When finetuning with more than one GPU, please use `nn.SyncBatchNorm` in the NormalCell of ViTAE models.*

#### Training-Detection

`cd ./Object Detection` 

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
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10.py \
../OBBDetection/work_dirs/faster/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10/latest.pth \
--out work_dirs/save/faster/full_det/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10/det_result.pkl --eval 'mAP' \
--show-dir work_dirs/save/faster/display/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10
```

*Note: the pathes of saved maps and outputs should be constructed before evaluating the DIOR-R testing set.*

#### Training & Evaluation-Segmentation

Training and evaluation the UperNet with ViT-B + RVSA backbone on Potsdam dataset:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=30000 tools/train.py \
configs/vit_base_win/upernet_vit_base_win_rvsa_v3_512x512_160k_potsdam_rgb_dpr10_lr6e5_lrd90_ps16_class5_ignore5.py \
--launcher 'pytorch' --cfg-options 'find_unused_parameters'=True
```

*Note: when training on the LoveDA, please add `--no-validate`*

##### ***When finetuning with more than 1 GPU for detection or segmentation, please use `nn.SyncBatchNorm` in the NormalCell of ViTAE models.***


