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

<p align="center">
  <a href="#updates">Updates</a> |
  <a href="#introduction">Introduction</a> |
  <a href="#results-and-models">Results & Models</a> |
  <a href="#usage">Usage</a> |
</p >

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
| MAE | ViT-B | 224 × 224 | 86| [Weights](https://1drv.ms/u/s!AimBgYV7JjTlgUNdvp0ulEweChb1?e=PUzFdU) |
| MAE | ViTAE-B | 224 × 224 | 89 | [Weights](https://1drv.ms/u/s!AimBgYV7JjTlgULC9Nz0-DVU-g3x?e=0uzlOs) |

### Object Detection

#### DOTA-V1.0 Single-Scale
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | --- | ------ | --- |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 78.75 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVIMtyrRiWH-t5YQ?e=BtQes4) |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 78.61 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVOp4xJwWLu6Vuey?e=au4xg7) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 78.96 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVDUBGDARsHTpBhW?e=pZ4hjy) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 78.99 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgVGWmS_5nylKxSkI?e=nl1EQ0) |


#### DOTA-V1.0 Multi-Scale
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | --- | ------ | --- |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 81.01 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgU5NKaWNgWSou8s1?e=kgucgJ) |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 80.51 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgU9TncBm6b9COF73?e=hUK7uk) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 81.16 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUwE_gb3u5TriDCd?e=XzInzG) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 80.97 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgU3ftuaaqkVmNWWb?e=BjvfdZ) |

#### DIOR-R
| Method | Pretrain | Backbone | Lr schd | mAP | Config | Log | Model |
| ------ |----------| -------- | --------- | ------- | --- | ------ | --- |
| Oriented R-CNN | MAE | ViT-B + RVSA | 1x | 70.67 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUhVXfiflUe0q020?e=VQsxOO) |
| Oriented R-CNN | MAE | ViT-B + RVSA $^ \Diamond$ | 1x | 70.85 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUvv90VwP88vuQ3V?e=6R1iI7) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA | 1x | 70.95 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUpNYT2Cn3xrgokO?e=OC61c2) |
| Oriented R-CNN | MAE | ViTAE-B + RVSA $^ \Diamond$ | 1x | 71.05 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgUlLPEW_fYo9Vdxp?e=cAQV4z) |

### Scene Classification

|Pretrain | Backbone | UCM-55 | AID-28 | AID-55 | NWPU-19 | NWPU-28 |
|----------|-------- | --------- | ------- | --- | ------ | --- | 
| MAE | ViT-B + RVSA | 99.70 | 96.92 | 98.33 | 93.79 | 95.49 |
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWwP9crvFEjxCAms?e=QMiskv) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXbTlEZBWwhG3x7E?e=c4fr2m) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXsz0bJG9ORHvCdJ?e=dFa9Bh) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgW9XRLsuNTxT2Uqn?e=VnYbEm) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXNlDhN46bvJeN_Q?e=sWuTLd) |
| MAE | ViT-B + RVSA $^ \Diamond$ | 99.58 | 96.86 | 98.44 | 93.74 | 95.45 |
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWnLKgu83zX08Kyq?e=YS55iD) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXgtdXL2tRuvxcV2?e=jO4tXz) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXz0haDLX0XO-Vom?e=cYnD2V) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXBSiVQmExNqHQLU?e=RIV3EA) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXSMsH12LRpafudZ?e=qYIOn3) |
| MAE | ViTAE-B + RVSA | 99.56 | 97.03 | 98.48 | 93.93 | 95.69 |
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWtInpfR-s3kPAlU?e=742yfX) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXWi5lX1mmCqBRja?e=UUjvmO) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXlt7LwJGVl6d3LX?e=75tDcx) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgW3Wn2KC2E78NF4m?e=4WHqWB) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXKfSfQ75M6N6rYv?e=V8RjAB) |
| MAE | ViTAE-B + RVSA $^ \Diamond$ | 99.50 | 97.01 | 98.50 | 93.92 | 95.66|
|     |              |[Model](https://1drv.ms/u/s!AimBgYV7JjTlgWrHaUI-7o71_ZuY?e=Fzh7qK) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXeMt9QBpzonQg28?e=9dCHpW) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXrUpum6DpMIuPGz?e=2fdo6T) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgW5VYQS7oCao3fNE?e=agatZi) | [Model](https://1drv.ms/u/s!AimBgYV7JjTlgXFaCj5vyTHBUXzP?e=ZgYZLj) |

### Semantic Segmentation

#### ISPRS Potsdam

| Method | Pretrain | Backbone | Crop size | Lr schd | OA | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 90.60 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggtgZIOaF6fVKvxq?e=bJ1dCL) |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 90.77 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggxCsqnFeUKfLNGX?e=DwntaA) |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 91.22 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggnb0VyF9Lf37-X7?e=8hYtSS) |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 91.15 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggo-6wH5O3P9CbNQ?e=AGKQCj) |

#### iSAID

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 896 × 896 | 160k | 63.76 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggEDf2JWqXWnS7XS?e=dJU1os) |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 63.85 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggSJm0qt7gLysKcA?e=JlqOM6) |
| UperNet| MAE | ViTAE-B + RVSA | 896 × 896 | 160k | 63.48 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggKoccMgQD6wCHRk?e=uDM4X3) |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 896 × 896 | 160k | 64.49 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggPs9ZRQxrv8yhWf?e=syTQB5) |

#### LoveDA

| Method | Pretrain | Backbone | Crop size | Lr schd | mIOU | Config | Log | Model |
| ------ | ----------|-------- | --------- | ------- | --- | ------ | --- | ----- |
| UperNet| MAE | ViT-B + RVSA | 512 × 512 | 160k | 51.95 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggeZZ5N7EbysmsaC?e=rbySyM) |
| UperNet| MAE | ViT-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 51.95 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggjO0_gnc2NWohH5?e=mP1wZn) |
| UperNet| MAE | ViTAE-B + RVSA | 512 × 512 | 160k | 52.26 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggVUiZz9skipmE4T?e=IhEuyT) |
| UperNet| MAE | ViTAE-B + RVSA $^ \Diamond$ | 512 × 512 | 160k | 52.44 | Coming Soon | Coming Soon | [Model](https://1drv.ms/u/s!AimBgYV7JjTlggYopXNjfWfa89Pz?e=RYxFkM) |

## Usage

Environment:

- Python 3.8.5
- Pytorch 1.9.0+cu111
- torchvision 0.10.0+cu111
- timm 0.4.12
- mmcv-full 1.4.1

### Pretraining & Finetuning-Classification

The codes are mainly borrowed from [MAE](https://github.com/facebookresearch/mae)

#### Pretraining (8 × A100 GPUs, 3~5 days)

1. Preparing the MillionAID: Download the [MillionAID](https://captain-whu.github.io/DiRS/). Here, we use previous `train_labels.txt` and `valid_labels.txt` of the [ViTAE-Transformer-Remote-Sensing](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing), which contain labels. However, since we conduct the **unsupervised pretraining**, the labels are not necessary. It is easy for users to record image names and revise corresponding codes `MAEPretrain_SceneClassification/util/datasets.py/class MillionAIDDataset`.

2. Pretraining: take ViT-B as an example (batchsize: 2048=8*256)

```
python -m torch.distributed.launch --nproc_per_node 8 --master_port 10000 main_pretrain.py \
--dataset 'millionAID' --model 'mae_vit_base_patch16' \
--batch_size 256 --epochs 1600 --warmup_epochs 40 \
--input_size 224 --mask_ratio 0.75 \
--blr 1.5e-4  --weight_decay 0.05 --gpu_num 8 \
--output_dir '../mae-main/output/'
```
*Note: Padding the convolutional kernel of PCM with `convertK1toK3.py` in the pretrained ViTAE-B for finetuning.*

3. Linear probe: an example of evaluating the pretrained ViT-B on UCM-55

```
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 10000 main_linprobe.py \
--dataset 'ucm' --model 'vit_base_patch16' \
--batch_size 256 --epochs 100 --warmup_epochs 10 \
--blr 1e-1  --weight_decay 0 --tag 0 \
--finetune '../mae-main/output/millionAID_224/1600_0.75_0.00015_0.05_2048/checkpoint-1599.pth'
```

#### Finetuning evaluation for pretraining & Finetuning-Classification

Finetuning ViTAE-B + RVSA on NWPU-28

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

For example, put `./Object Detection/mmdet/models/backbones/vit_win_rvsa_v3_wsz7.py` into `ViTAE-Transformer-Remote-Sensing/mmdet/models/backbones`

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
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/obb/oriented_rcnn/vit_base_win/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10.py \
../OBBDetection/work_dirs/faster/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10/latest.pth \
--out work_dirs/save/faster/full_det/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10/det_result.pkl --eval 'mAP' \
--show-dir work_dirs/save/faster/display/faster_rcnn_orpn_our_rsp_vitae-nc-base-win-rvsa_v3_wsz7_fpn_3x_dior_lr1e-4_ldr75_dpr10
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

### ***When finetuning with more than one GPU for detection or segmentation, please use `nn.SyncBatchNorm` in the NormalCell of ViTAE models.***


