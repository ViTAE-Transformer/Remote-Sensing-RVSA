# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import os

from pathlib import Path

from util.datasets import build_dataset

import paddle
import numpy as np
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default=None, type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/data/xiaolei.qin/Dataset/NWPU/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=45, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval',default=True, action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # other settings
    parser.add_argument("--dpr", default=0.1, type=float, help='drop path rate')

    parser.add_argument('--dataset', default=None, type=str, choices=['ucm','aid','nwpu','imagenet'], help='type of dataset')

    parser.add_argument("--split", default='28', type=int, help='trn-tes ratio')

    parser.add_argument("--tag", default='100', type=int, help='different idx (trn_num for millionaid, idx for others)')

    parser.add_argument("--exp_num", default=0, type=int, help='number of experiment times')

    parser.add_argument("--save_freq", default=10, type=int, help='number of saving frequency')

    parser.add_argument("--eval_freq", default=10, type=int, help='number of evaluation frequency')

    parser.add_argument("--postfix", default='dafault', type=str, help='postfix for save folder')

    parser.add_argument("--torch_path", default='/data/xiaolei.qin/Project/RVSA/Remote-Sensing-RVSA-main/MAEPretrain_SceneClassification_paddle/util/vit_rvsa_nwpu28.pth',
                        type=str, help='torch_path for save the original pth file')
    parser.add_argument("--paddle_path",
                        default='/data/xiaolei.qin/Project/RVSA/Remote-Sensing-RVSA-main/MAEPretrain_SceneClassification_paddle/util/vit_rvsa_nwpu28_paddle.pdparams',
                        type=str, help='torch_path for save the original pth file')
    return parser


def main(args):
    from vit_win_rvsa import ViT_Win_RVSA
    model = ViT_Win_RVSA(img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.dpr,
                         use_abs_pos_emb=True)

    if args.eval:
        paddle.set_device("gpu")
        

        ##把pth转成paddlepaddle权重
        def torch2paddle():
            import torch
            torch_path = args.torch_path
            paddle_path = args.paddle_path
            torch_state_dict = torch.load(torch_path)['model']

            if not os.path.exists(paddle_path):
                torch_state_dict = torch.load(torch_path)['model']
                # print('toch_state_dict',torch_state_dict.keys())
                fc_names = ["fc","mlp","head","proj","qkv"]
                paddle_state_dict = {}
                for k in torch_state_dict:
                    if "num_batches_tracked" in k:  # 飞桨中无此参数，无需保存
                        continue

                    v = torch_state_dict[k].detach().cpu().numpy()
                    flag = [i in k for i in fc_names]
                    if any(flag) and "weight" in k:
                        if 'patch_embed' not in k:
                            # new_shape = [1, 0] + list(range(2, v.ndim))
                            # print(
                            #     f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}"
                            # )
                            # v = v.transpose(new_shape)  # 转置 Linear 层的 weight 参数
                            print('v',k,v.shape)
                            v=v.transpose()
                            print('v1', v.shape)
                    # 将 torch.nn.BatchNorm2d 的参数名称改成 paddle.nn.BatchNorm2D 对应的参数名称
                    k = k.replace("running_var", "_variance")
                    k = k.replace("running_mean", "_mean")
                    # 添加到飞桨权重字典中
                    paddle_state_dict[k] = v
                paddle.save(paddle_state_dict, paddle_path)

        torch2paddle()

        final_checkpoint_dict = paddle.load(args.paddle_path)

        # 将load后的参数与模型关联起来
        model.set_state_dict(final_checkpoint_dict)

        model.eval()

        dataset_val = build_dataset(False, args)

        dataloader = paddle.io.DataLoader(dataset_val,batch_size=48)
        for iter, batch in enumerate(dataloader):

            print('iter',iter)
            images,target=batch
            output = model(images)
            if iter==0:
                output_concat=output.cpu().numpy()
                target_concat = target.cpu().numpy()
            else:
                output_concat=np.concatenate([output_concat,output.cpu().numpy()])
                target_concat = np.concatenate([target_concat, target.cpu().numpy()])

        acc1 = paddle.metric.accuracy(paddle.to_tensor(output_concat), paddle.to_tensor(np.expand_dims(target_concat,1)), k=1)
        print('Acc',acc1)
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
