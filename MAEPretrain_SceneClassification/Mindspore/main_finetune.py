from mindspore import nn
import argparse
from pathlib import Path
import mindspore as ms
from util.datasets import build_dataset
from mindspore.train import Top1CategoricalAccuracy
from mindspore.train import Accuracy


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
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
    parser.add_argument('--data_path',
                        default='./dataset/NWPU-RESISC45/', type=str,
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
    parser.add_argument('--eval', default=True, action='store_true',
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

    parser.add_argument('--dataset', default=None, type=str, choices=['ucm', 'aid', 'nwpu', 'imagenet'],
                        help='type of dataset')

    parser.add_argument("--split", default='28', type=int, help='trn-tes ratio')

    parser.add_argument("--tag", default='100', type=int, help='different idx (trn_num for millionaid, idx for others)')

    parser.add_argument("--exp_num", default=1, type=int, help='number of experiment times')

    parser.add_argument("--save_freq", default=10, type=int, help='number of saving frequency')

    parser.add_argument("--eval_freq", default=10, type=int, help='number of evaluation frequency')

    parser.add_argument("--postfix", default='sota', type=str, help='postfix for save folder')

    parser.add_argument("--torch_path",
                        default='./vit_rvsa_nwpu28.pth',
                        type=str, help='torch_path for save the original pth file')
    parser.add_argument("--mindspore_path",
                        default='./vit_rvsa_nwpu28_midspore.ckpt',
                        type=str, help='torch_path for save the original pth file')
    return parser


def main(args):
    from vit_win_rvsa import ViT_Win_RVSA
    import torch

    ms.set_context(device_target="GPU")

    model = ViT_Win_RVSA(img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.dpr,
                         use_abs_pos_emb=True)

    if args.eval:

        ##把pth转成mindspore权重
        from tqdm import tqdm
        import torch
        from mindspore import save_checkpoint, Tensor

        import mindspore
        import torch
        import pandas as pd
        import csv
        import numpy as np
        from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
        # mindspore模型

        def pytorch2mindspore(default_file='torch_resnet.pth'):
            # read pth file
            par_dict = torch.load(args.torch_path)['model']
            params_list = []
            for name in par_dict:
                param_dict = {}
                parameter = par_dict[name]
                param_dict['name'] = name
                param_dict['data'] = Tensor(parameter.cpu().numpy())
                if 'norm1.bias' in param_dict['name']:
                    param_dict['name'] = param_dict['name'].replace("norm1.bias", "norm1.beta")
                elif 'norm1.weight' in name:
                    param_dict['name'] = param_dict['name'].replace("norm1.weight", "norm1.gamma")
                elif 'norm2.bias' in param_dict['name']:
                    param_dict['name'] = param_dict['name'].replace("norm2.bias", "norm2.beta")
                elif 'norm2.weight' in param_dict['name']:
                    param_dict['name'] = param_dict['name'].replace("norm2.weight", "norm2.gamma")
                elif 'norm.bias' in param_dict['name']:
                    param_dict['name'] = param_dict['name'].replace("norm.bias", "norm.beta")
                elif 'norm.weight' in param_dict['name']:
                    param_dict['name'] = param_dict['name'].replace("norm.weight", "norm.gamma")
                params_list.append(param_dict)

            save_checkpoint(params_list, args.mindspore_path)

        pytorch2mindspore()

        final_checkpoint_dict = ms.load_checkpoint(args.mindspore_path)
        ms.load_param_into_net(model, final_checkpoint_dict)
        model.set_train(False)

        dataset_val = build_dataset(False, args)
        import numpy as np
        import mindspore.dataset as ds
        
        dataset = ds.GeneratorDataset(dataset_val, column_names=["data", "label"], num_parallel_workers=1, shuffle=False)
        dataloader = dataset.batch(args.batch_size, drop_remainder=False)
        iter = 0
        test_bar = tqdm(dataloader)
        for images, target in test_bar:
            output = model(images.squeeze(1))
            if iter == 0:
                output_concat = output.asnumpy()
                target_concat = target.asnumpy()
            else:
                output_concat = np.concatenate([output_concat, output.asnumpy()])
                target_concat = np.concatenate([target_concat, target.asnumpy()])
            iter += 1
        metric = Accuracy('classification')
        metric.clear()
        metric.update(output_concat, target_concat)
        accuracy = metric.eval()
        print(accuracy)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
