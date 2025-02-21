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
# import datetime
# import json
# import numpy as np
# import os
# import time
from pathlib import Path

import jittor as jt
from tqdm import tqdm

# import util.lr_decay as lrd
# # import util.misc as misc
# from MAEPretrain_SceneClassification.loss import trunc_normal_, SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
# from MAEPretrain_SceneClassification.mixup import Mixup
from util.datasets import build_dataset
from vit_win_rvsa import ViT_Win_RVSA

# from util.pos_embed import interpolate_pos_embed
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
# import jittor.optim as optim
# import jittor.dataset as dataset
# import os
# import numpy as np
# import time
# import datetime
# import json
# import logging
# from jittor import transform
# from jittor import optim

# import models_vit
#
# from engine_finetune import train_one_epoch, evaluate
#
# import jittor as jt
jt.flags.use_cuda = 0

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
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

    parser.add_argument("--exp_num", default=0, type=int, help='number of experiment times')

    parser.add_argument("--save_freq", default=10, type=int, help='number of saving frequency')

    parser.add_argument("--eval_freq", default=10, type=int, help='number of evaluation frequency')

    parser.add_argument("--postfix", default='dafault', type=str, help='postfix for save folder')

    return parser

class MetricLogger:
    def __init__(self):
        self.meters = {'acc1': AverageMeter()}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, jt.Var):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def synchronize_between_processes(self):
        # 如果使用多卡训练，需要在此同步各进程的结果
        # Jittor 提供了 mpi_all_reduce 等函数用于多进程通信
        # 这里只是示例，具体实现需要根据实际情况调整
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __str__(self):
        return ' '.join([f'{name}: {meter}' for name, meter in self.meters.items()])

class AverageMeter:
    """计算并存储指标的平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize_between_processes(self):
        # 如果使用多卡训练，需要在此同步各进程的结果
        # 这里只是示例，具体实现需要根据实际情况调整
        pass

    def __str__(self):
        return f'{self.avg:.4f}'

# def accuracy(output, target, topk=(1,)):
#     """计算指定 top k 的准确率"""
#     print('output', output.shape)
#     print('target', target.shape)
#
#     maxk = max(topk)
#     batch_size = target.shape[0]
#
#     # 使用 jt.topk 获取前 maxk 个预测索引（假设 jt.topk 返回 (values, indices)）
#     _, pred = jt.topk(output, k=maxk, dim=1)
#
#     # pred 形状为 [batch_size, maxk]，转置后变为 [maxk, batch_size]
#     pred = jt.transpose(pred, (1, 0))
#
#     # 将 target 调整为形状 [1, batch_size] 并扩展为与 pred 相同的形状
#     expanded_target = target.reshape((1, -1)).expand(pred.shape)
#
#     # 计算预测是否正确（注意：Jittor Var 支持 == 运算）
#     correct = (pred == expanded_target).float()
#
#     res = []
#     for k in topk:
#         # 对前 k 个预测结果，展平后求和，得到正确预测的数量
#         correct_k = correct[:k].reshape([-1]).sum()
#         # 计算准确率百分比
#         res.append(correct_k * (100.0 / batch_size))
#     print('res', res)
#
#     return res

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.equal(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

# 示例用法

def main(args):
    # Initialize distributed training mode
    # misc.init_distributed_mode(args)
    #
    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))
    #
    # device = jt.cuda if args.device == 'cuda' else jt.cpu
    #
    # path, _  = os.path.split(args.finetune)
    #
    # args.output_dir = os.path.join(path, str(args.model)+'_fintune'+'_'+str(args.dataset)+'_'+str(args.split) + '_'+str(args.input_size)) + '_' + str(args.postfix)
    # os.makedirs(args.output_dir, exist_ok=True)
    #
    # exp_record = np.zeros([3,args.exp_num + 2])
    #
    # open(os.path.join(args.output_dir, "log.txt"), mode="w", encoding="utf-8")
    #
    # for i in range(args.exp_num):
    #
    #     with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write('############# Experiment {} #############'.format(i) + "\n")
    #
    #     # fix the seed for reproducibility
    #     seed = i + misc.get_rank()
    #     jt.set_seed(seed)
    #     np.random.seed(seed)
    #
    #     dataset_train = build_dataset(is_train=True, args=args)
    #     dataset_val = build_dataset(is_train=False, args=args)
    #
    #     print(dataset_train)
    #     print(dataset_val)
    #
    #     if args.distributed:
    #         num_tasks = misc.get_world_size()
    #         global_rank = misc.get_rank()
    #         sampler_train = dataset.DistributedSampler(
    #             dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #         )
    #         print("Sampler_train = %s" % str(sampler_train))
    #         if args.dist_eval:
    #             if len(dataset_val) % num_tasks != 0:
    #                 print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
    #                     'This will slightly alter validation results as extra duplicate entries are added to achieve '
    #                     'equal num of samples per-process.')
    #             sampler_val = dataset.DistributedSampler(
    #                 dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    #         else:
    #             sampler_val = dataset.SequentialSampler(dataset_val)
    #     else:
    #         sampler_train = dataset.RandomSampler(dataset_train)
    #         sampler_val = dataset.SequentialSampler(dataset_val)
    #
    #
    #     log_writer = None
    #
    #     batch_size = int(args.batch_size / args.world_size)  # Split batch size across GPUs
    #     num_workers = int(args.num_workers / args.world_size)
    #
    #     data_loader_train = dataset.DataLoader(
    #         dataset_train, sampler=sampler_train,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=True,
    #     )
    #
    #     data_loader_val = dataset.DataLoader(
    #         dataset_val, sampler=sampler_val,
    #         batch_size=batch_size,
    #         num_workers=num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )
    #
    #     mixup_fn = None
    #     mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    #     if mixup_active:
    #         print("Mixup is activated!")
    #         mixup_fn = Mixup(
    #             mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
    #             prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
    #             label_smoothing=args.smoothing, num_classes=args.nb_classes)
    #
    #     # Model initialization based on args.model
    #     if args.model == 'vit_base_win_rvsa':
    #         from vit_win_rvsa import ViT_Win_RVSA
    #         model = ViT_Win_RVSA(img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.dpr,
    #                              use_abs_pos_emb=True)
    #     elif args.model == 'vit_base_win_rvsa_kvdiff':
    #         from vit_win_rvsa_kvdiff import ViT_Win_RVSA_KVDIFF
    #         model = ViT_Win_RVSA_KVDIFF(img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.dpr,
    #                                     use_abs_pos_emb=True)
    #     elif args.model == 'vitae_nc_base_win_rvsa':
    #         from vitae_nc_win_rvsa import ViTAE_NC_Win_RVSA
    #         model = ViTAE_NC_Win_RVSA(img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.dpr,
    #                                   use_abs_pos_emb=True)
    #     elif args.model == 'vitae_nc_base_win_rvsa_kvdiff':
    #         from vitae_nc_win_rvsa_kvdiff import ViTAE_NC_Win_RVSA_KVDIFF
    #         model = ViTAE_NC_Win_RVSA_KVDIFF(img_size=args.input_size, num_classes=args.nb_classes,
    #                                          drop_path_rate=args.dpr, use_abs_pos_emb=True)
    #
    #     if args.finetune and not args.eval:
    #         checkpoint = jt.load(args.finetune)
    #
    #         print("Load pre-trained checkpoint from: %s" % args.finetune)
    #         checkpoint_model = checkpoint['model']
    #         state_dict = model.state_dict()
    #         for k in ['head.weight', 'head.bias']:
    #             if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
    #                 print(f"Removing key {k} from pretrained checkpoint")
    #                 del checkpoint_model[k]
    #
    #         interpolate_pos_embed(model, checkpoint_model)
    #
    #         msg = model.load_state_dict(checkpoint_model, strict=False)
    #         print(f"Loaded model state dict.", msg)
    #
    #         if args.model == 'vit_base_patch16':
    #
    #             if args.global_pool:
    #                 assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    #             else:
    #                 assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    #
    #                 trunc_normal_(model.head.weight, std=2e-5)
    #
    #
    #     model = model.to(device)
    #
    #     model_without_ddp = model
    #     n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    #     print("Model = %s" % str(model_without_ddp))
    #     print('number of params (M): %.2f' % (n_parameters / 1.e6))
    #
    #
    #
    #     eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    #
    #     if args.lr is None:  # only base_lr is specified
    #         args.lr = args.blr * eff_batch_size / 256
    #
    #     print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    #     print("actual lr: %.2e" % args.lr)
    #
    #     print("accumulate grad iterations: %d" % args.accum_iter)
    #     print("effective batch size: %d" % eff_batch_size)
    #
    #
    #     # Optimizer setup
    #     param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #         no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #         layer_decay=args.layer_decay
    #     )
    #     optimizer = optim.AdamW(param_groups, lr=args.lr)
    #     loss_scaler = NativeScaler()
    #
    #     # Loss function
    #     if mixup_fn is not None:
    #         criterion = SoftTargetCrossEntropy()
    #     elif args.smoothing > 0.:
    #         criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    #     else:
    #         criterion = nn.CrossEntropyLoss()
    #
    #     print("criterion = %s" % str(criterion))
    #
    #     misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    #
    #     if args.eval:
    #         test_stats = evaluate(data_loader_val, model, device)
    #         print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #         exit(0)
    #
    #     # Training loop
    #     print(f"Start training for {args.epochs} epochs")
    #     start_time = time.time()
    #     max_accuracy = 0.0
    #     best_acc1 = 0
    #     for epoch in range(args.start_epoch, args.epochs):
    #         if args.distributed:
    #             data_loader_train.sampler.set_epoch(epoch)
    #         train_stats = train_one_epoch(
    #             model, criterion, data_loader_train,
    #             optimizer, device, epoch, loss_scaler,
    #             args.clip_grad, mixup_fn,
    #             log_writer=log_writer,
    #             args=args
    #         )
    #         if epoch % args.eval_freq == 0:
    #             test_stats = evaluate(data_loader_val, model, device)
    #             print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    #             max_accuracy = max(max_accuracy, test_stats["acc1"])
    #             print(f'Max accuracy: {max_accuracy:.2f}%')
    #
    #     # Save results
    #         if test_stats["acc1"] > best_acc1:
    #             save_state = {'model': model.module.state_dict(),
    #                           'optimizer': optimizer.state_dict(),
    #                           'max_accuracy': max_accuracy,
    #                           'epoch': epoch,
    #                           'args': args}
    #
    #             save_path = os.path.join(args.output_dir, 'best_ckpt.pth')
    #             print(f"{save_path} saving......")
    #             jt.save(save_state, save_path)
    #             print(f"{save_path} saved !!!")
    #             best_acc1 = test_stats["acc1"]
    #
    #         if log_writer is not None:
    #             log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
    #             log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
    #             log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
    #
    #         log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                      **{f'test_{k}': v for k, v in test_stats.items()},
    #                      'epoch': epoch,
    #                      'n_parameters': n_parameters}
    #
    #         if args.output_dir and misc.is_main_process():
    #             if log_writer is not None:
    #                 log_writer.flush()
    #             with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #                 f.write(json.dumps(log_stats) + "\n")
    #
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     print('Onetime training time {}'.format(total_time_str))
    #
    #     exp_record[0, i] = test_stats['acc1']
    #     exp_record[1, i] = max_accuracy
    #     exp_record[2, i] = int(total_time)
    #
    # # Save logs and results
    # exp_record[0,-2] = np.mean(exp_record[0,:args.exp_num])
    # exp_record[0,-1] = np.std(exp_record[0,:args.exp_num])
    # exp_record[1,-2] = np.mean(exp_record[1,:args.exp_num])
    # exp_record[1,-1] = np.std(exp_record[1,:args.exp_num])
    # exp_record[2,-2] = np.mean(exp_record[2,:args.exp_num])
    # exp_record[2,-1] = np.std(exp_record[2,:args.exp_num])
    #
    # with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #     f.write('Last acc1 of {} model on {} dataset: {:.2f} ± {:.2f} \n'.format(args.model, args.dataset, exp_record[0,-2], exp_record[0,-1]))
    #     f.write('Max acc1 of {} model on {} dataset: {:.2f} ± {:.2f} \n'.format(args.model, args.dataset, exp_record[1,-2], exp_record[1,-1]))
    #     f.write('Average training time on {} epoch: {} ± {} \n'.format(args.epochs, str(datetime.timedelta(seconds=int(exp_record[2,-2]))), \
    #                                                                                         str(datetime.timedelta(seconds=int(exp_record[2,-1])))))
    # print(exp_record)
    # print('Last acc1 of {} model on {} dataset: {:.2f} ± {:.2f}'.format(args.model, args.dataset, exp_record[0,-2], exp_record[0,-1]))
    # print('Max acc1 of {} model on {} dataset: {:.2f} ± {:.2f}'.format(args.model, args.dataset, exp_record[1,-2], exp_record[1,-1]))
    # print('Average training time on {} epoch: {} ± {}'.format(args.epochs, str(datetime.timedelta(seconds=int(exp_record[2,-2]))), \
    #                                                                                     str(datetime.timedelta(seconds=int(exp_record[2,-1])))))
    # print('number of params (M): %.2f' % (n_parameters / 1.e6))
    #

    # 加载模型
    model = ViT_Win_RVSA(img_size=args.input_size, num_classes=args.nb_classes, drop_path_rate=args.dpr,
                         use_abs_pos_emb=True)

    if args.eval:
        jt.flags.use_cuda = 1  # 设置使用 GPU

        # metric_logger = misc.MetricLogger(delimiter="  "

        # 加载权重
        #final_checkpoint_dict = jt.load("/root/autodl-tmp/lzm/Remote-Sensing-RVSA-jittor/MAEPretrain_SceneClassification/vit_rvsa_nwpu28_jittor.pth")


        final_checkpoint_dict = jt.load("/root/autodl-tmp/lzm/data/vit_rvsa_nwpu28.pth")['model']
        model.load_state_dict(final_checkpoint_dict)
        metric_logger = MetricLogger()
        # 设置为评估模式
        model.eval()

        # 数据集与数据加载
        dataset_val = build_dataset(False, args)
        dataloader = jt.dataset.DataLoader(dataset_val, batch_size=48, shuffle=False)

        # print('len(dataloader)', len(dataloader))
        for iter, batch in tqdm(enumerate(dataloader)):
            # print('iter', iter)
            images, target = batch
            # print('images', images.shape)
            # print('target', target.shape)
            output = model(images)

            # 计算准确率
            # print('output', output)
            # print('target', target)

            acc1 = accuracy(output, target)[0]
            # print("Top-1 Accuracy:", acc1)
            batch_size = images.shape[0]
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        # 同步结果
        metric_logger.synchronize_between_processes()
        print("Top-1 Accuracy:", metric_logger.meters['acc1'].avg)

        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)



#
'''def convert_torch_to_jittor(torch_ckpt_path, jittor_model, transpose_linear=True):
    """
    将 PyTorch 保存的模型权重转换为 Jittor 模型可加载的格式，并将权重加载到 jittor_model 中。

    参数：
      torch_ckpt_path: PyTorch 模型 checkpoint 路径
      jittor_model: Jittor 模型实例，必须实现 set_state_dict 方法
      transpose_linear: 是否对二维参数（全连接层权重）进行转置（默认为 True）

    备注：
      1. 该函数需要 PyTorch 支持（用于 torch.load）。
      2. 如果 checkpoint 中包含 "model" 键，则默认使用其值作为状态字典。
      3. 对于 BatchNorm 层，将 "running_mean" 替换为 "_mean"，"running_var" 替换为 "_variance"。
    """
    import torch
    # 加载 PyTorch checkpoint，确保在 CPU 上加载
    torch_state = torch.load(torch_ckpt_path, map_location="cpu")
    if "model" in torch_state:
        torch_state = torch_state["model"]

    jittor_state = {}
    for k, v in torch_state.items():
        # 跳过 PyTorch 中不需要的参数
        if "num_batches_tracked" in k:
            continue

        # 将 tensor 转为 numpy 数组
        np_v = v.detach().cpu().numpy()

        # 替换 BatchNorm 参数名称（如有需要）
        k = k.replace("running_mean", "_mean").replace("running_var", "_variance")

        # 如果是二维参数且需要转置，则转置全连接层的权重
        if transpose_linear and np_v.ndim == 2:
            np_v = np_v.T

        jittor_state[k] = np_v

    # 加载转换后的状态字典到 Jittor 模型中
    jittor_model.set_state_dict(jittor_state)
    print("转换成功，已将 PyTorch 权重加载到 Jittor 模型中。")'''
