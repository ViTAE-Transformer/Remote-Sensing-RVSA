import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser('MAE pre-training convert checkpoint', add_help=False)
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument('--out', default='', type=str)
parser.add_argument('--average', action='store_true')

args = parser.parse_args()

ckpt = torch.load('./output/mae_vitae_base_pretrn/millionAID_224/1600_0.75_0.00015_0.05_2048/checkpoint-1599.pth', map_location='cpu')['model']

args.out = './output/mae_vitae_base_pretrn/millionAID_224/1600_0.75_0.00015_0.05_2048/checkpoint-1599-transform-no-average.pth'

newCkpt = {}

for key, value in ckpt.items():
    if 'PCM' in key and 'weight' in key:
        print(key,value.shape)
        if len(value.shape) == 4 and value.shape[-1] == 1:
            value = F.pad(value, (1, 1, 1, 1, 0, 0, 0, 0))
            if args.average:
                value = torch.mean(value, [2, 3], keepdim=True).repeat(1, 1, 3, 3)
    if 'encoder.' in key:
        key = key[8:]
        newCkpt[key] = value
    else:
        newCkpt[key] = value

ckpt = {'model':newCkpt}
torch.save(ckpt, args.out)