import math
from numpy.core.fromnumeric import resize, shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import PatchEmbed
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer

class PRM(nn.Module):
    def __init__(self, img_size=224, kernel_size=4, downsample_ratio=4, dilations=[1,6,12], in_chans=3, embed_dim=64, share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio

        if share_weights:
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                stride=self.stride, padding=3*dilations[0]//2, dilation=dilations[0])

        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = math.ceil(((self.kernel_size-1)*dilation + 1 - self.stride) / 2)
                self.convs.append(nn.Sequential(*[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                    stride=self.stride, padding=padding, dilation=dilation),
                    nn.GELU()]))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, W, H = x.shape
        if self.share_weights:
            padding = math.ceil(((self.kernel_size-1)*self.dilations[0] + 1 - self.stride) / 2)
            y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                stride=self.downsample_ratio, padding=padding, dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size-1)*self.dilations[i] + 1 - self.stride) / 2)
                _y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                    stride=self.downsample_ratio, padding=padding, dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        B, C, W, H, N = y.shape
        if self.op == 'sum':
            y = y.sum(dim=-1).flatten(2).permute(0,2,1).contiguous()
        elif self.op == 'cat':
            y = y.permute(0,4,1,2,3).flatten(3).reshape(B, N*C, W*H).permute(0,2,1).contiguous()
        else:
            raise NotImplementedError('no such operation: {} for multi-levels!'.format(self.op))
        return y

class ReductionCell(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dims=64, token_dims=64, downsample_ratios=4, kernel_size=7,
                 num_heads=1, dilations=[1,2,3,4], share_weights=False, op='cat', tokens_type='performer', group=1,
                 drop=0., attn_drop=0., drop_path=0., mlp_ratio=1.0):
        super().__init__()

        self.img_size = img_size
        self.op = op
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        self.outSize = img_size
        PCMStride = []
        residual = downsample_ratios // 2

        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2

        self.pool = None
        patch_size = downsample_ratios
        self.pos_embed = None
        self.cls_token = None

        if tokens_type == 'none':
            self.PRM = None
            self.outSize = self.outSize // downsample_ratios

        elif tokens_type == 'embedding':
            self.PRM = PatchEmbed(img_size, patch_size, in_chans, token_dims)
            num_patches = self.PRM.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, token_dims))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dims))
            self.outSize = self.outSize // downsample_ratios

        elif tokens_type == 'pooling':
            PCMStride = [1, 1, 1]
            self.pool = nn.MaxPool2d(downsample_ratios, stride=downsample_ratios, padding=0)
            tokens_type = 'transformer'
            self.outSize = self.outSize // downsample_ratios
            downsample_ratios = 1


        if tokens_type not in ['none', 'embedding']:
            self.PCM = nn.Sequential(
                            nn.Conv2d(in_chans, embed_dims, kernel_size=(3, 3), stride=PCMStride[0], padding=(1, 1), groups=group),  # the 1st convolution
                            nn.SiLU(inplace=True),
                            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), stride=PCMStride[1], padding=(1, 1), groups=group),  # the 1st convolution
                            nn.BatchNorm2d(embed_dims),
                            nn.SiLU(inplace=True),
                            nn.Conv2d(embed_dims, token_dims, kernel_size=(3, 3), stride=PCMStride[2], padding=(1, 1), groups=group),  # the 1st convolution
                            nn.SiLU(inplace=True))

            self.PRM = PRM(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios, dilations=self.dilations,
                in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op)
            self.outSize = self.outSize // downsample_ratios

            in_chans = self.PRM.out_chans

        if tokens_type == 'performer':
            assert num_heads == 1
            self.attn = Token_performer(dim=in_chans, in_dim=token_dims, head_cnt=num_heads, kernel_ratio=0.5)

        elif tokens_type == 'performer_less' or 'none' or 'embedding':
            self.attn = None
            self.PCM = None

        elif tokens_type == 'transformer':
            self.attn = Token_transformer(dim=in_chans, in_dim=token_dims, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path)

        self.num_patches = (img_size // downsample_ratios) * (img_size // downsample_ratios)  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        
        if self.PRM is None:
            return x

        if len(x.shape) < 4:
            B, N, C  = x.shape
            n = int(np.sqrt(N))
            x = x.view(B, n, n, C).contiguous()
            x = x.permute(0, 3, 1, 2)

        if self.pool is not None:
            x = self.pool(x)

        PRM_x = self.PRM(x)
        B = PRM_x.shape[0]
        if self.pos_embed is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            PRM_x = torch.cat((cls_tokens, PRM_x), dim=1)
            PRM_x = PRM_x + self.pos_embed

        if self.attn is None:
            return PRM_x

        convX = self.PCM(x)
        x = self.attn.attn(self.attn.norm1(PRM_x))
        convX = convX.permute(0, 2, 3, 1).view(*x.shape).contiguous()
        x = x + convX
        x = x + self.attn.drop_path(self.attn.mlp(self.attn.norm2(x)))

        return x