# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'
import math
# import torch
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial


# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint

from einops import rearrange, repeat
from timm.models.layers import drop_path, to_2tuple, trunc_normal_


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(paddle.nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=qkv_bias)
        self.window_size = window_size
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1
        zeros_par=paddle.zeros([rel_sp_dim, head_dim])
        self.rel_pos_h = paddle.create_parameter(zeros_par.shape,zeros_par.numpy().dtype,default_initializer=paddle.nn.initializer.Assign(zeros_par)) # 2ws-1,C'
        self.rel_pos_w = paddle.create_parameter(zeros_par.shape,zeros_par.numpy().dtype,default_initializer=paddle.nn.initializer.Assign(zeros_par)) # 2ws-1,C'

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        qkv = paddle.transpose(qkv.reshape([B, N, 3, self.num_heads, -1]),[2, 0, 3, 1, 4]) # 3，B，H，N，C
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) # B，H，N，C

        q = q * self.scale
        attn = (q @ paddle.transpose(k,[0,1,3, 2])) # B,H,N,N

        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        sfm=paddle.nn.Softmax(-1)
        attn = sfm(attn)
        attn = self.attn_drop(attn)

        x = paddle.transpose((attn @ v),[0,2,1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view([B, H // window_size, window_size, W // window_size, window_size, C])
    windows = paddle.transpose(x,[0, 1, 3, 2, 4, 5]).contiguous().view([-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        paddle.arange(q_h)[:, None] * q_h_ratio - paddle.arange(k_h)[None, :] * k_h_ratio
    )

     # dist_h [1-ws,ws-1]->[0,2ws-2]

    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        paddle.arange(q_w)[:, None] * q_w_ratio - paddle.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # get pos encode, qwh, kwh, C'

    Rh = rel_pos_h[dist_h.astype('int64')]
    Rw = rel_pos_w[dist_w.astype('int64')]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape([B, n_head, q_h, q_w, dim]) # B, H, qwh, qww, C
    rel_h = paddle.einsum("byhwc,hkc->byhwk", r_q, Rh)# B, H, qwh, qww, C'; qwh, kWh, C' -> B,H,qwh,qww,kwh
    rel_w = paddle.einsum("byhwc,wkc->byhwk", r_q, Rw)# B,H,qwh,qww,kww

    # attn: B,H,qwh,qww,kwh,kww

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view([B, -1, q_h, q_w, k_h, k_w])
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view([B, -1, q_h * q_w, k_h * k_w])

    return attn

class RotatedVariedSizeWindowAttention(paddle.nn.Layer):
    def __init__(self, dim, num_heads, out_dim=None, window_size=1, qkv_bias=True, qk_scale=None, 
            attn_drop=0., proj_drop=0, attn_head_dim=None, relative_pos_embedding=True, learnable=True, restart_regression=True,
            attn_window_size=None, shift_size=0, img_size=(1,1), num_deform=None):
        super().__init__()
        
        window_size = window_size[0]
        
        self.img_size = to_2tuple(img_size)
        self.num_heads = num_heads
        self.dim = dim
        out_dim = out_dim or dim
        self.out_dim = out_dim
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        attn_window_size = attn_window_size or window_size
        self.attn_ws = attn_window_size or self.ws

        q_size = window_size
        rel_sp_dim = 2 * q_size - 1
        h_zeros=paddle.zeros([rel_sp_dim, head_dim])
        w_zeros=paddle.zeros([rel_sp_dim, head_dim])
        self.rel_pos_h = paddle.create_parameter(h_zeros.shape,dtype=h_zeros.numpy().dtype,default_initializer=paddle.nn.initializer.Assign(h_zeros))
        self.rel_pos_w = paddle.create_parameter(w_zeros.shape,dtype=h_zeros.numpy().dtype,default_initializer=paddle.nn.initializer.Assign(w_zeros))
        
        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            # if num_deform is None, we set num_deform to num_heads as default

            if num_deform is None:
                num_deform = 1
            self.num_deform = num_deform

            self.sampling_offsets = nn.Sequential(
                nn.AvgPool2D(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(), 
                nn.Conv2D(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AvgPool2D(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(), 
                nn.Conv2D(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            # add angle
            self.sampling_angles = nn.Sequential(
                nn.AvgPool2D(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(), 
                nn.Conv2D(dim, self.num_heads * self.num_deform * 1, kernel_size=1, stride=1)
            )

        self.shift_size = shift_size % self.ws
        # self.left_size = self.img_size
#        if min(self.img_size) <= self.ws:
#            self.shift_size = 0

        # if self.shift_size > 0:
        #     self.padding_bottom = (self.ws - self.shift_size + self.padding_bottom) % self.ws
        #     self.padding_right = (self.ws - self.shift_size + self.padding_right) % self.ws

        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, out_dim * 3, bias_attr=qkv_bias)

        #self.qkv = nn.Conv2D(dim, out_dim * 3, 1, bias=qkv_bias)
        # self.kv = nn.Conv2D(dim, dim*2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        #self.proj = nn.Conv2D(out_dim, out_dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            zeros_par=paddle.zeros([(window_size + attn_window_size - 1) * (window_size + attn_window_size - 1), num_heads])
            self.relative_position_bias_table = paddle.create_parameter(
                zeros_par.shape,dtype=zeros_par.numpy().dtype, default_initializer=paddle.nn.initializer.Assign(zeros_par))  # 2*Wh-1 * 2*Ww-1, nH


            # get pair-wise relative position index for each token inside the window
            coords_h = paddle.arange(self.attn_ws)
            coords_w = paddle.arange(self.attn_ws)
            coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose([1, 2, 0]).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.attn_ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.attn_ws - 1
            relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.02)
            paddle.nn.initializer.TruncatedNormal(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x, H, W):
        
        B,N,C = x.shape
        assert N == H * W
        # x = x.view(B, H, W, C)
        x=paddle.view(x,[B,H,W,C])
        # x = x.permute(0, 3, 1, 2).contiguous()
        x=paddle.transpose(x,[0,3,1,2]).contiguous()
        b, _, h, w = x.shape
        shortcut = x
        # assert h == self.img_size[0]
        # assert w == self.img_size[1]
        # if self.shift_size > 0:
        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left
        
        # padding on left-right-up-down
        expand_h, expand_w = h+padding_top+padding_down, w+padding_left+padding_right
        
        # window num in padding features
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws
        
        image_reference_h = paddle.linspace(-1, 1, expand_h).to(paddle.device.get_device())#.to(x.device)
        image_reference_w = paddle.linspace(-1, 1, expand_w).to(paddle.device.get_device())#.to(x.device)
        image_reference = paddle.transpose(paddle.stack(paddle.meshgrid(image_reference_w, image_reference_h), 0),[0, 2, 1]).unsqueeze(0) # 1, 2, H, W
        
        # position of the window relative to the image center
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws) # 1,2, nh, nw
        # image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)# 1, 2, nh, ws, nw, ws
        image_reference = paddle.reshape(image_reference,[1, 2, window_num_h, self.ws, window_num_w, self.ws])
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]
        
        window_reference = paddle.reshape(window_reference,[1, 2, window_num_h, 1, window_num_w, 1])# 1,2, nh,1, nw,1
        
        # coords of pixels in each window

        base_coords_h = paddle.arange(self.attn_ws).to(paddle.device.get_device()) * 2 * self.ws / self.attn_ws / (expand_h-1) # ws
        base_coords_h = (base_coords_h - base_coords_h.mean())

        base_coords_w = paddle.arange(self.attn_ws).to(paddle.device.get_device()) * 2 * self.ws / self.attn_ws / (expand_w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())
        # base_coords = torch.stack(torch.meshgrid(base_coords_w, base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, 1, self.attn_ws, 1, self.attn_ws)
        
        # extend to each window
        # expanded_base_coords_h = base_coords_h.unsqueeze(0).repeat(window_num_h, 1) # ws -> 1,ws -> nh,ws
        expanded_base_coords_h=paddle.repeat_interleave(base_coords_h.unsqueeze(0),window_num_h,0)

        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.attn_ws
        # expanded_base_coords_w = base_coords_w.unsqueeze(0).repeat(window_num_w, 1) # nw,ws
        expanded_base_coords_w =paddle.repeat_interleave(base_coords_w.unsqueeze(0),window_num_w,0)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.attn_ws
        expanded_base_coords_h = expanded_base_coords_h.reshape([window_num_h*self.attn_ws]) # nh*ws
        expanded_base_coords_w = expanded_base_coords_w.reshape([window_num_w*self.attn_ws]) # nw*ws

        window_coords = paddle.transpose(paddle.stack(paddle.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0),[0, 2, 1]).reshape([1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws]) # 1, 2, nh, ws, nw, ws
        # base_coords = window_reference+window_coords
        base_coords = image_reference
        
        # padding feature
        x = paddle.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))
        
        if self.restart_regression:
            # compute for each head in each batch
            coords = paddle.repeat_interleave(base_coords,b*self.num_heads, 0) # B*nH, 2, nh, ws, nw, ws
        if self.learnable:
            # offset factors
            sampling_offsets = self.sampling_offsets(x)

            num_predict_total = b * self.num_heads * self.num_deform

            sampling_offsets = sampling_offsets.reshape([num_predict_total, 2, window_num_h, window_num_w])
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (h // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (w // self.ws)
                       
            # scale fators
            sampling_scales = self.sampling_scales(x)       #B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape([num_predict_total, 2, window_num_h, window_num_w])

            # rotate factor
            sampling_angle = self.sampling_angles(x)
            sampling_angle = sampling_angle.reshape([num_predict_total, 1, window_num_h, window_num_w])

            # first scale
            
            window_coords = window_coords * (sampling_scales[:, :, :, None, :, None] + 1)

            # then rotate around window center

            window_coords_r = window_coords.clone()

            # 0:x,column, 1:y,row

            window_coords_r[:,0,:,:,:,:] = -window_coords[:,1,:,:,:,:]*paddle.sin(sampling_angle[:,0,:,None,:,None]) + window_coords[:,0,:,:,:,:]*paddle.cos(sampling_angle[:,0,:,None,:,None])
            window_coords_r[:,1,:,:,:,:] = window_coords[:,1,:,:,:,:]*paddle.cos(sampling_angle[:,0,:,None,:,None]) + window_coords[:,0,:,:,:,:]*paddle.sin(sampling_angle[:,0,:,None,:,None])
            
            # system transformation: window center -> image center
            
            coords = window_reference + window_coords_r + sampling_offsets[:, :, :, None, :, None]

        # final offset
        sample_coords = paddle.transpose(coords,[0, 2, 3, 4, 5, 1]).reshape([num_predict_total, self.attn_ws*window_num_h, self.attn_ws*window_num_w, 2])
        # qkv = self.qkv(shortcut.permute(0, 2, 3, 1).reshape(b, -1, self.dim)).permute(0, 2, 1).reshape(b, -1, h,
        #                                                                                                w).reshape(b, 3,
        #                                                                                                           self.num_heads,
        #                                                                                                           self.out_dim // self.num_heads,
        #                                                                                                           h,
        #                                                                                                           w).transpose(
        #     1, 0).reshape(3 * b * self.num_heads, self.out_dim // self.num_heads, h, w)
        qkv = paddle.transpose(paddle.transpose(self.qkv(paddle.transpose(shortcut,[0,2,3,1]).reshape([b,-1,self.dim])),[0,2,1]).reshape([b,-1,h,w]).reshape([b, 3, self.num_heads, self.out_dim // self.num_heads, h, w]),[1, 0]).reshape([3*b*self.num_heads, self.out_dim // self.num_heads, h, w])
        # if self.shift_size > 0:

        qkv = paddle.nn.functional.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape([3, b*self.num_heads, self.out_dim // self.num_heads, h+padding_td, w+padding_lr])
        # else:
        #     qkv = qkv.reshape(3, b*self.num_heads, self.dim // self.num_heads, h, w)
        q, k, v = qkv[0], qkv[1], qkv[2] # b*self.num_heads, self.out_dim // self.num_heads, H，W
        
        k_selected = F.grid_sample(
                        k.reshape([num_predict_total, self.out_dim // self.num_heads // self.num_deform, h+padding_td, w+padding_lr]),
                        grid=sample_coords, padding_mode='zeros', align_corners=True
                        ).reshape([b*self.num_heads, self.out_dim // self.num_heads, h+padding_td, w+padding_lr])
        v_selected = F.grid_sample(
                        v.reshape([num_predict_total, self.out_dim // self.num_heads // self.num_deform, h+padding_td, w+padding_lr]),
                        grid=sample_coords, padding_mode='zeros', align_corners=True
                        ).reshape([b*self.num_heads, self.out_dim // self.num_heads, h+padding_td, w+padding_lr])

        q = paddle.transpose(q.reshape([b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.ws, window_num_w, self.ws]),[0, 3, 5, 1, 4, 6, 2]).reshape([b*window_num_h*window_num_w, self.num_heads, self.ws*self.ws, self.out_dim//self.num_heads])
        k = paddle.transpose(k_selected.reshape([b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws]),[0, 3, 5, 1, 4, 6, 2]).reshape([b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.out_dim//self.num_heads])
        v = paddle.transpose(v_selected.reshape([b, self.num_heads, self.out_dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws]),[0, 3, 5, 1, 4, 6, 2]).reshape([b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.out_dim//self.num_heads])

        dots = (q @ paddle.transpose(k,[0,1,3, 2])) * self.scale
        
        dots = calc_rel_pos_spatial(dots, q, (self.ws, self.ws), (self.attn_ws, self.attn_ws), self.rel_pos_h, self.rel_pos_w)
        
        if self.relative_pos_embedding:
            viewshape=self.relative_position_index.shape[0]*self.relative_position_index.shape[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view([viewshape])].view([
            self.ws * self.ws, self.attn_ws * self.attn_ws, -1])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = paddle.transpose(relative_position_bias,[2, 0, 1]).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        sfm = paddle.nn.Softmax(-1)
        attn = sfm(dots)
        out = attn @ v

        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b, hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        # if self.shift_size > 0:
            # out = torch.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out = out[:, :, padding_top:h+padding_top, padding_left:w+padding_left]
        
        out = paddle.transpose(out,[0,2,3,1]).reshape([B, H*W, -1])
 
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
    
    def _clip_grad(self, grad_norm):
        # print('clip grads of the model for selection')
        nn.utils.clip_grad_norm_(self.sampling_offsets.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(self.sampling_scales.parameters(), grad_norm)

    def _reset_parameters(self):
        if self.learnable:
            # nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            # nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
            # nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            # nn.init.constant_(self.sampling_scales[-1].bias, 0.)

            init_constant = paddle.nn.initializer.Constant(value=0.)
            init_constant(self.sampling_offsets[-1].weight)
            init_constant(self.sampling_offsets[-1].bias)
            init_constant(self.sampling_scales[-1].weight)
            init_constant(self.sampling_scales[-1].bias)

    def flops(self, ):
        N = self.ws * self.ws
        M = self.attn_ws * self.attn_ws
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * M
        #  x = (attn @ v)
        flops += self.num_heads * N * M * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[1] + self.shift_size + self.padding_right
        flops *= (h / self.ws * w / self.ws)

        # for sampling
        flops_sampling = 0
        if self.learnable:
            # pooling
            flops_sampling += h * w * self.dim
            # regressing the shift and scale
            flops_sampling += 2 * (h/self.ws + w/self.ws) * self.num_heads*2 * self.dim
            # calculating the coords
            flops_sampling += h/self.ws * self.attn_ws * w/self.ws * self.attn_ws * 2
        # grid sampling attended features
        flops_sampling += h/self.ws * self.attn_ws * w/self.ws * self.attn_ws * self.dim
        
        flops += flops_sampling

        return flops

class Block(paddle.nn.Layer):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, window=False, restart_regression=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        else:
            self.attn = RotatedVariedSizeWindowAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, 
            restart_regression=restart_regression)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = paddle.create_parameter(init_values * paddle.ones((dim)),requires_grad=True)
            self.gamma_2 = paddle.create_parameter(init_values * paddle.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(paddle.nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2)
        x=paddle.transpose(x,[0,2,1])

        return x, (Hp, Wp)

class HybridEmbed(paddle.nn.Layer):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, paddle.nn.Layer)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(paddle.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Norm2d(paddle.nn.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)#, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

#@BACKBONES.register_module()
class ViT_Win_RVSA(paddle.nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm)#, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        #self.out_indices = out_indices

        if use_abs_pos_emb:
            zeros_par = paddle.zeros([1, num_patches + 1, embed_dim])
            self.pos_embed = paddle.create_parameter(zeros_par.shape,dtype=zeros_par.numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(zeros_par))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # self.use_rel_pos_bias = use_rel_pos_bias
        # self.use_checkpoint = use_checkpoint

        # MHSA after interval layers
        # WMHSA in other layers
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=(7, 7) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape, window=((i + 1) % interval != 0), 
                restart_regression=restart_regression)
            for i in range(depth)])
         
        #self.interval = interval

        if self.pos_embed is not None:
            paddle.nn.initializer.TruncatedNormal(self.pos_embed, std=.02)

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # manually initialize fc layer
        paddle.nn.initializer.TruncatedNormal(self.head.weight, std=2e-5)

        # self.fpn1 = nn.Sequential(
        #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        #     Norm2d(embed_dim),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        # )

        # self.fpn2 = nn.Sequential(
        #     nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        # )

        # self.fpn3 = nn.Identity()

        # self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            paddle.nn.initializer.TruncatedNormal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # nn.init.constant_(m.bias, 0)

                init_constant = paddle.nn.initializer.Constant(value=0.)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):
            # nn.init.constant_(m.bias, 0)
            # nn.init.constant_(m.weight, 1.0)

            init_constant = paddle.nn.initializer.Constant(value=0)
            init_constant(m.bias)
            init_constant = paddle.nn.initializer.Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, Hp, Wp)
            
        x = x.mean(1)  # global pool without cls token

        x = self.norm(x)

        x = self.head(x)
        
        return x
        # xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # for i in range(len(ops)):
        #     features.append(ops[i](xp)

        # return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x

import argparse
def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--data_path', default='/data/xiaolei.qin/Dataset/NWPU/', type=str,
                        help='dataset path')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--split', default='28', type=str,
                        help='number of the classification types')
    parser.add_argument('--tag', default='100', type=str,
                        help='number of the classification types')
    return parser

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    model=ViT_Win_RVSA()
    from paddle.vision.models import MobileNetV3Small
    import numpy as np
    # model=MobileNetV3Small
    from util.datasets import NWPURESISCDataset
    from util.datasets import build_dataset
    # dataset_train = NWPURESISCDataset('/data/xiaolei.qin/Dataset/NWPU/', split='28', tag='100', train=False)
    dataset_train=build_dataset(True,args)
    # input=paddle.concat([dataset_train[0][0],dataset_train[1][0]])
    dataloader=paddle.io.DataLoader(dataset_train)
    for iter,batch in enumerate(dataloader):
        print('batch',batch[0].shape,batch[1].shape)
        break
    # input=paddle.Tensor(np.random.randn(1,3,224,224).astype('float32'))
    output=model(batch[0])
    print('output.sahpe',output.shape)