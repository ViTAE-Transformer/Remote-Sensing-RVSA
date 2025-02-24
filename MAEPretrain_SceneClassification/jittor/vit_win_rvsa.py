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

# import jt
from functools import partial

# import jittor.nn as nn
import jittor as jt
from jittor import nn
# import jt.nn as nn
# import jt.nn.functional as F
# import jt.utils.checkpoint as checkpoint
#
from jittor.einops import rearrange


# from timm.models.layers import drop_path, to_2tuple, trunc_normal_
#from einops import rearrange, repeat

class DropPath(jt.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f'p={self.drop_prob}'


class Mlp(jt.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(jt.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE: scale factor was wrong in my original version, can set manually to be compatible with previous weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False if not qkv_bias else True)
        self.window_size = window_size
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1
        zeros_par = jt.zeros([rel_sp_dim, head_dim])

        # Initialize learnable parameters for relative positions
        self.rel_pos_h = jt.nn.Parameter(zeros_par)  # 2ws-1, C'
        self.rel_pos_w = jt.nn.Parameter(zeros_par)  # 2ws-1, C'

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape

        # Compute qkv
        qkv = self.qkv(x)
        qkv = jt.transpose(qkv.reshape([B, N, 3, self.num_heads, -1]), [2, 0, 3, 1, 4])  # 3, B, H, N, C
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, N, C

        # Apply scale to q
        q = q * self.scale
        # Calculate attention
        attn = jt.matmul(q, jt.transpose(k, [0, 1, 3, 2]))  # B, H, N, N

        # Apply relative position bias (Assuming `calc_rel_pos_spatial` is defined)
        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        # Softmax normalization
        attn = nn.Softmax(dim=-1)(attn)  # Attention normalization
        attn = self.attn_drop(attn)

        # Attention output
        x = jt.transpose(jt.matmul(attn, v), [0, 2, 1, 3]).reshape([B, N, -1])
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
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def drop_path(x, drop_prob: float, training: bool = True):
    if not training or drop_prob == 0:
        return x
    keep_prob = 1 - drop_prob
    shape = x.shape[0]  # batch_size
    random_tensor = jt.rand([shape]) + keep_prob  # Random values between [0, 1]
    random_tensor = random_tensor.floor()  # Keep 0 or 1 based on probability
    random_tensor = random_tensor.view([-1] + [1] * (x.ndim - 1))  # Expand dimensions
    return x * random_tensor

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Truncated normal initialization for tensors.
    Args:
        tensor: Tensor to initialize
        mean: Mean of the distribution
        std: Standard deviation of the distribution
        a, b: Lower and upper bounds for truncation
    """
    # Generate normal distribution and truncate it
    size = tensor.shape
    values = jt.randn(size, dtype=jt.float32) * std + mean  # Normal distribution

    # Clip values to be within the bounds [a, b]
    values = jt.maximum(values, a)  # values >= a
    values = jt.minimum(values, b)  # values <= b

    tensor[:] = values   # Assign to tensor
    return tensor

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
        jt.arange(q_h)[:, None] * q_h_ratio - jt.arange(k_h)[None, :] * k_h_ratio
    )

     # dist_h [1-ws,ws-1]->[0,2ws-2]

    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        jt.arange(q_w)[:, None] * q_w_ratio - jt.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # get pos encode, qwh, kwh, C'

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim) # B, H, qwh, qww, C
    rel_h = jt.linalg.einsum("byhwc,hkc->byhwk", r_q, Rh)# B, H, qwh, qww, C'; qwh, kWh, C' -> B,H,qwh,qww,kwh
    rel_w = jt.linalg.einsum("byhwc,wkc->byhwk", r_q, Rw)# B,H,qwh,qww,kww

    # attn: B,H,qwh,qww,kwh,kww

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


# def rearrange(tensor, pattern, **kwargs):
#     """
#     Rearranges the input tensor based on the pattern and parameters passed in `kwargs`.
#
#     Args:
#         tensor: The input tensor
#         pattern: A string pattern that defines how to rearrange the dimensions
#         kwargs: Additional arguments for dimensions such as `h`, `b`, `hh`, etc.
#
#     Returns:
#         A tensor with rearranged dimensions according to the pattern
#     """
#
#     # Parse the pattern into input and output parts
#     input_pattern, output_pattern = pattern.split(" -> ")
#
#     # Extract the parameter names from the input pattern
#     param_names = re.findall(r'\((.*?)\)', input_pattern)
#
#     # Replace the parameters in the pattern with the actual values passed via kwargs
#     for param in param_names:
#         dims = param.split()
#         for i, dim in enumerate(dims):
#             if dim in kwargs:
#                 dims[i] = str(kwargs[dim])  # Replace with actual value
#         input_pattern = input_pattern.replace(f"({param})", f"({' '.join(dims)})")
#
#     # Convert the pattern to match the actual tensor shapes
#     input_dims = input_pattern.split()
#     output_dims = output_pattern.split()
#
#     # Create a mapping from input to output dimensions
#     dim_map = {dim: idx for idx, dim in enumerate(input_dims)}
#
#     # Initialize shape list for output tensor
#     shape = list(tensor.shape)
#
#     # Create the new shape based on the output pattern
#     new_shape = []
#     for dim in output_dims:
#         if dim in dim_map:
#             # Use the shape of the corresponding input dimension
#             idx = dim_map[dim]
#             new_shape.append(shape[idx])
#         else:
#             # Flatten the dimension if necessary
#             new_shape.append(-1)  # Flatten if needed
#
#     # Reshape the tensor accordingly
#     rearranged_tensor = jt.reshape(tensor, new_shape)
#
#     # Return the rearranged tensor
#     return rearranged_tensor


class RotatedVariedSizeWindowAttention(jt.Module):
    def __init__(self, dim, num_heads, out_dim=None, window_size=1, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0, attn_head_dim=None, relative_pos_embedding=True,
                 learnable=True, restart_regression=True, attn_window_size=None, shift_size=0,
                 img_size=(1, 1), num_deform=None):
        super().__init__()

        window_size = window_size[0]  # 确保 window_size 是数字

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
        self.rel_pos_h = jt.nn.Parameter(jt.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = jt.nn.Parameter(jt.zeros(rel_sp_dim, head_dim))

        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            if num_deform is None:
                num_deform = 1
            self.num_deform = num_deform

            self.sampling_offsets = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            self.sampling_angles = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 1, kernel_size=1, stride=1)
            )

        self.shift_size = shift_size % self.ws
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, out_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            self.relative_position_bias_table = jt.zeros((window_size + attn_window_size - 1) * (window_size + attn_window_size - 1), num_heads)

            coords_h = jt.arange(self.attn_ws)
            coords_w = jt.arange(self.attn_ws)
            coords = jt.stack(jt.meshgrid([coords_h, coords_w]), 0)
            coords_flatten = jt.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.attn_ws - 1
            relative_coords[:, :, 1] += self.attn_ws - 1
            relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            # Custom initialization function for Jittor (equivalent to trunc_normal_)
            self.relative_position_bias_table = self._trunc_normal(self.relative_position_bias_table, std=0.02)
            print('The relative_pos_embedding is used')

    def _trunc_normal(self, tensor, std=0.02):
        return jt.normal(mean=0, std=std, size=tensor.shape)

    def execute(self, x, H, W):
        # execute 代替 forward 方法
        B, N, C = x.shape
        assert N == H * W
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        b, _, h, w = x.shape
        shortcut = x

        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left

        expand_h, expand_w = h + padding_top + padding_down, w + padding_left + padding_right
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws

        image_reference_h = jt.linspace(-1, 1, expand_h)  # Jittor的linspace
        image_reference_w = jt.linspace(-1, 1, expand_w)
        image_reference = jt.stack(jt.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0)

        window_reference = jt.nn.avg_pool2d(image_reference, kernel_size=self.ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w, self.ws)
        base_coords_h = jt.arange(self.attn_ws) * 2 * self.ws / self.attn_ws / (expand_h - 1)  # ws
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = jt.arange(self.attn_ws) * 2 * self.ws / self.attn_ws / (expand_w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())
        # base_coords = jt.stack(jt.meshgrid(base_coords_w, base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, 1, self.attn_ws, 1, self.attn_ws)

        # extend to each window
        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)  # ws -> 1,ws -> nh,ws
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.attn_ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)  # nw,ws
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.attn_ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)  # nh*ws
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)  # nw*ws


        window_coords = jt.stack(jt.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws) # 1, 2, nh, ws, nw, ws
        base_coords = image_reference

        if self.restart_regression:
            # compute for each head in each batch
            coords = base_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1) # B*nH, 2, nh, ws, nw, ws





        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)
        x = jt.nn.pad(x, (padding_left, padding_right, padding_top, padding_down))

        if self.restart_regression:
            coords = base_coords.repeat(b * self.num_heads, 1, 1, 1, 1, 1)

        if self.learnable:
            # offset factors
            sampling_offsets = self.sampling_offsets(x)

            num_predict_total = b * self.num_heads * self.num_deform

            sampling_offsets = sampling_offsets.reshape([num_predict_total, 2, window_num_h, window_num_w])
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (h // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (w // self.ws)

            # scale fators
            sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape([num_predict_total, 2, window_num_h, window_num_w])

            # rotate factor
            sampling_angle = self.sampling_angles(x)
            sampling_angle = sampling_angle.reshape([num_predict_total, 1, window_num_h, window_num_w])

            # first scale

            window_coords = window_coords * (sampling_scales[:, :, :, None, :, None] + 1)

            # then rotate around window center

            window_coords_r = window_coords.clone()

            # 0:x,column, 1:y,row

            window_coords_r[:, 0, :, :, :, :] = -window_coords[:, 1, :, :, :, :] * jt.sin(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * jt.cos(
                sampling_angle[:, 0, :, None, :, None])
            window_coords_r[:, 1, :, :, :, :] = window_coords[:, 1, :, :, :, :] * jt.cos(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * jt.sin(
                sampling_angle[:, 0, :, None, :, None])

            # system transformation: window center -> image center

            coords = window_reference + window_coords_r + sampling_offsets[:, :, :, None, :, None]

        # final offset
        sample_coords = jt.transpose(coords, [0, 2, 3, 4, 5, 1]).reshape(
            (num_predict_total, self.attn_ws * window_num_h, self.attn_ws * window_num_w, 2))
        # qkv = self.qkv(shortcut.permute(0, 2, 3, 1).reshape(b, -1, self.dim)).permute(0, 2, 1).reshape(b, -1, h,
        #                                                                                                w).reshape(b, 3,
        #                                                                                                           self.num_heads,
        #                                                                                                           self.out_dim // self.num_heads,
        #                                                                                                           h,
        #                                                                                                           w).transpose(
        #     1, 0).reshape(3 * b * self.num_heads, self.out_dim // self.num_heads, h, w)
        qkv = self.qkv(shortcut.permute(0,2,3,1).reshape(b,-1,self.dim)).permute(0,2,1).reshape(b,-1,h,w).reshape(b, 3, self.num_heads, self.out_dim // self.num_heads, h, w).transpose(1, 0).reshape(3*b*self.num_heads, self.out_dim // self.num_heads, h, w)


        # if self.shift_size > 0:

        qkv = jt.nn.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape(
            [3, b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr])
        # else:
        #     qkv = qkv.reshape(3, b*self.num_heads, self.dim // self.num_heads, h, w)
        q, k, v = qkv[0], qkv[1], qkv[2]  # b*self.num_heads, self.out_dim // self.num_heads, H，W

        k_selected = jt.nn.grid_sample(
            k.reshape(
                [num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td, w + padding_lr]),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape([b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr])
        v_selected = jt.nn.grid_sample(
            v.reshape(
                [num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td, w + padding_lr]),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape([b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr])

        q = jt.transpose(q.reshape(
            [b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w, self.ws]),
                             [0, 3, 5, 1, 4, 6, 2]).reshape(
            [b * window_num_h * window_num_w, self.num_heads, self.ws * self.ws, self.out_dim // self.num_heads])
        k = jt.transpose(k_selected.reshape(
            [b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws, window_num_w,
             self.attn_ws]), [0, 3, 5, 1, 4, 6, 2]).reshape(
            [b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
             self.out_dim // self.num_heads])
        v = jt.transpose(v_selected.reshape(
            [b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws, window_num_w,
             self.attn_ws]), [0, 3, 5, 1, 4, 6, 2]).reshape(
            [b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
             self.out_dim // self.num_heads])

        dots = (q @ jt.transpose(k, [0, 1, 3, 2])) * self.scale

        dots = calc_rel_pos_spatial(dots, q, (self.ws, self.ws), (self.attn_ws, self.attn_ws), self.rel_pos_h,
                                    self.rel_pos_w)

        if self.relative_pos_embedding:
            viewshape = self.relative_position_index.shape[0] * self.relative_position_index.shape[1]
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view([viewshape])].view([
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = jt.transpose(relative_position_bias,
                                                      [2, 0, 1]).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        sfm = jt.nn.Softmax(-1)
        attn = sfm(dots)
        out = attn @ v

        # out = out.numpy()
        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                        hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        #out = jt.Var(out)
        # if self.shift_size > 0:
        # out = jt.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out = out[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out = jt.transpose(out, [0, 2, 3, 1]).reshape([B, H * W, -1])

        out = self.proj(out)
        out = self.proj_drop(out)

        return out

    def _grid_sample(self, input, grid):
        # 自定义grid采样函数，具体实现插值逻辑
        # 如果需要，可以在此定义具体的插值方法
        pass

    def _clip_grad(self, grad_norm):
        # print('clip grads of the model for selection')
        nn.utils.clip_grad_norm_(self.sampling_offsets.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(self.sampling_scales.parameters(), grad_norm)

    def _reset_parameters(self):
        if self.learnable:
            nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
            nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            nn.init.constant_(self.sampling_scales[-1].bias, 0.)

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

class Block(jt.Module):
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
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * jt.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * jt.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def execute(self, x, H, W):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(jt.Module):
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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # ('patch x.shape:', x.shape)
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)

class HybridEmbed(jt.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, jt.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with jt.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(jt.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def execute(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Norm2d(jt.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def execute(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

#@BACKBONES.register_module()
class ViT_Win_RVSA(jt.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
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
            self.pos_embed = nn.Parameter(jt.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # self.use_rel_pos_bias = use_rel_pos_bias
        # self.use_checkpoint = use_checkpoint

        # MHSA after interval layers
        # WMHSA in other layers
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=(7, 7) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape, window=((i + 1) % interval != 0), 
                restart_regression=restart_regression)
            for i in range(depth)])
         
        #self.interval = interval

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # manually initialize fc layer
        trunc_normal_(self.head.weight, std=2e-5)

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
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)


    def forward_features(self, x):
        B, C, H, W = x.shape
        # print('x.shape:', x.shape)
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, Hp, Wp)
            
        x = x.mean(dim=1)  # global pool without cls token

        x = self.norm(x)

        x = self.head(x)
        
        return x
        # xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

        # ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        # for i in range(len(ops)):
        #     features.append(ops[i](xp)

        # return tuple(features)

    def execute(self, x):
        x = self.forward_features(x)
        # print('x.shape:', x.shape)
        return x
