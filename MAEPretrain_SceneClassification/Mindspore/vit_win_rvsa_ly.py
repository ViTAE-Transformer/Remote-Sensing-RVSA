from functools import partial
import mindspore.numpy as mnp
import mindspore as ms
from mindspore import nn, ops
from mindspore import Tensor
import numpy as np
from mindspore.common.initializer import Constant
import collections
from itertools import repeat
import mindspore
from mindspore.common.initializer import initializer, TruncatedNormal
from mindspore import Parameter
from einops import rearrange
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.operations import nn_ops as NN_OPS


def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
    if input.ndim == 4:
        _grid_sampler_2d = _get_cache_prim(NN_OPS.GridSampler2D)(mode, padding_mode, align_corners)
        return _grid_sampler_2d(input, grid)
    _grid_sampler_3d = _get_cache_prim(NN_OPS.GridSampler3D)(mode, padding_mode, align_corners)
    return _grid_sampler_3d(input, grid)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with different dimension tensors, not just 2D ConvNets

    # 创建与x形状匹配的随机张量
    random_tensor = ops.bernoulli(mnp.ones(shape) * keep_prob, dtype=mnp.float32)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob

    return x * random_tensor


def _ntuple(n):
    def parse(x):
        # 如果x是一个可迭代对象，且不是字符串类型，返回元组
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        # 否则，重复x，直到元组长度为n
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class DropPath(nn.Cell):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def construct(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def __repr__(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define layers
        self.fc1 = nn.Dense(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)  # MindSpore uses keep_prob, so drop is 1 - drop

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)  # Apply dropout after the second linear layer
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., window_size=None, attn_head_dim=None):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        # Scale factor
        self.scale = qk_scale or head_dim ** -0.5

        # Linear layers
        self.qkv = nn.Dense(dim, all_head_dim * 3, has_bias=qkv_bias)
        self.window_size = window_size
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1

        # Parameter initialization for relative positional encodings
        zeros_par = ops.Zeros()((rel_sp_dim, head_dim), ms.float32)
        self.rel_pos_h = ms.Parameter(zeros_par, name="rel_pos_h")
        self.rel_pos_w = ms.Parameter(zeros_par, name="rel_pos_w")

        # Dropout layers
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(all_head_dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape
        # Compute QKV
        qkv = self.qkv(x)
        qkv = ops.Transpose()(qkv.reshape(B, N, 3, self.num_heads, -1), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, H, N, C

        # Scale Q
        q = q * self.scale
        matmul_op = ops.BatchMatMul(transpose_b=True)
        attn = matmul_op(q, k)
        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        # Softmax and attention dropout
        attn = ops.Softmax(-1)(attn)
        attn = self.attn_drop(attn)
        matmul_op = ops.BatchMatMul()
        x0 = matmul_op(attn, v)
        x = ops.Transpose()(x0, (0, 2, 1, 3)).reshape(B, N, -1)
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
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))  # B, H // ws, W // ws, ws, ws, C
    windows = windows.reshape(-1, window_size, window_size, C)  # Flatten the window dimension
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
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = ops.Transpose()(x, (0, 1, 3, 2, 4, 5))  # B, H//ws, W//ws, ws, ws, C
    x = x.reshape(B, H, W, -1)  # Flatten the window dimensions back
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
    dist_h = ops.Range()(ms.Tensor(0, dtype=ms.int32), ms.Tensor(q_h, dtype=ms.int32), ms.Tensor(1, dtype=ms.int32))[:,
             None] * q_h_ratio - ops.Range()(ms.Tensor(0, dtype=ms.int32), ms.Tensor(k_h, dtype=ms.int32),
                                             ms.Tensor(1, dtype=ms.int32))[None, :] * k_h_ratio
    dist_h += (k_h - 1) * k_h_ratio

    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = ops.Range()(ms.Tensor(0, dtype=ms.int32), ms.Tensor(q_w, dtype=ms.int32), ms.Tensor(1, dtype=ms.int32))[:,
             None] * q_w_ratio - ops.Range()(ms.Tensor(0, dtype=ms.int32), ms.Tensor(k_w, dtype=ms.int32),
                                             ms.Tensor(1, dtype=ms.int32))[None, :] * k_w_ratio
    dist_w += (k_w - 1) * k_w_ratio

    # get pos encode, qwh, kwh, C'
    dist_h = dist_h.astype(ms.int64)
    dist_w = dist_w.astype(ms.int64)

    Rh = rel_pos_h[dist_h]  # Relational positional encoding for height
    Rw = rel_pos_w[dist_w]  # Relational positional encoding for width

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)

    einsum_op1 = ops.Einsum("byhwc,hkc->byhwk")
    rel_h = einsum_op1((r_q, Rh))
    einsum_op2 = ops.Einsum("byhwc,wkc->byhwk")
    rel_w = einsum_op2((r_q, Rw))
    attn_slice = attn[:, :, sp_idx:, sp_idx:]
    reshaped_attn = attn_slice.reshape(B, -1, q_h, q_w, k_h, k_w)
    updated_attn = reshaped_attn + rel_h[:, :, :, :, :, None] + rel_w[:, :, :, :, None, :]
    attn = updated_attn.reshape(B, -1, q_h * q_w, k_h * k_w)
    return attn


class RotatedVariedSizeWindowAttention(nn.Cell):
    def __init__(self, dim, num_heads, out_dim=None, window_size=1, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0, attn_head_dim=None, relative_pos_embedding=True, learnable=True,
                 restart_regression=True,
                 attn_window_size=None, shift_size=0, img_size=(1, 1), num_deform=None):
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
        h_zeros = ms.Tensor(mnp.zeros((rel_sp_dim, head_dim)), ms.float32)
        w_zeros = ms.Tensor(mnp.zeros((rel_sp_dim, head_dim)), ms.float32)
        self.rel_pos_h = ms.Parameter(h_zeros, name="rel_pos_h")
        self.rel_pos_w = ms.Parameter(w_zeros, name="rel_pos_w")

        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            # if num_deform is None, we set num_deform to num_heads as default
            if num_deform is None:
                num_deform = 1
            self.num_deform = num_deform

            self.sampling_offsets = nn.SequentialCell([
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(alpha=1e-2),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1, has_bias=True,
                          pad_mode='pad')
            ])
            self.sampling_scales = nn.SequentialCell([
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(alpha=1e-2),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1, has_bias=True,
                          pad_mode='pad')
            ])
            # add angle
            self.sampling_angles = nn.SequentialCell([
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(alpha=1e-2),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 1, kernel_size=1, stride=1, has_bias=True,
                          pad_mode='pad')
            ])

        self.shift_size = shift_size % self.ws
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Dense(dim, out_dim * 3, has_bias=qkv_bias)

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(out_dim, out_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            zeros_par = ops.Zeros()(
                ((window_size + attn_window_size - 1) * (window_size + attn_window_size - 1), num_heads), ms.float32)
            self.relative_position_bias_table = Parameter(zeros_par,
                                                          name="relative_position_bias_table")  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = ops.arange(self.attn_ws)
            coords_w = ops.arange(self.attn_ws)
            coords = ops.stack(ops.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
            coords_flatten = coords.view(2, -1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.attn_ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.attn_ws - 1
            relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.relative_position_index = Parameter(relative_position_index, requires_grad=False)

    def construct(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W
        x = ms.ops.reshape(x, (B, H, W, C))
        x = ms.ops.transpose(x, (0, 3, 1, 2))
        b, _, h, w = x.shape
        shortcut = x
        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left
        expand_h, expand_w = h + padding_top + padding_down, w + padding_left + padding_right

        # window num in padding features
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws

        image_reference_h = ops.linspace(-1, 1, expand_h)
        image_reference_w = ops.linspace(-1, 1, expand_w)
        # 创建网格，并堆叠和转置
        image_reference = ops.transpose(ops.stack(ops.meshgrid(image_reference_w, image_reference_h, indexing='ij'), 0),
                                        (0, 2, 1))
        image_reference = ops.expand_dims(image_reference, 0)
        # position of the window relative to the image center
        from mindspore.ops.operations import AvgPool

        # 创建 AvgPool 实例
        avg_pool = AvgPool(kernel_size=self.ws, strides=self.ws, pad_mode="valid")

        # 使用 AvgPool 对输入进行池化
        window_reference = avg_pool(image_reference)

        image_reference = ops.reshape(image_reference, (1, 2, window_num_h, self.ws, window_num_w, self.ws))
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]

        window_reference = ops.reshape(window_reference, (1, 2, window_num_h, 1, window_num_w, 1))
        base_coords_h = (ops.arange(self.attn_ws).astype(ms.float32) * 2 * self.ws / self.attn_ws) / (expand_h - 1)
        base_coords_h = base_coords_h - ops.ReduceMean()(base_coords_h.astype(ms.float32))
        base_coords_w = ops.arange(self.attn_ws).astype(ms.float32) * 2 * self.ws / self.attn_ws / (expand_w - 1)
        base_coords_w = base_coords_w - ops.ReduceMean()(base_coords_w.astype(ms.float32))
        # 扩展 base_coords_h
        base_coords_h_0 = ops.expand_dims(base_coords_h, 0)
        expanded_base_coords_h = ops.repeat_elements(base_coords_h_0, window_num_h, 0)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.attn_ws
        # expanded_base_coords_w = base_coords_w.unsqueeze(0).repeat(window_num_w, 1) # nw,ws
        expanded_base_coords_w = ops.repeat_elements(ops.expand_dims(base_coords_w, 0), window_num_w, 0)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.attn_ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(window_num_h * self.attn_ws)  # nh*ws
        expanded_base_coords_w = expanded_base_coords_w.reshape(window_num_w * self.attn_ws)  # nw*ws
        window_coords = ops.transpose(
            ops.stack(ops.meshgrid(expanded_base_coords_w, expanded_base_coords_h, indexing='ij'), 0),
            (0, 2, 1)
        ).reshape((1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws))  # 1, 2, nh, ws, nw, ws
        # base_coords = window_reference+window_coords
        base_coords = image_reference

        # padding feature
        pad = ops.Pad(paddings=([0, 0], [0, 0], [padding_top, padding_down], [padding_left, padding_right]))
        x = pad(x)
        # 应用 padding 操作
        if self.learnable:
            # offset factors
            sampling_offsets = self.sampling_offsets(x)
            num_predict_total = b * self.num_heads * self.num_deform
            sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (h // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (w // self.ws)
            # scale fators
            sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)
            sampling_angle = self.sampling_angles(x)
            sampling_angle = sampling_angle.reshape(num_predict_total, 1, window_num_h, window_num_w)

            window_coords = window_coords * (sampling_scales[:, :, :, None, :, None] + 1)

            window_coords_r = window_coords.asnumpy().copy()  # 使用 NumPy 的 copy 来复制
            window_coords_r = ms.Tensor(window_coords_r)  # 将 NumPy 数组转换回 Tensor

            # 计算新的 window_coords_r
            window_coords_r[:, 0, :, :, :, :] = -window_coords[:, 1, :, :, :, :] * ops.sin(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * ops.cos(
                sampling_angle[:, 0, :, None, :, None])
            window_coords_r[:, 1, :, :, :, :] = window_coords[:, 1, :, :, :, :] * ops.cos(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * ops.sin(
                sampling_angle[:, 0, :, None, :, None])

            coords = window_reference + window_coords_r + sampling_offsets[:, :, :, None, :, None]

        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.attn_ws * window_num_h,
                                                                 self.attn_ws * window_num_w, 2)
        qkv = self.qkv(ms.ops.reshape(shortcut.permute(0, 2, 3, 1), (b, -1, self.dim)))  # 对的
        qkv = qkv.permute(0, 2, 1)
        qkv = qkv.reshape(b, -1, h, w)
        qkv = qkv.reshape(b, 3, self.num_heads, self.out_dim // self.num_heads, h, w)
        qkv = ops.Transpose()(qkv, (1, 0, 2, 3, 4, 5))
        qkv = qkv.reshape(3 * b * self.num_heads, self.out_dim // self.num_heads, h, w)  # 对的

        # if self.shift_size > 0:
        # Create the Pad operator
        pad = ops.Pad(paddings=([0, 0], [0, 0], [padding_top, padding_down], [padding_left, padding_right]))
        qkv = pad(qkv)  # 对的

        qkv = qkv.reshape(3, b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr)

        q, k, v = qkv[0], qkv[1], qkv[2]  # b*self.num_heads, self.out_dim // self.num_heads, H，W

        new_k = k.reshape(num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td,
                          w + padding_lr)
        k_selected = grid_sample(new_k, grid=sample_coords, padding_mode='zeros',
                                 align_corners=True)
        k_selected = k_selected.reshape(b * self.num_heads, self.out_dim // self.num_heads, h + padding_td,
                                        w + padding_lr)
        v_selected = ms.ops.grid_sample(
            v.reshape(num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td,
                      w + padding_lr),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr)
        q = q.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w,
                      self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w, self.num_heads,
                                                                    self.ws * self.ws, self.out_dim // self.num_heads)

        k = k_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws,
                               window_num_w, self.attn_ws)

        k = k.permute(0, 3, 5, 1, 4, 6, 2)
        k = k.reshape(b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
                      self.out_dim // self.num_heads)

        v = v_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws,
                               window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(
            b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
            self.out_dim // self.num_heads)

        dots = (q @ ops.Transpose()(k, (0, 1, 3, 2))) * self.scale
        dots = calc_rel_pos_spatial(dots, q, (self.ws, self.ws), (self.attn_ws, self.attn_ws), self.rel_pos_h,
                                    self.rel_pos_w)
        if self.relative_pos_embedding:
            viewshape = self.relative_position_index.shape[0] * self.relative_position_index.shape[1]
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(viewshape)]
            relative_position_bias = relative_position_bias.view(self.ws * self.ws, self.attn_ws * self.attn_ws, -1)
            relative_position_bias = ops.Transpose()(relative_position_bias, (2, 0, 1))  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)
        sfm = ops.Softmax(axis=-1)
        attn = sfm(dots)
        out = attn @ v
        out = out.reshape(b, window_num_h, window_num_w, self.num_heads, self.ws, self.ws, -1)  # 先进行维度扩展
        d = out.shape[-1]
        out = out.permute(0, 3, 6, 1, 4, 2, 5)  # 调整维度顺序
        out = out.reshape(b, self.num_heads * d, window_num_h * self.ws, window_num_w * self.ws)
        out = out[:, :, padding_top:h + padding_top, padding_left:w + padding_left]
        out = ops.transpose(out, (0, 2, 3, 1)).reshape(B, H * W, -1)  
        out = self.proj(out)
        out = self.proj_drop(out)
        #
        return out

    def _clip_grad(self, grad_norm):
        nn.utils.clip_grad_norm_(self.sampling_offsets.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(self.sampling_scales.parameters(), grad_norm)

    def _reset_parameters(self):
        if self.learnable:
            init_constant = Constant(value=0.)
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
        h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[
            1] + self.shift_size + self.padding_right
        flops *= (h / self.ws * w / self.ws)

        # for sampling
        flops_sampling = 0
        if self.learnable:
            # pooling
            flops_sampling += h * w * self.dim
            # regressing the shift and scale
            flops_sampling += 2 * (h / self.ws + w / self.ws) * self.num_heads * 2 * self.dim
            # calculating the coords
            flops_sampling += h / self.ws * self.attn_ws * w / self.ws * self.attn_ws * 2
        # grid sampling attended features
        flops_sampling += h / self.ws * self.attn_ws * w / self.ws * self.attn_ws * self.dim

        flops += flops_sampling

        return flops


class Block(nn.Cell):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, window=False, restart_regression=True):
        super().__init__()
        self.norm1 = norm_layer(normalized_shape=(dim,))

        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        else:
            self.attn = RotatedVariedSizeWindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                restart_regression=restart_regression)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(normalized_shape=(dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1, self.gamma_2 = None, None

    def construct(self, x, H, W):
        if self.gamma_1 is None:
            x0 = self.norm1(x)

            x1 = self.attn(x0, H, W)
            
            x = x + self.drop_path(x1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
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

        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              has_bias=True, pad_mode='pad')  # mindspore默认不添加bias，所以要加上

    def construct(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = ops.Reshape()(x, (B, -1, Hp * Wp))  # Flattening the spatial dimensions
        x = ops.Transpose()(x, (0, 2, 1))  # Transpose to (B, num_patches, embed_dim)

        return x, (Hp, Wp)


class ViT_Win_RVSA(nn.Cell):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False,
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm)  # , eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        if use_abs_pos_emb:
            zeros_par = ops.Zeros()((1, num_patches + 1, embed_dim), ms.float32)
            self.pos_embed = Parameter(zeros_par, name="pos_embed", requires_grad=True)
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.asnumpy().item() for x in mnp.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.CellList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values,
                window_size=(7, 7) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0),
                restart_regression=restart_regression)
            for i in range(depth)])
        self.norm = norm_layer(normalized_shape=(embed_dim,))

        self.head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            m.weight.init_parameters_data(TruncatedNormal(std=0.02))
            if isinstance(m, nn.Dense) and m.bias is not None:
                # nn.init.constant_(m.bias, 0)

                init_constant = Constant(value=0.)
                init_constant(m.bias)
        elif isinstance(m, nn.LayerNorm):

            init_constant = Constant(value=0)
            init_constant(m.bias)
            init_constant = Constant(value=1.0)
            init_constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def construct(self, x):
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:, :]
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, Hp, Wp)
        x = x.mean(1)  # global pool without cls token
        x = self.norm(x)
        x = self.head(x)
        return x


import argparse
import mindspore.dataset as ds


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--data_path',
                        default='./NWPU-RESISC45/', type=str,
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
    model = ViT_Win_RVSA()
    from util.datasets import build_dataset

    dataset_train = build_dataset(True, args)
    dataset = ds.GeneratorDataset(dataset_train, column_names=["data", "label"])
    # 创建 DataLoader，批处理大小为48
    dataloader = dataset.batch(8, drop_remainder=True)
    for iter, batch in enumerate(dataloader):
        # # print('batch', batch[0].shape, batch[1].shape)
        break
    output = model(batch[0])
    # print('output.sahpe', output.shape)
