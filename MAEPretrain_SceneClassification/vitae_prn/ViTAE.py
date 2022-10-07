import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
import numpy as np
from .vitmodules import ViTAE_ViT_basic

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ViTAE_basic_Tiny': _cfg(),
}

#  The tiny model
@register_model
def ViTAE_basic_Tiny(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 128], token_dims=[64, 64, 256], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 7], NC_heads=[1, 1, 4], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The 6M model
@register_model
def ViTAE_basic_6M(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 128], token_dims=[64, 64, 256], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 10], NC_heads=[1, 1, 4], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The 13M model
@register_model
def ViTAE_basic_13M(pretrained=False, **kwargs): # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 160], token_dims=[64, 64, 320], 
                            downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 11], NC_heads=[1, 1, 5], RC_heads=[1, 1, 1], mlp_ratio=2., NC_group=[1, 1, 64], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The small model
@register_model
def ViTAE_basic_Small(pretrained=False, **kwargs):  # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['performer', 'performer', 'performer_less'], NC_tokens_type=['transformer', 'transformer', 'transformer'], stages=3, embed_dims=[64, 64, 192], token_dims=[96, 192, 384], 
                        downsample_ratios=[4, 2, 2], NC_depth=[0, 0, 14], NC_heads=[1, 1, 6], RC_heads=[1, 1, 1], mlp_ratio=3., NC_group=[1, 1, 96], RC_group=[1, 1, 1], **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The base model
#@register_model
def ViTAE_basic_Base(pretrained=False, **kwargs):  # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['embedding', 'none', 'none'], NC_tokens_type=['transformer_shallow', 'transformer_shallow', 'transformer_shallow'], stages=3, embed_dims=[768, 768, 768], token_dims=[768, 768, 768], 
                            downsample_ratios=[16, 1, 1], NC_depth=[0, 0, 12], NC_heads=[1, 1, 12], RC_heads=[1, 1, 1], mlp_ratio=4., NC_group=[1, 1, 192], RC_group=[1, 1, 1], class_token=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The large model
@register_model
def ViTAE_basic_Large(pretrained=False, **kwargs):  # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['embedding', 'none', 'none'], NC_tokens_type=['transformer_shallow', 'transformer_shallow', 'transformer_shallow'], stages=3, embed_dims=[1024, 1024, 1024], token_dims=[1024, 1024, 1024], 
                            downsample_ratios=[16, 1, 1], NC_depth=[0, 0, 24], NC_heads=[1, 1, 16], RC_heads=[1, 1, 1], mlp_ratio=4., NC_group=[1, 1, 256], RC_group=[1, 1, 1], class_token=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

# The huge model
@register_model
def ViTAE_basic_Huge(pretrained=False, **kwargs):  # adopt performer for tokens to token
    model = ViTAE_ViT_basic(RC_tokens_type=['embedding', 'none', 'none'], NC_tokens_type=['transformer_shallow', 'transformer_shallow', 'transformer_shallow'], stages=3, embed_dims=[1280, 1280, 1280], token_dims=[1280, 1280, 1280], 
                            downsample_ratios=[16, 1, 1], NC_depth=[0, 0, 32], NC_heads=[1, 1, 16], RC_heads=[1, 1, 1], mlp_ratio=4., NC_group=[1, 1, 320], RC_group=[1, 1, 1], class_token=True, **kwargs)
    model.default_cfg = default_cfgs['ViTAE_basic_Tiny']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model