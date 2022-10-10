_base_ = [
    '../_base_/models/upernet_vit_base_win.py', #'../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'isAIDDataset'
data_root = '/workspace/CV/users/wangdi153/Dataset/isaid_patches'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (896, 896)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(896, 896), ratio_range=None, keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip',prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(896, 896),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Pad', size=(896,896), pad_val=0, seg_pad_val=255),
            dict(type='RandomFlip',prob=1.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        pipeline=val_pipeline))
        
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                        num_layers=12, 
                        layer_decay_rate=0.9,
                            )
                            )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
 
optimizer_config = dict(grad_clip=None)

model = dict(
    pretrained = '../mae-main/output/millionAID_224/1600_0.75_0.00015_0.05_2048/checkpoint-1599.pth',
    backbone=dict(
        type='ViT_Win_RVSA_V3_KVDIFF_WSZ7',
        img_size=896,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        use_abs_pos_emb=True
    ),
    decode_head=dict(
        num_classes=16,
        ignore_index=255
    ),
    auxiliary_head=dict(
        num_classes=16,
        ignore_index=255
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )
