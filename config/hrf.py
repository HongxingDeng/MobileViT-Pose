default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5))
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.0005, betas=(
            0.9,
            0.999,
        ), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(relative_position_bias_table=dict(decay_mult=0.0))))
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[
            170,
            200,
        ],
        gamma=0.1,
        by_epoch=True),
]
auto_scale_lr = dict(base_batch_size=256)
codec = dict(
    type='MSRAHeatmap',
    input_size=(
        192,
        256,
    ),
    heatmap_size=(
        48,
        64,
    ),
    sigma=2)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRFormer',
        in_channels=3,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        extra=dict(
            drop_path_rate=0.1,
            with_rpe=True,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, ),
                num_heads=[
                    2,
                ],
                num_mlp_ratios=[
                    4,
                ]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMERBLOCK',
                num_blocks=(
                    2,
                    2,
                ),
                num_channels=(
                    32,
                    64,
                ),
                num_heads=[
                    1,
                    2,
                ],
                mlp_ratios=[
                    4,
                    4,
                ],
                window_sizes=[
                    7,
                    7,
                ]),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMERBLOCK',
                num_blocks=(
                    2,
                    2,
                    2,
                ),
                num_channels=(
                    32,
                    64,
                    128,
                ),
                num_heads=[
                    1,
                    2,
                    4,
                ],
                mlp_ratios=[
                    4,
                    4,
                    4,
                ],
                window_sizes=[
                    7,
                    7,
                    7,
                ]),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMERBLOCK',
                num_blocks=(
                    2,
                    2,
                    2,
                    2,
                ),
                num_channels=(
                    32,
                    64,
                    128,
                    256,
                ),
                num_heads=[
                    1,
                    2,
                    4,
                    8,
                ],
                mlp_ratios=[
                    4,
                    4,
                    4,
                    4,
                ],
                window_sizes=[
                    7,
                    7,
                    7,
                    7,
                ])),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpose/pretrain_models/hrformer_small-09516375_20220226.pth'
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=14,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(
                192,
                256,
            ),
            heatmap_size=(
                48,
                64,
            ),
            sigma=2)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '../data/coco/'
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=(
        192,
        256,
    )),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='MSRAHeatmap',
            input_size=(
                192,
                256,
            ),
            heatmap_size=(
                48,
                64,
            ),
            sigma=2)),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(
        192,
        256,
    )),
    dict(type='PackPoseInputs'),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root='../data/coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(type='TopdownAffine', input_size=(
                192,
                256,
            )),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(
                        192,
                        256,
                    ),
                    heatmap_size=(
                        48,
                        64,
                    ),
                    sigma=2)),
            dict(type='PackPoseInputs'),
        ]))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='../data/coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(
                192,
                256,
            )),
            dict(type='PackPoseInputs'),
        ]))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='../data/coco/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(
                192,
                256,
            )),
            dict(type='PackPoseInputs'),
        ]))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='../data/coco/annotations/person_keypoints_val2017.json')
test_evaluator = dict(
    type='CocoMetric',
    ann_file='../data/coco/annotations/person_keypoints_val2017.json')
fp16 = dict(loss_scale='dynamic')
launcher = 'none'
work_dir = './work_dirs/td-hm_hrformer-small_8xb32-210e_coco-256x192'
