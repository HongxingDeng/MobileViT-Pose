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
load_from = 'work_dirs/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192/best_coco_AP_epoch_180.pth'
resume = False
backend_args = dict(backend='local')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005))
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
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='SimCCLabel',
    input_size=(
        192,
        256,
    ),
    sigma=6.0,
    simcc_split_ratio=2.0)
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
        type='MobileNetV2',
        widen_factor=1.0,
        out_indices=(7, ),
        init_cfg=dict(type='Pretrained', checkpoint='mmcls://mobilenet_v2')),
    head=dict(
        type='SimCCHead',
        in_channels=1280,
        out_channels=14,
        input_size=(
            192,
            256,
        ),
        in_featuremap_size=(
            6,
            8,
        ),
        simcc_split_ratio=2.0,
        deconv_out_channels=None,
        loss=dict(type='KLDiscretLoss', use_target_weight=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(
                192,
                256,
            ),
            sigma=6.0,
            simcc_split_ratio=2.0)),
    test_cfg=dict(flip_test=True))
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
            type='SimCCLabel',
            input_size=(
                192,
                256,
            ),
            sigma=6.0,
            simcc_split_ratio=2.0)),
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
    batch_size=64,
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
                    type='SimCCLabel',
                    input_size=(
                        192,
                        256,
                    ),
                    sigma=6.0,
                    simcc_split_ratio=2.0)),
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
launcher = 'none'
work_dir = './work_dirs/simcc_mobilenetv2_wo-deconv-8xb64-210e_coco-256x192'
