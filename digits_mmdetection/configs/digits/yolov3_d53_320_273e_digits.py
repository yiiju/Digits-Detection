_base_ = '../yolo/yolov3_d53_mstrain-608_273e_coco.py'
model = dict(
    bbox_head=dict(num_classes=10,))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '../data/'
classes = ('10', '1', '2', '3', '4', '5', '6', '7', '8', '9')
# img_norm_cfg = dict(mean=[112.1745, 110.67, 111.945],
#                     std=[32.487, 33.354, 30.855], to_rgb=True)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=30,
    workers_per_gpu=20,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_coco_anno.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        classes=classes,),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'train_coco_anno.json',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline,
        classes=classes,),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'fake_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        classes=classes,))
total_epochs = 50
evaluation = dict(interval=1, metric=['bbox'])
