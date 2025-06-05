_base_ = [
    "../_base_/detectors/rotated-fcos-le90_r50_fpn_dotav15.py",
    "../_base_/default_runtime.py",
    "../_base_/datasets/semi_dotav15_detection.py",
]
# todo: fix this import issue
custom_imports = dict(
    imports=["mmrotate.engine.hooks.mean_teacher_hook"], allow_failed_imports=False
)

detector = _base_.model
model = dict(
    _delete_=True,
    type="RotatedDenserTeacher",
    detector=detector,
    data_preprocessor=dict(
        type="mmdet.MultiBranchDataPreprocessor",
        data_preprocessor=detector.data_preprocessor,
    ),
    semi_train_cfg=dict(
        freeze_teacher=True,
        iter_count=0,
        burn_in_steps=10000,
        sup_weight=1.0,
        unsup_weight=0.5,
        cls_weight=1.0,
        bbox_loss_type="l1",  # option: 'l1'
        bbox_weight=1.0,
        centerness_weight=1.0,
        centerness_loss_type="SmoothL1",  # option: 'SmoothL1', 'BCE'
        dense_msl=True,
        dense_msl_weight=1.0,
    ),
    semi_test_cfg=dict(predict_on="teacher"),
)

percentage = 30
labeled_dataset = _base_.labeled_dataset
labeled_dataset.ann_file = f"train_{percentage}_labeled/annfiles"
labeled_dataset.data_prefix = dict(img_path=f"train_{percentage}_labeled/images/")

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.ann_file = f"train_{percentage}_unlabeled/empty_annfiles/"
unlabeled_dataset.data_prefix = dict(img_path=f"train_{percentage}_unlabeled/images/")

batch_size = 3
num_workers = 6
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(
        type="mmdet.MultiSourceSampler", batch_size=batch_size, source_ratio=[2, 1]
    ),
    dataset=dict(datasets=[labeled_dataset, unlabeled_dataset]),
)

# training schedule for 180k
train_cfg = dict(type="IterBasedTrainLoop", max_iters=180000, val_interval=3200)
val_cfg = dict(type="mmdet.TeacherStudentValLoop")
test_cfg = dict(type="TestLoop")

# learning rate policy
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type="MultiStepLR",
        begin=1000,
        end=180000,
        by_epoch=False,
        milestones=[120000, 160000],
        gamma=0.1,
    ),
]

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.0025, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=3200, max_keep_ckpts=1000, save_best="auto"
    ),
    logger=dict(type="LoggerHook", interval=50),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

custom_hooks = [
    dict(type="MeanTeacherHook", start_iter=3200, momentum=0.0004),
]

vis_backends = [dict(type="TensorboardVisBackend")]

visualizer = dict(
    type="RotLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
