custom_imports = dict(
    imports=["mmpretrain.datasets.transforms"], allow_failed_imports=False
)

# dataset settings
dataset_type = "DOTADataset"
data_root = "data/split_ss_dota/"
backend_args = None

branch_field = ["sup", "unsup_teacher", "unsup_student"]
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(type="mmdet.Pad", size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="mmdet.MultiBranch",
        branch_field=branch_field,
        sup=dict(type="mmdet.PackDetInputs"),
    ),
]

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    dict(type="mmdet.Pad", size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "homography_matrix",
        ),
    ),
]

# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    dict(
        type="RandomApply",
        transforms=dict(
            type="mmpretrain.ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        prob=0.8,
    ),
    dict(type="mmpretrain.RandomGrayscale", prob=0.2, keep_channels=True),
    dict(
        type="mmpretrain.GaussianBlur",
        radius=None,
        prob=0.5,
        magnitude_level=1.9,
        magnitude_range=[0.1, 2.0],
        magnitude_std="inf",
        total_level=1.9,
    ),
    dict(type="mmdet.Pad", size_divisor=32, pad_val=dict(img=(114, 114, 114))),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "homography_matrix",
        ),
    ),
]

# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.LoadEmptyAnnotations"),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.RandomFlip",
        prob=0.75,
        direction=["horizontal", "vertical", "diagonal"],
    ),
    dict(
        type="mmdet.MultiBranch",
        branch_field=branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    ),
]

val_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(type="mmdet.LoadAnnotations", with_bbox=True, box_type="qbox"),
    dict(type="ConvertBoxType", box_type_mapping=dict(gt_bboxes="rbox")),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

test_pipeline = [
    dict(type="mmdet.LoadImageFromFile", backend_args=backend_args),
    dict(type="mmdet.Resize", scale=(1024, 1024), keep_ratio=True),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

batch_size = 3
num_workers = 6

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="train_10_labeled/annfiles",
    data_prefix=dict(img_path="train_10_labeled/images/"),
    filter_cfg=dict(filter_empty_gt=True), 
    pipeline=sup_pipeline,
)

unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="train_10_unlabeled/empty_annfiles/",
    data_prefix=dict(img_path="train_10_unlabeled/images/"),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(
        type="mmdet.MultiSourceSampler", batch_size=batch_size, source_ratio=[2, 1]
    ),
    dataset=dict(type="ConcatDataset", datasets=[labeled_dataset, unlabeled_dataset]),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="val/annfiles/",
        data_prefix=dict(img_path="val/images/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="DOTAMetric", metric="mAP")
test_evaluator = val_evaluator