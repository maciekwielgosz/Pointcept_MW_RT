# Configuration file for point cloud semantic segmentation on the “ptv3_forest_segmentation” dataset.
# All comments are in English.

_base_ = ["../_base_/default_runtime.py"]

# ---------------------------------------------------------------------------- #
# Miscellaneous custom settings
# ---------------------------------------------------------------------------- #

# Total batch size across all GPUs
batch_size = 8 # 16 works ok
# Number of data loader workers per GPU
num_worker = 24  
# Probability for mix augmentations (if used)
mix_prob = 0.8  
# Whether to clear GPU cache between epochs
empty_cache = False  
# Enable automatic mixed precision (AMP)
enable_amp = True  

# ---------------------------------------------------------------------------- #
# Model settings
# ---------------------------------------------------------------------------- #
model = dict(
    type="DefaultSegmentorV2",
    num_classes=3,                     # Four semantic classes: ground, low_vegetation, wood, leaf
    backbone_out_channels=64,          # Output channels of the backbone’s last layer
    backbone=dict(
        type="PT-v3m1",                # Point Transformer V3 (variant m1)
        in_channels=3,                 # Input features: x, y, z coordinates
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria = [
    dict(type="CrossEntropyLoss", loss_weight=0.5, weight=[1.0, 10.0, 2.0], ignore_index=-1),
    dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5, ignore_index=-1),
    ]
)

# ---------------------------------------------------------------------------- #
# Optimizer and scheduler settings
# ---------------------------------------------------------------------------- #
evaluate = True
epoch = 20
eval_epoch = 4


optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
# Lower learning rate for modules whose names include “block”
param_dicts = [dict(keyword="block", lr=0.0006)]

# ---------------------------------------------------------------------------- #
# Dataset settings
# ---------------------------------------------------------------------------- #
dataset_type = "DefaultDataset"
# data_root = "ForInstanceDatasetVer2_output_correct"
data_root = "ForInstanceDatasetVer2_Binbin"

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=[
        "ground",
        "wood",
        "leaf",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # Shift coordinates so that centroid is at origin (with Z included)
            dict(type="CenterShift", apply_z=True),
            dict(
                type="CylinderCropSubsampling",
                radius=20.0,  # Set your desired radius before 12 m, 20 m
                # num_points=10_000,  # Limit to maximum 102400 points
                mode='random',  # Centered spherical cropping
                random_max_range=True,  # Randomly select radius between 10 and 20 m
                min_num_points=8_000,
                max_num_points=12_000,  # Limit to maximum 102400 points
            ),

            dict(type="LabelShift", offset=-1), # Shift labels to start from 0


            # Randomly drop out 20% of points (for robustness)
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),

            # Random rotation around Z axis, with small angle jitter on X and Y axes
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),

            # Random scaling of point cloud
            dict(type="RandomScale", scale=[0.9, 1.1]),

            # Random mirror flip
            dict(type="RandomFlip", p=0.5),

            # Add small jitter noise to coordinates
            dict(type="RandomJitter", sigma=0.005, clip=0.02),

            # Elastic distortion of point positions
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            # Voxel downsampling (grid size = 2cm) and compute grid coordinates & segment labels
            dict(
                type="GridSample",
                grid_size=0.03,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment"),
            ),

            # Random spherical cropping to limit maximum number of points to 32768
            # dict(type="SphereCrop", point_max=102400, mode="random"),
            # Shift coordinates back to move Z centroid to 0
            # dict(type="CenterShift", apply_z=False),

            # Convert everything to PyTorch tensors
            dict(type="ToTensor"),

            # Collect keys into final dictionary for model input
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,        
        loop=2,  # sampling weight
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            # Shift coordinates so that centroid is at origin (with Z included)
            dict(type="CenterShift", apply_z=True),
            dict(
                type="CylinderCropSubsampling",
                radius=20.0,  # Set your desired radius
                num_points=10_000,  # Limit to maximum 250000 points
                mode='center'  # Centered spherical cropping

            ),

            dict(type="LabelShift", offset=-1), # Shift labels to start from 0


            # Voxel downsampling (grid size = 2cm) and compute grid coordinates & segment labels
            dict(
                type="GridSample",
                grid_size=0.03,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment"),
            ),
            # Convert to PyTorch tensors
            dict(type="ToTensor"),
            # Collect keys into final dictionary for model input
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",          # Evaluate on validation split
        data_root=data_root,
        transform=[
            # Shift coordinates so that centroid is at origin (with Z included)
            dict(type="CenterShift", apply_z=True),
            dict(type="LabelShift", offset=-1), # Shift labels to start from 0
            dict(type="SphereCrop", point_max=30000, mode="center"),  # pre-voxel crop
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=True,
                keys=("coord", "segment"),
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "inverse"),
                    feat_keys=["coord"],
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],  # No rotation (angle = 0)
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
    ),
)
