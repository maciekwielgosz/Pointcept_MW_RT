# exp/ptv3_forest_segmentation/ptv3_with_decoder.py

_base_ = ["../_base_/default_runtime.py"]

# ---------------------------------------------------------------------------- #
# Misc
# ---------------------------------------------------------------------------- #
batch_size = 16              # safer default; raise if memory allows
num_worker = 24
empty_cache = False
enable_amp = True
mix_prob = 0.8

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
model = dict(
    type="PTV3WithDecoder",
    num_classes=3,
    backbone_out_channels=64,

    # PTv3 backbone (same as your working setup)
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
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

    # Lightweight query decoder (your module)
    decoder=dict(
        type="SimpleQueryDecoderWithCrossAttention",
        in_channels=64,      # must match backbone_out_channels
        d_model=128,
        num_classes=3,
        return_point_logits=False, 
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        use_score_head=False,        # default & recommended unless you train a score loss
        # train_score_head=False,    # ignored when use_score_head=False
    ),

    # Per-point seg losses (for seg_head)
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=0.5, weight=[1.0, 10.0, 2.0], ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=0.5, ignore_index=-1),
    ],

    # Decoder mask loss (for decoder masks)
    decoder_criteria=[
        dict(
            type="SemanticMaskBCELoss",   # place your loss at pointcept/models/losses/query_mask_loss.py
            ignore_index=-1,
            loss_weight=1.0
        )
    ],
    decoder_loss_weight=1.0,

    # Optional: freeze backbone first if you want to warm up decoder only
    freeze_backbone=False,
)

# ---------------------------------------------------------------------------- #
# Optimizer & Scheduler
# ---------------------------------------------------------------------------- #
evaluate = True
epoch = 500
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
param_dicts = [dict(keyword="block", lr=0.0006)]  # lower LR for heavy blocks

# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #
dataset_type = "DefaultDataset"
data_root = "ForInstanceDatasetVer2_Binbin"

data = dict(
    num_classes=3,
    ignore_index=-1,
    names=["ground", "wood", "leaf"],

    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="CylinderCropSubsampling",
                radius=20.0,
                mode="random",
                random_max_range=True,
                min_num_points=8000,
                max_num_points=12000,
            ),
            dict(type="LabelShift", offset=-1),

            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            dict(
                type="GridSample",
                grid_size=0.03,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment"),
            ),

            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=["coord"],   # features == xyz
            ),
        ],
        test_mode=False,
        loop=8,
    ),

    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="CylinderCropSubsampling",
                radius=20.0,
                num_points=10000,
                mode="center",
            ),
            dict(type="LabelShift", offset=-1),
            dict(
                type="GridSample",
                grid_size=0.03,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                keys=("coord", "segment"),
            ),
            dict(type="ToTensor"),
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
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="LabelShift", offset=-1),
            dict(
                type="CylinderCropSubsampling",
                radius=20.0,
                num_points=30000,
                mode="center",
            ),
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
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ]
            ],
        ),
    ),
)
