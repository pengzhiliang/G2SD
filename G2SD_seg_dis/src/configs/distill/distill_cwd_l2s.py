
_base_ = [
     '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
find_unused_parameters=True
weight=10.0
tau=1.0
crop_size = (512, 512)
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = '',
    student_pretrained = 'student.pth',
    distill_cfg = [ dict(student_module = 'decode_head.conv_seg',
                         teacher_module = 'decode_head.conv_seg',
                         output_hook = True,
                         methods=[dict(type= 'ChannelWiseDivergence',
                                       name='loss_cwd',
                                       student_channels = 150,
                                       teacher_channels = 150,
                                       tau = tau,
                                       weight =weight,
                                       )
                                ]
                        ),
                    
                   ]
    )
# 'configs/mae/upernet_mae_small_12_512_slide_160k_ade20k.py'
norm_cfg = dict(type='SyncBN', requires_grad=True)
student_cfg = dict(                
        model = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=dict(
                type='MAE',
                img_size=512,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                mlp_ratio=4,
                qkv_bias=True,
                use_abs_pos_emb=True, # here different
                use_rel_pos_bias=True,
                init_values=1.,
                drop_path_rate=0.1,
                out_indices=[3, 5, 7, 11]
                ),
            decode_head=dict(
                type='UPerHead',
                in_channels=[384, 384, 384, 384],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                num_classes=150,
                channels=384,
                dropout_ratio=0.1,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=384,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=150,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
                ), 
            train_cfg=dict(),
            test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
            )
    )
#'configs/mae/upernet_mae_base_12_512_slide_160k_ade20k.py'
teacher_cfg = dict(
        model = dict(
            type='EncoderDecoder',
            pretrained=None,
            backbone=dict(
                type='MAE',
                img_size=512,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                use_abs_pos_emb=True, # here different
                use_rel_pos_bias=True,
                init_values=1.,
                drop_path_rate=0.2,
                out_indices=[7, 11, 15, 23]
                ),
            decode_head=dict(
                type='UPerHead',
                in_channels=[1024, 1024, 1024, 1024],
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                num_classes=150,
                channels=1024,
                dropout_ratio=0.1,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                ),
            auxiliary_head=dict(
                type='FCNHead',
                in_channels=1024,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=150,
                norm_cfg=norm_cfg,
                align_corners=False,
                loss_decode=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
                ), 
            train_cfg=dict(),
            test_cfg = dict(mode='slide', crop_size=crop_size, stride=(341, 341))
            )
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,)
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor_distill', 
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65))
                 
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunnerAmp')
fp16 = None
optimizer_config = dict(
            type="DistOptimizerHook",
            update_interval=1,
            grad_clip=None,
            coalesce=True,
            bucket_size_mb=-1,
            use_fp16=True,
        )
