from functools import partial

from .mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_tiny.pth"

model.backbone.net.embed_dim = 192
model.backbone.net.depth = 12
model.backbone.net.num_heads = 3
model.backbone.net.drop_path_rate = 0.1
# 5, 11, 17, 23 for global attention
# model.backbone.net.window_block_indexes = (
#     list(range(0, 5)) + list(range(6, 11)) + list(range(12, 17)) + list(range(18, 23))
# )

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
