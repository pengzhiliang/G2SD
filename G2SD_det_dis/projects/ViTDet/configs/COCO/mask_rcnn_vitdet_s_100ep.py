from functools import partial

from .mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

train.init_checkpoint = "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_small.pth"

model.backbone.net.embed_dim = 384
model.backbone.net.depth = 12
model.backbone.net.num_heads = 6
model.backbone.net.drop_path_rate = 0.1
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)

