from functools import partial
from detectron2 import model_zoo
from ..mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)
model.backbone.net.embed_dim = 384
model.backbone.net.depth = 12
model.backbone.net.num_heads = 6
model.backbone.net.drop_path_rate = 0.1
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=0.7, num_layers=12)
model.distill = True
model.stu_adapter.in_channels = 256
model.stu_adapter.out_channels = 256
model.stu_adapter.num = 5
model_teacher = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model_teacher.backbone.net.embed_dim = 768
model_teacher.backbone.net.depth = 12
model_teacher.backbone.net.num_heads = 12
teacher_path="path_to_teacher"
distill_feat_weight = 0.001
distill_cls_weight = 0.1
distill_warm_step = 500


