 ./tools/plain_distill_net.py \
        --num-gpus 8  \
        --resume  \
        --config-file projects/ViTDet/configs/COCO/distillation/mask_rcnn_vitdet_b2s_100ep.py \
          train.init_checkpoint=path_to_student_init  \
          train.output_dir=outputdir \
          model.backbone.net.drop_path_rate=0.1 \
          dataloader.train.total_batch_size=64 \
          optimizer.lr=1e-4 \
          optimizer.weight_decay=0.05 \
          teacher_path=teacher_path \
          distill_feat_weight=0.001 \
          distill_cls_weight=0.1 \
          distill_warm_step=500