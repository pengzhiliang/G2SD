IMAGENET_DIR='ImageNet_train_dir'  
JOB_DIR='output_dir'
python -m torch.distributed.launch --nproc_per_node=8 --master_port 50010   main_distill.py \
--data_path $IMAGENET_DIR --batch_size 256  --accum_iter 2 \
--teacher_model mae_vit_base_patch16 \
--student_model mae_vit_small_patch16_dec256d4b  \
--mask_ratio 0.75 --mask True --norm_pix_loss  --loss_type smoothl1 \
--epochs 300 --warmup_epochs 10 --blr 1.5e-4 --weight_decay 0.05 \
--output_dir $JOB_DIR --log_dir $JOB_DIR --drop_path 0.0  --beta 0.95   \
--teacher_path  'teacher_pretrain_path' 
