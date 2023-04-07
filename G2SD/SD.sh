IMAGENET_DIR='ImageNet_train_dir'
VAL_DIR='ImageNet_val_dir'
JOB_DIR='outputdir'
PRETRAIN_CHKPT='student_init'
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune_dis.py \
    --accum_iter 1 \
    --batch_size 128 \
    --model vit_small_patch16 \
    --epochs 200 \
    --blr 1.e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.  --mixup 0.8 --cutmix 1.0 --reprob 0.25  \
    --dist_eval --data_path ${IMAGENET_DIR} --output_dir $JOB_DIR   --log_dir $JOB_DIR \
    --finetune ${PRETRAIN_CHKPT}  --eval_data_path ${VAL_DIR}  --teacher_model vit_base_patch16 \
    --teacher_path  mae_finetuned_vit_base.pth 

