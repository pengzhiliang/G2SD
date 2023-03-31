
from contextlib import asynccontextmanager
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn as nn
import torch.nn.functional as F
kl_loss = torch.nn.KLDivLoss(reduction='none')
def cal_relation_loss(student_attn_list, teacher_attn_list, Ar=1, layer_index=1):
    layer_num = len(student_attn_list)
    relation_loss = 0.
    l=0
    for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
        l = l + 1
      
        B, N, Cs = student_att[0].shape
        _, _, Ct = teacher_att[0].shape

        for i in range(3):
            for j in range(3):
                # (B, Ar, N, Cs // Ar) @ (B, Ar, Cs // Ar, N)
                # (B, Ar) + (N, N)
                matrix_i_s = student_att[i].view(B, N, Ar, Cs//Ar).transpose(1, 2) / (Cs/Ar)**0.5
                matrix_j_s = student_att[j].view(B, N, Ar, Cs//Ar).permute(0, 2, 3, 1)
                As_ij = (matrix_i_s @ matrix_j_s) 

                matrix_i = teacher_att[i].view(B, N, Ar, Ct//Ar).transpose(1, 2) / (Ct/Ar)**0.5
                matrix_j = teacher_att[j].view(B, N, Ar, Ct//Ar).permute(0, 2, 3, 1)
                At_ij = (matrix_i @ matrix_j)
              
               
                loss = kl_loss(F.log_softmax(As_ij.float(),dim=-1),F.softmax(At_ij.float(),dim=-1)).sum(dim=-1).mean()
                relation_loss += loss
    
    return relation_loss/(9. * layer_num)

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None, model_teacher=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
       
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            noise = torch.rand(samples.shape[0], 196, device=samples.device)  
            
            if model_teacher is not None:
                   pred_decoder_feature_tea, decoder_hidden_tea, mask, prenorm_feature_teacher, qkv_list_encoder_tea, qkv_list_decoder_tea = model_teacher(samples,noise, mask_ratio=args.mask_ratio)
            pred_decoder_feature_stu, decoder_hidden_stu, mask, prenorm_feature, pred_feature, qkv_list_encoder_stu, qkv_list_decoder_stu = model(samples,noise, mask_ratio=args.mask_ratio)


            labels_feature_decoder = nn.LayerNorm(decoder_hidden_tea.shape[-1], eps=1e-6, elementwise_affine=False)(decoder_hidden_tea)
            loss_feature_decoder = F.smooth_l1_loss(pred_decoder_feature_stu, labels_feature_decoder, beta=2.0)
           
            labels_feature = nn.LayerNorm(prenorm_feature_teacher.shape[-1], eps=1e-6, elementwise_affine=False)(prenorm_feature_teacher)
            loss_feature = F.smooth_l1_loss(pred_feature, labels_feature, beta=2.0)
         
        loss = 0.*loss_feature + loss_feature_decoder
        loss_value = loss.item()
  
        loss_feature_value = loss_feature.item()
        loss_feature_decoder_value = loss_feature_decoder.item()
     
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
      
        metric_logger.update(loss_feature_value=loss_feature_value)
        metric_logger.update(loss_feature_decoder_value=loss_feature_decoder_value)
      
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}