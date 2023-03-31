#!/usr/bin/env python
 
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import pickle
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer, build_lr_scheduler_distill
from detectron2.utils.events import EventStorage
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(cfg, model,model_teacher, resume=False):
    model.train()
    # optimizer = build_optimizer(cfg, model)
    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    #scheduler=instantiate(cfg.lr_multiplier)
    scheduler = build_lr_scheduler_distill(cfg, optimizer)
    

    checkpointer = DetectionCheckpointer(
        model, cfg.train.output_dir, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.train.max_iter

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.train.checkpointer.period, max_iter=max_iter
    )
    model = create_ddp_model(model, **cfg.train.ddp)
    writers = default_writers(cfg.train.output_dir, max_iter) if comm.is_main_process() else []
  
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    # data_loader = build_detection_train_loader(cfg)
    data_loader = instantiate(cfg.dataloader.train)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            teacher_loss_dict, tea_feature, pred_objectness_logits_teacher = model_teacher.forward_rpn(data) 
            tea_adapt_feature=[]
            for key,feature in tea_feature.items():
                  tea_adapt_feature.append(feature)
                  
            loss_dict, stu_adapt_feature, pred_objectness_logits_student = model(data,distill=True)
            
            ###distillation loss frs#############
            layers = len(stu_adapt_feature)
            distill_feat_loss, distill_cls_loss = 0, 0

            for layer in range(layers):
                stu_cls_score_sigmoid = pred_objectness_logits_student[layer].sigmoid()
                tea_cls_score_sigmoid = pred_objectness_logits_teacher[layer].sigmoid()
                mask = torch.max(tea_cls_score_sigmoid, dim=1).values
                mask = mask.detach()

                feat_loss = torch.pow((tea_adapt_feature[layer] - stu_adapt_feature[layer]), 2)
                cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')

                distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum()
                distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()
                 
            distill_feat_loss = distill_feat_loss *cfg.distill_feat_weight
            distill_cls_loss = distill_cls_loss * cfg.distill_cls_weight
            if cfg.distill_warm_step > iteration:
                distill_feat_loss = (iteration / cfg.distill_warm_step) * distill_feat_loss
                distill_cls_loss = (iteration / cfg.distill_warm_step) * distill_cls_loss
            loss_dict.update({"distill_feat_loss":distill_feat_loss})
            loss_dict.update({"distill_cls_loss":distill_cls_loss})
     
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.train.eval_period > 0
                and (iteration + 1) % cfg.train.eval_period == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # cfg = get_cfg()
    # cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()
    # default_setup(
    #     cfg, args
    # )  # if you don't like any of the default setup, write your own setup code
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = instantiate(cfg.model).cuda()
    # model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    model_teacher = instantiate(cfg.model_teacher).cuda()
    # model_teacher = build_model(cfg.model_teacher)
    for p in model_teacher.parameters():
        p.requires_grad = False
    logger.info("Model teacher:\n{}".format(model_teacher))

    if cfg.teacher_path:
        with open(cfg.teacher_path, 'rb') as f:
            obj=f.read()
        checkpoint=pickle.loads(obj, encoding='latin1')
        weights = {key: torch.Tensor(weight_dict) for key, weight_dict in checkpoint['model'].items()} 
        print('!!!!!!!!!!!!load teacher from',cfg.teacher_path)
        missing_keys, unexpected_keys = model_teacher.load_state_dict(weights,strict=False)
        print('missing_keys:', missing_keys)
        print('unexpected_keys', unexpected_keys)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.train.output_dir).resume_or_load(
           cfg.train.init_checkpoint, resume=args.resume
        )
        return do_test(cfg, model)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model,model_teacher, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    if args.dist_on_itp:
        import os
        args.machine_rank = int(os.environ["RANK"])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
