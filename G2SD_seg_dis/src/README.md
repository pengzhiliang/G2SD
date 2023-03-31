# ADE20k Semantic segmentation with G2SD

## Getting started 

1. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

2. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

3. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md) to prepare the ADE20k dataset.


## Fine-tuning

Command format:
```
bash src/tools/dist_distill.sh <CONFIG_PATH> <NUM_GPUS>  --work-dir <SAVE_PATH> --seed 0  --deterministic --options distiller.teacher_pretrained=<IMAGENET_CHECKPOINT_PATH/URL>
distiller.student_pretrained=<IMAGENET_CHECKPOINT_PATH>
```

For example, using a ViT-Small backbone with UperNet:
```bash
bash src/tools/dist_distill.sh \
    src/configs/distill/distill_cwd.py 1 \
    --work-dir output --seed 0  --deterministic \
    --options distiller.teacher_pretrained=teacher.pth \
              weight=3. \
              tau=1 \
              distiller.student_pretrained=student.pth \
```

More config files can be found at [`configs/beit/upernet`](configs/beit/upernet).


## Evaluation

Command format:
```
tools/dist_test.sh  <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU
```







