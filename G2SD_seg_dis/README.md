## Getting started 
1. install pytorch
   ```bash
   pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111  -f https://download.pytorch.org/whl/cu111/torch_stable.html
   ```
3. Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages.

```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.1/index.html
pip install mmsegmentation==0.11.0
pip3 install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy timm==0.3.2
```

3. Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

4. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md) to prepare the ADE20k dataset.
