# Hierarchical Neural Memory Network

This repo is a PyTorch implementation of HMNet proposed in our paper: [Hierarchical Neural Network for Low Latency Event Processing](https://hamarh.github.io/hmnet/).

## Results and models

The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

**DSEC-Semantic (Semantic Segmentation)**

| Model | size | mIoU [%] | latency V100 [ms] | latency V100 x 3 [ms] | weights |
| --- | --- | --- | --- | --- | --- |
| HMNet-B1 | 640 x 440 | 51.2 | 7.0  | -    | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B1.pth) |
| HMNet-L1 | 640 x 440 | 55.0 | 10.5 | -    | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_L1.pth) |
| HMNet-B3 | 640 x 440 | 53.9 | 9.7  | 8.0  | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_B3.pth) |
| HMNet-L3 | 640 x 440 | 57.1 | 13.9 | 11.9 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/dsec_hmnet_L3.pth) |

**GEN1 (Object Detection)**

| Model | size | mAP [%] | latency V100 [ms] | latency V100 x 3 [ms] | weights |
| --- | --- | --- | --- | --- | --- |
| HMNet-B1 | 304 x 240 | 45.5 | 4.6 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B1_tbptt.pth) |
| HMNet-L1 | 304 x 240 | 47.0 | 5.6 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L1_tbptt.pth) |
| HMNet-B3 | 304 x 240 | 45.2 | 7.0 | 5.9 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_B3_tbptt.pth) |
| HMNet-L3 | 304 x 240 | 47.1 | 7.9 | 7.0 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/gen1_hmnet_L3_tbptt.pth) |

**MVSEC day1 (Monocular Depth Estimation)**

| Model | size | AbsRel | RMS | RMSElog | latency V100 [ms] | latency V100 x 3 [ms] | weights |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HMNet-B1 | 346 x 260 | 0.385 | 9.088 | 0.509 | 2.4 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B1.pth) |
| HMNet-L1 | 346 x 260 | 0.310 | 8.383 | 0.393 | 4.1 | -   | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L1.pth) |
| HMNet-B3 | 346 x 260 | 0.270 | 7.101 | 0.332 | 5.0 | 4.1 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3.pth) |
| HMNet-L3 | 346 x 260 | 0.254 | 6.890 | 0.319 | 6.9 | 5.4 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3.pth) |
| HMNet-B3 w/ RGB | 346 x 260 | 0.252 | 6.972 | 0.318 | 5.4 | 4.1 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3_fuse_rgb.pth) |
| HMNet-L3 w/ RGB | 346 x 260 | 0.230 | 6.922 | 0.310 | 7.1 | 5.4 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3_fuse_rgb.pth) |

# Requirements

- PyTorch >= 1.12.1
- torch_scatter
- timm
- hdf5plugin

# Installation

Create a new conda environment

```bash
conda create -n hmnet python=3.7
conda activate hmnet
```

Install dependencies

```bash
pip install -r requirements.txt
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install timm
```

# Experiments

Please see the instructions on each task page.

- [Semantic Segmentation](https://github.com/hamarh/HMNet_pth/blob/main/experiments/segmentation/)
- [Object Detection](https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/)
- [Monocular Depth Estimation](https://github.com/hamarh/HMNet_pth/blob/main/experiments/depth/)

# License

The majority of this project is licensed under BSD 3-clause License. However, some code ([psee_evaluator.py](https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/scripts/psee_evaluator.py), [coco_eval.py](https://github.com/hamarh/HMNet_pth/blob/main/experiments/detection/scripts/coco_eval.py), [det_head_yolox.py](https://github.com/hamarh/HMNet_pth/blob/main/hmnet/models/base/head/task_head/det_head_yolox.py)) is available under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) license.
The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

# Acknowledgments

This work is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
