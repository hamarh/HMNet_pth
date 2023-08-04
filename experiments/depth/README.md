# Dataset Preparation

## [Eventscape](https://github.com/uzh-rpg/rpg_ramnet)

Download files and place them in `./data/eventscape/source/`

```bash
mkdir ./data/eventscape/source/
cd ./data/eventscape/source/
wget http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town01-03_train.zip
wget http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_val.zip
wget http://rpg.ifi.uzh.ch/data/RAM_Net/dataset/Town05_test.zip
unzip Town01-03_train.zip
unzip Town05_val.zip
unzip Town05_test.zip
```

Run the following script.

```bash
cd ./data/eventscape/
bash ./scripts/prepair.sh
```

Download and unzip metadata.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_meta.tar
tar -xvf eventscape_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
bash ./scripts/prepair.sh -a
```

## [MVSEC](https://daniilidis-group.github.io/mvsec/)

Download files at [this page](https://daniilidis-group.github.io/mvsec/download/) and place them in `./data/mvsec/source/`

```bash
./data/mvsec/source/
├── outdoor_day1_data.hdf5
├── outdoor_day1_gt.hdf5
├── outdoor_day2_data.hdf5
├── outdoor_day2_gt.hdf5
├── outdoor_night1_data.hdf5
└── outdoor_night1_gt.hdf5
```

Download metadata.

```bash
cd ./data/mvsec/source/
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_meta.tar
tar -xvf mvsec_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
cd ./data/mvsec/
bash ./scripts/prepair.sh
```

# Reproduce our results

To reproduce the results of HMNet-B3:

(1) Download pretrained weights.

```bash
wget https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3.pth
```
Put the weights in `./pretrained/`

(2) Run inference with the following commands.

```bash
# inference on outdoor day1
python ./scripts/test_mvsec.py ./config/hmnet_B3.py day1 --fast --speed_test --pretrained ./pretrained/hmnet_B3_mvsec.pth
# inference on outdoor night1
python ./scripts/test_mvsec.py ./config/hmnet_B3.py night1 --fast --speed_test --pretrained ./pretrained/hmnet_B3_mvsec.pth
```

(3) Evaluate the results.

```bash
sh ./scripts/run_eval_mvsec.sh ./config/hmnet_B3.py
```

# Training & Inference

## Step1. Pre-training on Eventscape

Single node training:

```bash
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed
```

Multi-node training:

```bash
# Run the following command at the first node.
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed --master ${master} --node 1/2
# Run the following command at the second node.
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed --master ${master} --node 2/2
```

`${master}`is the IP address of the first node.

## Step2. Fine-tuning on MVSEC outdoor day2

```bash
python ./scripts/train.py ./config/hmnet_B3.py --amp --distributed --finetune --overwrite
```

## Step3. Inference using the trained model

Eventscape dataset:

```bash
python ./scripts/test.py ./config/hmnet_B3.py ./data/eventscape/list/test/ ./data/eventscape/ --fast --speed_test
```

MVSEC outdoor day1/night1:

```bash
python ./scripts/test_mvsec.py ./config/hmnet_B3.py day1 --fast --speed_test
python ./scripts/test_mvsec.py ./config/hmnet_B3.py night1 --fast --speed_test
```

# Evaluation

```bash
# Eval on Eventscape
sh ./scripts/run_eval_eventscape.sh ./config/hmnet_B3.py
# Eval on MVSEC outdoor day1/night1
sh ./scripts/run_eval_mvsec.sh ./config/hmnet_B3.py
```

# Training Details

The pre-trained weights are released under the Creative Commons BY-SA 4.0 License.

Pre-training on Eventscape

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 8 | 11.3 | 0.0317 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_B1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_B1.csv) |
| hmnet_L1 | A100 (40GB) x 8 | 20.2 | 0.0293 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_L1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_L1.csv) |
| hmnet_B3 | A100 (40GB) x 8 | 20.6 | 0.0304 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_B3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_B3.csv) |
| hmnet_L3 | A100 (40GB) x 8 | 30.1 | 0.0294 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_L3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_L3.csv) |
| hmnet_B3_fuse_rgb | A100 (40GB) x 8 | 25.9 | 0.0277 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_B3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_B3_fuse_rgb.csv) |
| hmnet_L3_fuse_rgb | A100 (40GB) x 8 | 30.2 | 0.0273 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/eventscape_hmnet_L3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/eventscape_hmnet_L3_fuse_rgb.csv) |

Fine-tuning on MVSEC

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 8 | 1.8 | 0.0319 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_B1.csv) |
| hmnet_L1 | A100 (40GB) x 8 | 1.2 | 0.0296 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L1.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_L1.csv) |
| hmnet_B3 | A100 (40GB) x 8 | 1.7 | 0.0286 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_B3.csv) |
| hmnet_L3 | A100 (40GB) x 8 | 1.3 | 0.0279 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_L3.csv) |
| hmnet_B3_fuse_rgb | A100 (40GB) x 8 | 2.7 | 0.0270 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_B3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_B3_fuse_rgb.csv) |
| hmnet_L3_fuse_rgb | A100 (40GB) x 8 | 2.7 | 0.0264 | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.2.0/mvsec_hmnet_L3_fuse_rgb.pth) | [github](https://github.com/hamarh/HMNet_pth/releases/download/v0.1.0/mvsec_hmnet_L3_fuse_rgb.csv) |
