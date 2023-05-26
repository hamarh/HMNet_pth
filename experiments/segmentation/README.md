# Dataset Preparation

## [DSEC-Semantic](https://dsec.ifi.uzh.ch/dsec-semantic/)

Download events, images, and labels and place them in `./data/dsec/source/`

```bash
mkdir ./data/dsec/source/
cd ./data/dsec/source/
wget https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_events.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/train_coarse/train_images.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/semantic/train_semantic_segmentation.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_events.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/test_coarse/test_images.zip
wget https://download.ifi.uzh.ch/rpg/DSEC/semantic/test_semantic_segmentation.zip
```

Unzip the files and run the following script.

```bash
cd ./data/dsec/
bash ./scripts/prepair.sh
```

Download and unzip metadata.

```bash
wget https://github.com/hamarh/HMNet/releases/download/v1.0.0/dsec_meta.tar
tar -xvf dsec_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
bash ./scripts/prepair.sh -a
```

# Reproduce our results

To reproduce the results of HMNet-B3:

(1) Download pretrained weights.

```bash
wget https://github.com/hamarh/HMNet/releases/download/v1.0.0/dsec_hmnet_B3.pth
```

(2) Run inference with the following commands.

```bash
python ./scripts/test.py ./config/hmnet_B3.py ./data/dsec/list/test/ ./data/dsec/ --speed_test --fast --pretrained ./pretrained/dsec_hmnet_B3.pth
```

(3) Evaluate the results.

```bash
sh ./scripts/run_eval.sh ./config/hmnet_B3.py
```

# Training & Inference

## Step1. Training

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

## Step2. Inference using the trained model

```bash
python ./scripts/test.py ./config/hmnet_B3.py ./data/dsec/list/test/ ./data/dsec/ --fast --speed_test
```

# Training Details

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 16 | 31.0 | 0.2817 | github | github |
| hmnet_L1 | A100 (40GB) x 16 | 42.5 | 0.1881 | github | github |
| hmnet_B3 | A100 (40GB) x 16 | 46.0 | 0.1685 | github | github |
| hmnet_L3 | A100 (40GB) x 16 | 63.1 | 0.1410 | github | github |
