# Dataset Preparation

## [GEN1](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/)

Clone and copy toolbox for gen1 dataset.

```bash
git clone https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox.git
mv prophesee-automotive-dataset-toolbox/src ${HMNet}/hmnet/utils/psee_toolbox
rm -rf ./prophesee-automotive-dataset-toolbox
```

Download events and bbox annotations at [dataset page](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/) and place the files in `./data/gen1/source/`

```bash
./data/gen1/source/
├── test_a.7z
├── test_b.7z
├── train_a.7z
├── train_b.7z
├── train_c.7z
├── train_d.7z
├── train_e.7z
├── train_f.7z
├── val_a.7z
└── val_b.7z
```

Unzip the files and run the following script.

```bash
cd ./data/gen1/
bash ./scripts/prepair.sh
```

Download and unzip metadata.

```bash
wget https://github.com/hamarh/HMNet/releases/download/v1.0.0/gen1_meta.tar
tar -xvf gen1_meta.tar
```

If you want to generate metadata from scratch, run the following command:

```bash
bash ./scripts/prepair.sh -a
```

# Reproduce our results

To reproduce the results of HMNet-B3:

(1) Download pretrained weights.

```bash
wget https://github.com/hamarh/HMNet/releases/download/v1.0.0/gen1_hmnet_B3_tbptt.pth
```

(2) Run inference with the following commands.

```bash
python ./scripts/test.py ./config/hmnet_B3_yolox_tbptt.py ./data/gen1/list/test/ ./data/gen1/ --pretrained ./pretrained/gen1_hmnet_B3_tbptt.pth --fast --speed_test
```

(3) Evaluate the results.

```bash
sh ./scripts/run_eval.sh ./config/hmnet_B3_yolox_tbptt.py
```

# Training & Inference

## Step1. Training with short sequences

Single node training:

```bash
python ./scripts/train.py ./config/hmnet_B3_yolox.py --amp --distributed
```

Multi-node training:

```bash
# Run the following command at the first node.
python ./scripts/train.py ./config/hmnet_B3_yolox.py --amp --distributed --master ${master} --node 1/2
# Run the following command at the second node.
python ./scripts/train.py ./config/hmnet_B3_yolox.py --amp --distributed --master ${master} --node 2/2
```

`${master}`is the IP address of the first node.

## Step2. Training with long sequences (TBPTT)

```bash
python ./scripts/train.py ./config/hmnet_B3_yolox_tbptt.py --amp --distributed
```

## Step3. Inference using the trained model

```bash
python ./scripts/test.py ./config/hmnet_B3_yolox_tbptt.py ./data/gen1/list/test/ ./data/gen1/ --fast --speed_test
```

# Training Details

|  | GPU | Training Time [hr] | Loss | Weights | Log |
| --- | --- | --- | --- | --- | --- |
| hmnet_B1 | A100 (40GB) x 8 | 28.8 | 2.3797 | github | github |
| hmnet_B1_tbptt | A100 (40GB) x 8 | 12.3 | 2.8653 | github | github |
| hmnet_L1 | A100 (40GB) x 8 | 42.0 | 2.0687 | github | github |
| hmnet_L1_tbptt | A100 (40GB) x 16 | 10.5 | 2.8095 | github | github |
| hmnet_B3 | A100 (40GB) x 8 | 47.9 | 2.1555 | github | github |
| hmnet_B3_tbptt | A100 (40GB) x 16 | 18.7 | 2.5021 | github | github |
| hmnet_L3 | A100 (40GB) x 8 | 64.5 | 1.9462 | github | github |
| hmnet_L3_tbptt | A100 (40GB) x 16 | 19.0 | 2.4505 | github | github |
