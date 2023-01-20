# Generative Attention Model-Based Feature Semantics for Temporal Action Detection

This is an official implementation in PyTorch of GAF. 

## Abstract

Temporal action detection is an important yet challenging task in video analysis, aiming at inferring both the action category and localization of the start and end frame for each action instance in the untrimmed video. However, current methods ignore intra and inter relations of feature semantics, and suffer from deviated action boundaries and inaccurate detection, which greatly limits the utility of the detection task. In this paper, we propose a novel generative attention mechanism to simultaneously model the intra and inter dependencies of temporal action feature semantics by using the differences of actions’ foreground and background. Extensive experiments show that, compared with the state-of-the-art, our method achieves better performance on THUMOS14, and comparable performance on ActivityNet v1.3. Particularly, for complex background and small objective action detection tasks, our method achieves around 3.6% mAP improvement on THUMOS14.

## Summary

- Propose a novel generative attention mechanism to simultaneously model the intra and inter dependencies of temporal action feature representation.
- Leverage inter and intra relations of the feature dimension to locate the areas and boundaries of action instances.
- For complex background and small objective action detection tasks, our method get better performance.

## Getting Started

### Environment

- Python 3.7
- PyTorch == 1.4.0
- NVIDIA GPU

### Setup

```shell script
pip3 install -r requirements.txt
python3 setup.py develop
```

### Data Preparation

- **THUMOS14 RGB data:**

1. Download pre-processed RGB npy data (13.7GB): [Baiduyun](https://pan.baidu.com/s/1MRm6F9cgOv4MSlNajwaI4g ), password：xot6
2. Unzip the RGB npy data to `./datasets/thumos14/validation_npy/` and `./datasets/thumos14/test_npy/`

- **THUMOS14 flow data:**

1. Because it costs more time to generate flow data for THUMOS14, to make easy to run flow model, we provide the pre-processed flow data in Baiduyun(3.4GB):
   [Baiduyun](https://pan.baidu.com/s/1_Zm_FQRnTtTkXEAkCQgnAg ), password：7rpw
1. Unzip the flow npy data to `./datasets/thumos14/validation_flow_npy/` and `./datasets/thumos14/test_flow_npy/`



### Inference

We provide the pretrained models contain RGB and flow models for THUMOS14 dataset:[Google_Drive](https://drive.google.com/drive/folders/10RO2OrTm3p-ATiSnOyhYPRAc80y_4UMS?usp=sharing)

```shell script
# run RGB model
python3 GAF/thumos14/test_stu.py configs/thumos14.yaml --checkpoint_path=models/thumos14/student/checkpoint-16.ckpt --output_json=thumos14_rgb_16.json

# run flow model
python3 GAF/thumos14/test.py configs/thumos14_flow.yaml --checkpoint_path=models/thumos14_flow/student/checkpoint-16.ckpt --output_json=thumos14_flow.json

# run fusion (RGB + flow) model
python3 GAF/thumos14/test.py configs/thumos14.yaml --fusion --output_json=thumos14_fusion.json
```

### Evaluation

The output json results of pretrained model can be downloaded from:[Google_Drive](https://drive.google.com/file/d/1pmQjIT57OlJLLJZh0a7GExhr3V3HKvtf/view?usp=sharing)

```shell script
# evaluate THUMOS14 fusion result as example
python3 GAF/thumos14/eval.py output/thumos14_fusion.json

python3 GAF/thumos14/eval.py output/thumos14_rgb_41.json
```

### Training

```shell script
# train the RGB model
python3 GAF/thumos14/train_KD.py configs/thumos14.yaml --lw=10 --cw=1 --piou=0.5 --resume=60

# train the flow model
python3 GAF/thumos14/train.py configs/thumos14_flow.yaml --lw=10 --cw=1 --piou=0.5
```
