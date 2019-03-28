# Pytorch Connectomics: A Deep Learning Toolbox for EM Connectomics Research

## Introduction

The field of connectomics aims to reconstruct the wiring diagram of the brain by mapping the neural connections at the level of individual synapses.

## Key Features

- Multitask Learning
- Active Learning
- Transfer Learning

## Environment

The code is developed and tested under the following configurations.
- Hardware: 1-8 Nvidia GPUs (with at least 12G GPU memories) (change ```[--num-gpu GPUS]``` accordingly)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=9.0, Python>=3.5, PyTorch>=1.0.0***

## Installation
```
conda create -n py3_torch python=3.6
source activate activate py3_torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install -r requirements.txt
```

## Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE) file for details.

## Contact
Zudi Lin
Donglai Wei
