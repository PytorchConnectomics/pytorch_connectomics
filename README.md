# PyTorch Connectomics

## Introduction

The field of connectomics aims to reconstruct the wiring diagram of the brain by mapping the neural connections at the level of individual synapses. Here we provide a deep learning toolbox for automatic and semi-automatic data annotation in connectomics.

## Key Features

- Multitask Learning
- Active Learning

## Environment

The code is developed and tested under the following configurations.
- Hardware: 1-8 Nvidia GPUs (with at least 12G GPU memories) (change ```[--num-gpu GPUS]``` accordingly)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=9.0, Python>=3.5, PyTorch>=1.0.0***

## Installation
```
conda create -n py3_torch python=3.6
source activate activate py3_torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

git clone git@github.com:zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install -r requirements.txt
pip install --editable .
```
For more information and frequently asked questions about installation, please check the [installation guide]().

## Visulazation

### Training
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

### Test
* Visualize the affinity graph and segmentation using Neuroglancer.

## Notes

### Data Augmentation
We provide a data augmentation interface several different kinds of commonly used augmentation method for EM images. The interface is pure-python, and operate on and output only numpy arrays, so it can be easily incorporated into any kinds of python-based deep learning frameworks (e.g. TensorFlow). For more details about the design of the data augmentation module, please check the [documentation]().

### Model Zoo
We provide several encoder-decoder architectures.

### Syncronized Batch Normalization on PyTorch
This module computes the mean and standard-deviation across all devices during training. We empirically find that a reasonable large batch size is important for segmentation. We thank [Jiayuan Mao](http://vccy.xyz/) for his kind contributions, please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details.

The implementation is easy to use as:
- It is pure-python, no C++ extra extension libs.
- It is completely compatible with PyTorch's implementation. Specifically, it uses unbiased variance to update the moving average, and use sqrt(max(var, eps)) instead of sqrt(var + eps).
- It is efficient, only 20% to 30% slower than UnsyncBN.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE) file for details.

## Contact
[Zudi Lin](linzudi@g.harvard.edu)
