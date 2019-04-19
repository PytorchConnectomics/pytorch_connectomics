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

Create new conda environment:
```
conda create -n py3_torch python=3.6
source activate activate py3_torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Download and install the package:
```
git clone git@github.com:zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install -r requirements.txt
pip install --editable .
```
For more information and frequently asked questions about installation, please check the [installation guide](). If you meet compilation errors, please check [TROUBLESHOOTING.md](https://github.com/zudi-lin/pytorch_connectomics/blob/master/TROUBLESHOOTING.md).

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
Previous works have suggested that a reasonable large batch size can improve the performance of detection and segmentation models. Here we use a syncronized batch normalization module that computes the mean and standard-deviation across all devices during training. Please refer to [Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) for details. The implementation is pure-python, and uses unbiased variance to update the moving average, and use `sqrt(max(var, eps))` instead of `sqrt(var + eps)`.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE) file for details.

## Contact
[Zudi Lin](https://github.com/zudi-lin)
