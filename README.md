<a href="https://github.com/zudi-lin/pytorch_connectomics">
<img src="./.github/logo_fullname.png" width="450"></a>

<p align="left">
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8-ff69b4.svg" /></a>
    <a href= "https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.8-2BAF2B.svg" /></a>
    <a href= "https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
    <a href= "https://zudi-lin.github.io/pytorch_connectomics/build/html/index.html">
      <img src="https://img.shields.io/badge/Doc-Latest-2BAF2B.svg" /></a>
    <a href= "https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w">
      <img src="https://img.shields.io/badge/Slack-Join-CC8899.svg" /></a>
    <a href= "https://arxiv.org/abs/2112.05754">
      <img src="https://img.shields.io/badge/arXiv-2112.05754-FF7F50.svg" /></a>
</p>

<hr/>

## Introduction

The field of *connectomics* aims to reconstruct the wiring diagram of the brain by mapping the neural connections at the level of individual synapses. Recent advances in electronic microscopy (EM) have enabled the collection of a large number of image stacks at nanometer resolution, but the annotation requires expertise and is super time-consuming. Here we provide a deep learning framework powered by [PyTorch](https://pytorch.org/) for automatic and semi-automatic semantic and instance segmentation in connectomics, which is called **PyTorch Connectomics** (PyTC). This repository is mainly maintained by the Visual Computing Group ([VCG](https://vcg.seas.harvard.edu)) at Harvard University.

*PyTorch Connectomics is currently under active development!*

## Key Features

- Multi-task, active and semi-supervised learning
- Distributed and mixed-precision optimization
- Scalability for handling large datasets
- Comprehensive augmentations for volumetric data

If you want new features that are relatively easy to implement (*e.g.*, loss functions, models), please open a feature requirement discussion in issues or implement by yourself and submit a pull request. For other features that requires substantial amount of design and coding, please contact the [author](https://github.com/zudi-lin) directly.

## Environment

The code is developed and tested under the following configurations.

- Hardware: 1-8 Nvidia GPUs with at least 12G GPU memory (change ```SYSTEM.NUM_GPU``` accordingly based on the configuration of your machine)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=11.1, Python>=3.8, PyTorch>=1.9.0, YACS>=0.1.8***

## Installation

Create a new conda environment and install PyTorch:

```shell
conda create -n py3_torch python=3.8
source activate py3_torch
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
```

Please note that this package is mainly developed on the Harvard [FASRC](https://www.rc.fas.harvard.edu) cluster. More information about GPU computing on the FASRC cluster can be found [here](https://www.rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-the-cluster/).

### Potential Issues
After following the above steps to compile PyTorch, there is a possibility that it might not have been compiled with CUDA enabled. In such cases, it is advisable to perform a quick test before running any code. You can do this by opening a Python prompt and executing the following lines of code
```python
import torch
a = torch.randn(3,3)
a.to("cuda")
```
If there is an issue with the installation of PyTorch being compiled with CUDA, you are likely to encounter the following problem:
```
AssertionError: Torch not compiled with CUDA enabled
```
This can be fixed by installing the correct PyTorch version (corresponding to your CUDA drivers) using the wheels available on their [website](https://pytorch.org/get-started/previous-versions/). E.g., if your system has CUDA=11.8, the executing the following code in the environment should solve the issue:
```shell
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```
Once again, you can verify a successful installation of PyTorch by running the test mentioned earlier.

Download and install the package:

```shell
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install --editable .
```

Since the codebase is under active development, the **editable** installation will allow any changes to the original package to reflect directly in the environment. For more information and frequently asked questions about installation, please check the [installation guide](https://connectomics.readthedocs.io/en/latest/notes/installation.html).

### Docker

Besides the installation guidance above, we also push a PyTC Docker image to the public docker 
registry (03/12/2022) to improve usability.
Additionally, we provide the corresponding Dockerfile to enable individual modifications.
Pleas refer to our [PyTC Docker Guidance](docker/README.md) for more information.

## Notes

### Data Augmentation

We provide a data augmentation interface several different kinds of commonly used augmentation method for EM images. The interface is pure-python, and operate on and output only numpy arrays, so it can be easily incorporated into any kinds of python-based deep learning frameworks (e.g., TensorFlow). For more details about the design of the data augmentation module, please check the [documentation](http://connectomics.readthedocs.io/).

### YACS Configuration

We use the *Yet Another Configuration System* ([YACS](https://github.com/rbgirshick/yacs)) library to manage the settings and hyperparameters in model training and inference. The configuration files for tutorial examples can be found [here](https://github.com/zudi-lin/pytorch_connectomics/tree/master/configs). All available configuration options can be found at [```connectomics/config/defaults.py```](https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/config/defaults.py). Please note that the default value of several options is ```None```, which is only supported after YACS v0.1.8.

### Segmentation Models

We provide several encoder-decoder architectures, which are customized 3D UNet and Feature Pyramid Network (FPN) models with various blocks and backbones. Those models can be applied for both semantic segmentation and bottom-up instance segmentation of 3D image stacks. Those models can also be constructed specifically for isotropic
and anisotropic datasets. Please check the [documentation](http://connectomics.readthedocs.io/) for more details.

## Acknowledgement

This project is built upon numerous previous projects. Especially, we'd like to thank the contributors of the following github repositories:

- [pyGreenTea](https://github.com/naibaf7/PyGreentea): HHMI Janelia FlyEM Team
- [DataProvider](https://github.com/torms3/DataProvider): Princeton SeungLab
- [Detectron2](https://github.com/facebookresearch/detectron2): Facebook AI Reserach

We gratefully acknowledge the support from NSF awards IIS-1835231 and IIS-2124179.

## License

This project is licensed under the MIT License and the copyright belongs to all PyTorch Connectomics contributors - see the [LICENSE](https://github.com/zudi-lin/pytorch_connectomics/blob/master/LICENSE) file for details.

## Citation

For a detailed description of our framework, please read this [technical report](https://arxiv.org/abs/2112.05754). If you find PyTorch Connectomics (PyTC) useful in your research, please cite:

```bibtex
@article{lin2021pytorch,
  title={PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics},
  author={Lin, Zudi and Wei, Donglai and Lichtman, Jeff and Pfister, Hanspeter},
  journal={arXiv preprint arXiv:2112.05754},
  year={2021}
}
```
