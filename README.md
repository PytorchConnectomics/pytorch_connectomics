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

The field of *connectomics* aims to reconstruct the wiring diagram of the brain by mapping the neural connections at the level of individual synapses. Recent advances in electronic microscopy (EM) have enabled the collection of a large number of image stacks at nanometer resolution, but annotation requires expertise and is super time-consuming. Here we provide a deep learning framework powered by [PyTorch](https://pytorch.org/) for automatic and semi-automatic semantic and instance segmentation in connectomics, which we call **PyTorch Connectomics** (PyTC). This repository is mainly maintained by the Visual Computing Group ([VCG](https://vcg.seas.harvard.edu)) at Harvard University.

*PyTorch Connectomics is currently under active development!*

## Key Features

- Multi-task, active and semi-supervised learning
- Distributed and mixed-precision optimization
- Scalability for handling large datasets
- Comprehensive augmentations for volumetric data

## Installation

Refer to the [Pytorch Connectomics wiki](https://connectomics.readthedocs.io), specifically the [installation page](https://connectomics.readthedocs.io/en/latest/notes/installation.html), for the most up-to-date instructions on installation on a local machine or high-performance cluster.

### Docker

Besides the installation guidance above, we also push a PyTC Docker image to the public docker 
registry (03/12/2022) to improve usability.
Additionally, we provide the corresponding Dockerfile to enable individual modifications.
Pleas refer to our [PyTC Docker Guidance](docker/README.md) for more information.

## Notes

### Segmentation Models

We provide several encoder-decoder architectures, which are customized 3D UNet and Feature Pyramid Network (FPN) models with various blocks and backbones. Those models can be applied for both semantic segmentation and bottom-up instance segmentation of 3D image stacks. Those models can also be constructed specifically for isotropic and anisotropic datasets. Please check the [documentation](http://connectomics.readthedocs.io/) for more details.

### Data Augmentation

We provide a data augmentation interface for several common augmentation methods for EM images. The interface operates on NumPy arrays, so it can be easily incorporated alongside many Python-based deep learning framework (e.g. TensorFlow). For more details about the design of the data augmentation module, please check the [documentation](http://connectomics.readthedocs.io/), specifically the [```utils``` documentation](https://connectomics.readthedocs.io/en/latest/modules/utils.html).

### YACS Configuration

We use the *Yet Another Configuration System* ([YACS](https://github.com/rbgirshick/yacs)) library to manage settings and hyperparameters in model training and inference. The configuration files for tutorial examples can be found [here](https://github.com/zudi-lin/pytorch_connectomics/tree/master/configs). All available configuration options can be found at [```connectomics/config/defaults.py```](https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/config/defaults.py). Please note that the default value of several options is ```None```, which is only supported after YACS v0.1.8.

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
