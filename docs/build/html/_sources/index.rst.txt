:github_url: https://github.com/zudi-lin/pytorch_connectomics

PyTorch Connectomics documentation
===================================

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ is a deep learning 
framework for automatic and semi-automatic annotation of connectomics datasets, powered by `PyTorch <https://pytorch.org/>`_.

.. note::
    This package is under development and should not be considered as formally released.

The field of connectomics aims to reconstruct the wiring diagram of the brain by mapping the neural 
connections at the level of individual synapses. Recent advances in electronic microscopy (EM) have enabled 
the collection of a large number of image stacks at nanometer resolution, but the annotation requires expertise 
and is super time-consuming. 

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ consists of various deep learning based object detection, 
semantic segmentation and instance segmentation methods for the annotation and analysis of 3D image stacks. In addition, it consists of an easy-to-use 
data augmentation interface, tutorials on several common benchmark datasets, and helpful image stack processing functions, both 
for reproducing state-of-the-art results on benchmark datasets, and labelling large-scale volumes.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/snemi
   tutorials/cremi
   tutorials/lucchi

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/model
   modules/augmentation
   modules/datasets
   modules/utils

Indices and Tables
====================

* :ref:`genindex`
* :ref:`modindex`
