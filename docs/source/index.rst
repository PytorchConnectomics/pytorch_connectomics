:github_url: https://github.com/zudi-lin/pytorch_connectomics

PyTorch Connectomics documentation
===================================

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ is a deep learning 
framework for automatic and semi-automatic annotation of connectomics datasets, powered by `PyTorch <https://pytorch.org/>`_. 
This repository is actively under development by **Visual Computing Group** (`VCG <https://vcg.seas.harvard.edu>`_) at Harvard University.

The field of connectomics aims to reconstruct the wiring diagram of the brain by mapping the neuronal connections at the level of individual synapses. 
Recent advances in electronic microscopy (EM) have enabled the collection of a large number of image stacks at nanometer resolution, but the annotation 
requires expertise and is super time-consuming, which restricts the progress in downstream biological/medical analysis. 

Our `PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ implements various deep learning-based object detection, 
semantic segmentation, and instance segmentation approaches for the annotation and analysis of 3D image stacks. In addition, it provides 
an easy-to-use data augmentation interface, detailed tutorials on common benchmark datasets, and handy functions for processing image stacks. 
This package can not only reproduce state-of-the-art performance on benchmark datasets but also improve the annotation efficiency of large-scale volumes.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   notes/installation
   notes/config
   notes/dataloading
   notes/faq

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/neuron
   tutorials/mito
   tutorials/synapse
   tutorials/artifact

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: External
   
   external/neuroglancer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/data
   modules/engine
   modules/model
   modules/utils

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Team

   about/team

Indices and Tables
====================

* :ref:`genindex`
* :ref:`modindex`
