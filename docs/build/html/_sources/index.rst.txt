:github_url: https://github.com/zudi-lin/pytorch_connectomics

PyTorch Connectomics documentation
===============================

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ is a deep learning framework for automatic and semi-automatic annotation of connectomics datasets, powered by `PyTorch <https://pytorch.org/>`_.

It consists of various methods for deep learning on graphs and other irregular structures, also known as `geometric deep learning <http://geometricdeeplearning.com/>`_, from a variety of published papers.
In addition, it consists of an easy-to-use mini-batch loader, a large number of common benchmark datasets (based on simple interfaces to create your own), and helpful transforms, both for learning on arbitrary graphs as well as on 3D meshes or point clouds.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/installation
   notes/introduction

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/snemi

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/model
   modules/augmentation
   modules/datasets
   modules/utils

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
