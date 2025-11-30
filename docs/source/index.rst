:github_url: https://github.com/zudi-lin/pytorch_connectomics

PyTorch Connectomics Documentation
===================================

.. note::
   **Version 2.0 is here!** PyTorch Connectomics has been completely rewritten with PyTorch Lightning orchestration
   and MONAI medical imaging tools. See the :ref:`installation guide <Installation>` for updated instructions.

`PyTorch Connectomics <https://github.com/zudi-lin/pytorch_connectomics>`_ is a deep learning
framework for automatic and semi-automatic annotation of connectomics datasets, powered by
`PyTorch <https://pytorch.org/>`_, `PyTorch Lightning <https://lightning.ai/>`_, and
`MONAI <https://monai.io/>`_. This repository is actively developed and maintained by the
**Visual Computing Group** (`VCG <https://vcg.seas.harvard.edu>`_) at Harvard University.

The field of connectomics aims to reconstruct the wiring diagram of the brain by mapping neuronal
connections at the level of individual synapses. Recent advances in electron microscopy (EM) have
enabled the collection of large-scale image stacks at nanometer resolution, but annotation requires
expertise and is extremely time-consuming, which restricts progress in downstream biological and
medical analysis.

What's New in v2.0
------------------

PyTorch Connectomics v2.0 brings major architectural improvements:

- ⚡ **PyTorch Lightning Integration**: Distributed training, mixed-precision, and automatic optimization
- 🏥 **MONAI Integration**: Medical imaging models, transforms, and losses
- 🔧 **Hydra Configuration**: Type-safe, composable configuration management
- 📦 **Architecture Registry**: Easy model management and extensibility
- 🔬 **MedNeXt Models**: State-of-the-art ConvNeXt-based models (MICCAI 2023)
- 🧩 **Deep Supervision**: Multi-scale training support
- 📊 **Enhanced Monitoring**: TensorBoard and Weights & Biases integration

Key Features
------------

**Modern Architecture**

- PyTorch Lightning for training orchestration
- MONAI for medical imaging domain expertise
- Clean separation of concerns: Lightning (shell) + MONAI (toolbox)

**Training & Optimization**

- Multi-task, active, and semi-supervised learning
- Distributed training (DDP) with automatic GPU parallelization
- Mixed-precision training (FP16/BF16) for speed and memory efficiency
- Gradient accumulation and checkpointing
- Advanced learning rate scheduling with warmup

**Models & Architectures**

- **MONAI Models**: BasicUNet3D, UNet, UNETR, Swin UNETR
- **MedNeXt**: ConvNeXt-based models (S/B/M/L variants)
- Custom architecture support through registry system
- Deep supervision for multi-scale loss computation

**Data Processing**

- Support for HDF5, TIFF, Zarr formats
- MONAI-based augmentations for volumetric data
- Efficient caching and preprocessing
- Multi-scale and multi-task label handling

**Monitoring & Logging**

- TensorBoard integration (default)
- Weights & Biases support
- Early stopping and model checkpointing
- Rich metrics tracking with TorchMetrics

Quick Start
-----------

**Installation:**

.. code-block:: bash

    # Install PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Install PyTorch Connectomics
    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install -e .[full]

**Train a model:**

.. code-block:: bash

    python scripts/main.py --config tutorials/lucchi.yaml

**Python API:**

.. code-block:: python

    from connectomics.config import load_config
    from connectomics.training.lit import ConnectomicsModule, ConnectomicsDataModule, create_trainer

    # Load configuration
    cfg = load_config("tutorials/lucchi.yaml")

    # Create components
    datamodule = ConnectomicsDataModule(cfg)
    model = ConnectomicsModule(cfg)
    trainer = create_trainer(cfg)

    # Train
    trainer.fit(model, datamodule=datamodule)

See the :ref:`installation guide <Installation>` and :ref:`tutorials <Tutorials>` for more details.

Documentation
-------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Get Started

   notes/installation
   notes/config
   notes/dataloading
   notes/migration
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
   :caption: External Tools

   external/neuroglancer

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/lightning
   modules/model
   modules/data
   modules/utils
   modules/engine

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: About

   about/team

Community & Support
-------------------

- 💬 **Slack**: `Join our community <https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w>`_
- 📧 **GitHub**: `Issues and discussions <https://github.com/zudi-lin/pytorch_connectomics/issues>`_
- 📚 **Documentation**: `https://connectomics.readthedocs.io <https://connectomics.readthedocs.io>`_
- 📄 **Paper**: `arXiv:2112.05754 <https://arxiv.org/abs/2112.05754>`_

Migration from v1.0
-------------------

If you're upgrading from v1.0, key changes include:

- **Configuration**: Hydra/OmegaConf configuration system
- **Training**: PyTorch Lightning for orchestration
- **Models**: MONAI native models with registry system
- **Entry point**: ``scripts/main.py``

v2.0 uses:

- Hydra/OmegaConf configs (``tutorials/*.yaml``)
- Lightning modules (``connectomics/lightning/``)
- ``scripts/main.py`` entry point

Citation
--------

If you find PyTorch Connectomics useful in your research, please cite:

.. code-block:: bibtex

    @article{lin2021pytorch,
      title={PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics},
      author={Lin, Zudi and Wei, Donglai and Lichtman, Jeff and Pfister, Hanspeter},
      journal={arXiv preprint arXiv:2112.05754},
      year={2021}
    }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
