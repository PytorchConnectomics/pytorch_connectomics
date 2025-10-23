Installation
==============

.. note::
   **PyTorch Connectomics v2.0** features a complete rewrite with PyTorch Lightning and MONAI integration.
   Please follow the updated installation instructions below.

The PyTorch Connectomics package is primarily developed on Linux machines with NVIDIA GPUs. We recommend following
the `Linux Installation <installation.html#id1>`_ guide to ensure compatibility with the latest features.
For Windows users, please check the `Windows Installation <installation.html#id2>`_ section.

Linux Installation
---------------------

.. tip::
   We do not recommend installation as root user on your system Python. Please set up an
   `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment
   and add the required packages to the environment.

Prerequisites
^^^^^^^^^^^^^

- **Python**: 3.8 or higher (3.10 recommended)
- **CUDA**: 10.2 or higher (11.8+ recommended for PyTorch 2.0+)
- **GPU**: NVIDIA GPU with compute capability 3.5+

Quick Installation
^^^^^^^^^^^^^^^^^^

For users who want to get started quickly:

.. code-block:: bash

    # Create conda environment
    conda create -n pytc python=3.10
    conda activate pytc

    # Install PyTorch (adjust for your CUDA version)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Clone and install PyTorch Connectomics
    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install -e .[full]

Detailed Installation Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1 - Create and Activate Virtual Environment
""""""""""""""""""""""""""""""""""""""""""""

.. code-block:: bash

    conda create -n pytc python=3.10
    conda activate pytc

2 - Install PyTorch
"""""""""""""""""""

Visit `PyTorch Get Started <https://pytorch.org/get-started/locally/>`_ to find the correct installation
command for your system and CUDA version.

**For CUDA 11.8:**

.. code-block:: bash

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

**For CUDA 12.1:**

.. code-block:: bash

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

**For CPU only (not recommended):**

.. code-block:: bash

    pip install torch torchvision

Verify PyTorch installation:

.. code-block:: python

    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Expected output:

.. code-block:: console

    PyTorch: 2.0.0+cu118
    CUDA available: True

3 - Install PyTorch Connectomics
"""""""""""""""""""""""""""""""""

**Option A: Full Installation (Recommended)**

Includes all optional features (Weights & Biases, TIFF support, hyperparameter optimization, etc.):

.. code-block:: bash

    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install -e .[full]

**Option B: Basic Installation**

Core dependencies only:

.. code-block:: bash

    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install -e .

**Option C: Custom Installation**

Install specific feature sets:

.. code-block:: bash

    # With Weights & Biases tracking
    pip install -e .[wandb]

    # With hyperparameter optimization (Optuna)
    pip install -e .[optim]

    # With TIFF file support
    pip install -e .[tiff]

    # With 3D visualization (Neuroglancer)
    pip install -e .[viz]

    # Multiple features
    pip install -e .[full,dev,docs]

**Option D: Direct Install from GitHub**

If you only want the library without cloning the repository:

.. code-block:: bash

    pip install git+https://github.com/zudi-lin/pytorch_connectomics.git

.. note::
   We use editable mode (``-e``) by default so there's no need to re-install when making changes to the code.

4 - Install MedNeXt (Optional)
"""""""""""""""""""""""""""""""

For state-of-the-art MedNeXt models (MICCAI 2023):

.. code-block:: bash

    git clone https://github.com/MIC-DKFZ/MedNeXt.git
    cd MedNeXt
    pip install -e .

.. tip::
   MedNeXt is optional. PyTorch Connectomics will work without it, but you won't be able to use
   MedNeXt architectures.

5 - Verify Installation
"""""""""""""""""""""""

Check that everything is installed correctly:

.. code-block:: bash

    # Check version
    python -c "import connectomics; print(f'PyTC Version: {connectomics.__version__}')"

    # List available models
    python -c "from connectomics.models.architectures import list_architectures; print('Available models:', list_architectures())"

Expected output:

.. code-block:: console

    PyTC Version: 2.0.0
    Available models: ['monai_basic_unet3d', 'monai_unet', 'monai_unetr', 'monai_swin_unetr', 'mednext', 'mednext_custom']

6 - Verify CUDA Setup (Optional)
"""""""""""""""""""""""""""""""""

Check that PyTorch is properly configured with CUDA:

.. code-block:: python

    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

Add CUDA to your environment (if needed):

.. code-block:: bash

    export PATH=/usr/local/cuda/bin:$PATH
    export CPATH=/usr/local/cuda/include:$CPATH

Verify ``nvcc`` is accessible:

.. code-block:: bash

    nvcc --version

Dependencies
^^^^^^^^^^^^

PyTorch Connectomics v2.0 has the following core dependencies:

**Core Frameworks:**

- PyTorch (>=1.8.0)
- PyTorch Lightning (>=2.0.0) - Training orchestration
- MONAI (>=0.9.1) - Medical imaging toolkit
- OmegaConf (>=2.1.0) - Configuration management

**Scientific Computing:**

- NumPy, SciPy, scikit-learn, scikit-image

**Data I/O:**

- h5py (HDF5), imageio, OpenCV

**Utilities:**

- TensorBoard (logging), tqdm (progress bars), einops, psutil

**Post-processing:**

- cc3d (connected components)

All core dependencies are automatically installed with ``pip install -e .``

**Optional Dependencies:**

Install via ``pip install -e .[extra_name]``:

- ``[full]``: All recommended features (wandb, tifffile, jupyter)
- ``[wandb]``: Weights & Biases experiment tracking
- ``[optim]``: Hyperparameter optimization (Optuna)
- ``[tiff]``: TIFF file support (tifffile)
- ``[viz]``: 3D visualization (Neuroglancer)
- ``[dev]``: Development tools (pytest)
- ``[docs]``: Documentation building (Sphinx)

See `setup.py <https://github.com/zudi-lin/pytorch_connectomics/blob/master/setup.py>`_ for complete list.

Cluster Installation (FASRC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For Harvard FASRC cluster users:

.. code-block:: bash

    # Load required modules
    module load cuda cudnn
    module load Anaconda3/2023.09-0

    # Create environment
    conda create -n pytc python=3.10
    conda activate pytc

    # Install PyTorch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # Install PyTorch Connectomics
    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install -e .[full]

See `FASRC documentation <https://www.rc.fas.harvard.edu>`_ for more details on module loading.

Windows Installation
--------------------

.. warning::
   Windows support is experimental. We recommend using Linux or WSL2 (Windows Subsystem for Linux)
   for production use.

Using WSL2 (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^^

1. Install WSL2 and Ubuntu: `WSL Installation Guide <https://docs.microsoft.com/en-us/windows/wsl/install>`_
2. Follow the Linux installation instructions above inside WSL2

Native Windows Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Install `Anaconda for Windows <https://www.anaconda.com/download>`_

2. Open Anaconda Prompt and create environment:

.. code-block:: bat

    conda create -n pytc python=3.10
    conda activate pytc

3. Install PyTorch:

.. code-block:: bat

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

4. Install PyTorch Connectomics:

.. code-block:: bat

    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install -e .[full]

Docker Installation
-------------------

We provide Docker images for easy deployment:

.. code-block:: bash

    # Pull the latest image
    docker pull pytorchconnectomics/pytc:latest

    # Run container
    docker run --gpus all -it pytorchconnectomics/pytc:latest

    # Or build from Dockerfile
    cd docker
    docker build -t pytc .

See `docker/README.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/docker/README.md>`_
for detailed instructions.

Troubleshooting
---------------

Common Installation Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue: No module named 'torch'**

.. code-block:: bash

    # Solution: Install PyTorch first
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

**Issue: No module named 'omegaconf'**

.. code-block:: bash

    # Solution: Update omegaconf
    pip install --upgrade omegaconf

**Issue: No module named 'cc3d'**

.. code-block:: bash

    # Solution: Install connected-components-3d
    pip install connected-components-3d

**Issue: Could not import MedNeXt**

This is expected if MedNeXt is not installed. MedNeXt is optional. Install it if needed:

.. code-block:: bash

    git clone https://github.com/MIC-DKFZ/MedNeXt.git
    cd MedNeXt
    pip install -e .

**Issue: CUDA out of memory**

Solutions:

- Reduce batch size in config: ``data.batch_size: 1``
- Use mixed precision: ``training.precision: "16-mixed"``
- Reduce patch size: ``data.patch_size: [64, 64, 64]``

**Issue: ImportError on startup**

Reset your environment:

.. code-block:: bash

    pip uninstall connectomics
    pip cache purge
    cd pytorch_connectomics
    pip install -e .[full]

Version Requirements
^^^^^^^^^^^^^^^^^^^^

- **Python**: 3.8+ (3.10 recommended)
- **PyTorch**: 1.8+ (2.0+ recommended)
- **PyTorch Lightning**: 2.0+
- **MONAI**: 0.9.1+ (1.0+ recommended)
- **CUDA**: 10.2+ (11.8+ recommended)

Getting Help
^^^^^^^^^^^^

If you encounter issues:

1. Check the `FAQ <faq.html>`_
2. Search `GitHub Issues <https://github.com/zudi-lin/pytorch_connectomics/issues>`_
3. Ask on `Slack <https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w>`_
4. Open a new issue on GitHub

Next Steps
----------

After installation:

1. Read the `Configuration Guide <config.html>`_ to learn about Hydra configs
2. Follow the `Tutorials <../tutorials/neuron.html>`_ for hands-on examples
3. Check the `Data Loading Guide <dataloading.html>`_ for dataset preparation
4. Explore the `API Reference <../modules/model.html>`_ for advanced usage

Quick Start
^^^^^^^^^^^

Train your first model:

.. code-block:: bash

    # Download example data (if needed)
    # ...

    # Train with example config
    python scripts/main.py --config tutorials/lucchi.yaml

See tutorials for complete examples!
