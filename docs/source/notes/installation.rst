Installation
=============

The code is developed and tested on a machine with 8 NVIDIA GPUs with the CentOS Linux 7.4 (Core) operation system. 

.. tip::

    We do not recommend installation as root user on your system python.
    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment and add
    the required packages to the environment.

Please follow the steps below for a successful installation:

1 - Install PyTorch in a virtual environment
----------------------------------------------

.. code-block:: none

    conda create -n py3_torch python=3.8
    source activate py3_torch
    conda install pytorch torchvision cudatoolkit=11.0 -c pytorch

More options to install PyTorch can be found `here <https://pytorch.org/get-started/locally/>`_. Our package has been tested with 
CUDA 10.2 and 11.0. Then please ensure that at least PyTorch **1.8.0** is installed:

.. code-block:: none

    python -c 'import torch; print(torch.__version__)'
    >>> 1.8.0

2 - Install PyTorch Connectomics
----------------------------------

.. code-block:: none

    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install --editable .

We install the package in editable mode by default so that there is no need to
re-install it when making changes to the code. 

3 - Ensure CUDA is setup correctly (*optional*)
-------------------------------------------------

Check that PyTorch is installed with CUDA support:

.. code-block:: none

    python -c 'import torch; print(torch.cuda.is_available())'
    >>> True

Add CUDA to ``$PATH`` and ``$CPATH`` (note that your actual CUDA path may vary from ``/usr/local/cuda``):

.. code-block:: none

    PATH=/usr/local/cuda/bin:$PATH
    echo $PATH
    >>> /usr/local/cuda/bin:...

    CPATH=/usr/local/cuda/include:$CPATH
    echo $CPATH
    >>> /usr/local/cuda/include:...

Verify that ``nvcc`` is accessible from terminal:

.. code-block:: none

    nvcc --version
    >>> nvcc: NVIDIA (R) Cuda compiler driver
    >>> Copyright (c) 2005-2020 NVIDIA Corporation
    >>> Built on Wed_Jul_22_19:09:09_PDT_2020
    >>> Cuda compilation tools, release 11.0, V11.0.221
    >>> Build cuda_11.0_bu.TC445_37.28845127_0

Ensure that PyTorch and system CUDA versions match:

.. code-block:: none

    python -c 'import torch; print(torch.version.cuda)'
    >>> 11.0
    
The codebase is mainly developed and tested on the Harvard `FASRC <https://www.rc.fas.harvard.edu>`_ cluster. 
For FASRC users, please load required CUDA modules from the `RC server module list <https://portal.rc.fas.harvard.edu/p3/build-reports/>`_ during 
running and development on the cluster. For example:

.. code-block:: none

    module load cuda cudnn

.. note::

    If you meet compilation errors, please open an issue and describe the steps to reproduce the errors.
    It is highly recommended to first play with the Jupyter `notebooks <https://github.com/zudi-lin/pytorch_connectomics/tree/master/notebooks>`_ to 
    make sure that the installation is correct and also have an intial taste of the functions/modules.
