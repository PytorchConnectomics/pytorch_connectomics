Installation
==============

The PyTorch Connectomics package is mainly developed on Linux machines with NVIDIA GPUs. We recommend the users to
follow the `Linux Installation <installation.html#id1>`_ guidance to ensure the compatibility of latest
features with your system. For Windows users, please check the `Windows Installation <installation.html#id2>`_ section.

Linux Installation
---------------------

The code is developed and tested on a machine with 8 NVIDIA GPUs with the CentOS Linux 7.4 (Core) operation system. 

.. tip::

    We do not recommend installation as root user on your system python.
    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment and add
    the required packages to the environment.

Please follow the steps below for a successful installation:

1 - Install PyTorch in a virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    conda create -n py3_torch python=3.8
    source activate py3_torch
    conda install pytorch torchvision cudatoolkit=11.3 -c pytorch

More options to install PyTorch can be found `here <https://pytorch.org/get-started/locally/>`_. Our package has been tested with 
CUDA 10.2 and 11.4. Then please ensure that at least PyTorch **1.10.0** is installed:

.. code-block:: none

    python -c 'import torch; print(torch.__version__)'
    >>> 1.10.0

2 - Install PyTorch Connectomics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install --editable .

We install the package in editable mode by default so that there is no need to
re-install it when making changes to the code. 

3 - Ensure CUDA is setup correctly (*optional*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

.. code-block:: console

    python -c 'import torch; print(torch.version.cuda)'
    >>> 11.0
    
The codebase is mainly developed and tested on the Harvard `FASRC <https://www.rc.fas.harvard.edu>`_ cluster. 
For FASRC users, please load required CUDA modules from the `RC server module list <https://portal.rc.fas.harvard.edu/p3/build-reports/>`_ during 
running and development on the cluster. For example:

.. code-block:: console

    module load cuda cudnn

If you only want to install pytorch_connectomics as a Python library without clone the repository with all the pre-defined configuration files, please
use ``pip`` to directly install it from GitHub:

.. code-block:: console

    pip install git+https://github.com/zudi-lin/pytorch_connectomics.git    

.. note::

    If you meet compilation errors, please open an issue and describe the steps to reproduce the errors.
    It is highly recommended to first play with the Jupyter `notebooks <https://github.com/zudi-lin/pytorch_connectomics/tree/master/notebooks>`_ to 
    make sure that the installation is correct and also have an intial taste of the functions/modules.

Windows Installation
----------------------

These installation instructions were tested on two different Windows 10 machines, each with 1 GPU device. 

.. note::

    These instructions were designed to be used on a Windows computer without assuming any previous software was installed, or any command-line familiarity.

Please follow the steps below for a successful installation:

1 - Install Miniconda
^^^^^^^^^^^^^^^^^^^^^^^^^

The instructions to install miniconda can be found `here <https://docs.conda.io/en/latest/miniconda.html>`_.
Most likely you want to use the link for "Miniconda3 Windows 64-bit"

2 - Open Anaconda Prompt
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open the anaconda prompt. You should be able to find this in the windows start menu with your other programs. Either search for it, or look in the folder most likely called "Anaconda 3 (64-bit)" Another way to find it is by clicking the start menu / press the windows key, start typing miniconda, and select "Anaconda Prompt (Miniconda3)"

3 - Navigate to where you want to install the package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set miniconda's working directory to where you want to install the program by typing the following command with out the <>. You can install the program wherever you want, just remember where you choose to install it. The default is to install it in your C:\Users\YourUsername folder. If you are ok with that location, skip this step.

.. code-block:: none

    cd <path of where you want to install the program folder, example: C:\\Users\\YourUsername\\Documents>
    
4 - Run the following commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The next few commands create a virtual environment, install pytorch and pytorch_connectomics, and also some libraries that windows needs to process images properly.

.. code-block:: none

    conda create --name py3_torch python=3.8.11 -y
    conda activate py3_torch
    conda install git -y
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
    git clone https://github.com/zudi-lin/pytorch_connectomics.git
    cd pytorch_connectomics
    pip install --editable .
    cd ..
    conda install -c conda-forge imagecodecs -y
    echo Completely finished with installation. Software is ready to use
    
.. note::

    The software is now installed. When you want to use the software, you must open the anaconda prompt and type the command ``conda activate py3_torch``.
