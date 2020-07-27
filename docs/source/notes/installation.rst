Installation
=============

The code is developed and tested on a machine with 8 NVIDIA GPUs with the CentOS Linux 7.4 (Core) operation system. 

.. note::
    We do not recommend installation as root user on your system python.
    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment and add
    the required packages to the environment.

Please follow the steps below for a successful installation:

#. Create a new conda environment:

    .. code-block:: none

        $ conda create -n py3_torch python=3.8
        $ source activate py3_torch
        $ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

#. Ensure that at least PyTorch 1.5.1 is installed:

    .. code-block:: none

        $ python -c 'import torch; print(torch.__version__)'
        >>> 1.5.1

#. Ensure CUDA is setup correctly (optional):

    #. Check if PyTorch is installed with CUDA support:

        .. code-block:: none

            $ python -c 'import torch; print(torch.cuda.is_available())'
            >>> True

    #. Add CUDA to ``$PATH`` and ``$CPATH`` (note that your actual CUDA path may vary from ``/usr/local/cuda``):

        .. code-block:: none

            $ PATH=/usr/local/cuda/bin:$PATH
            $ echo $PATH
            >>> /usr/local/cuda/bin:...

            $ CPATH=/usr/local/cuda/include:$CPATH
            $ echo $CPATH
            >>> /usr/local/cuda/include:...

    #. Verify that ``nvcc`` is accessible from terminal:

        .. code-block:: none

            $ nvcc --version
            >>> 10.2

    #. Ensure that PyTorch and system CUDA versions match:

        .. code-block:: none

            $ python -c 'import torch; print(torch.version.cuda)'
            >>> 10.2

            $ nvcc --version
            >>> 10.2
    
    The codebased is mainly developed and tested on the Harvard `FASRC <https://www.rc.fas.harvard.edu>`_ cluster. 
    Please load required CUDA modules from the `RC server module list <https://portal.rc.fas.harvard.edu/p3/build-reports/>`_ during 
    running and development on the RC server.
     
#. Download and install the package:

    .. code-block:: none

        $ git clone https://github.com/zudi-lin/pytorch_connectomics.git
        $ cd pytorch_connectomics
        $ pip install --upgrade pip
        $ pip install -r requirements.txt
        $ pip install --editable .

    We install the package in editable mode by default so that there is no need to
    re-install it when making changes to the code. 

.. note::
    If you meet compilation errors, please open an issue and describe the steps to reproduce the errors.
    It is highly recommended to first play with the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demos>`_ Jupyter notebooks to 
    make sure that the installation is correct and also have an intial taste of the modules.
