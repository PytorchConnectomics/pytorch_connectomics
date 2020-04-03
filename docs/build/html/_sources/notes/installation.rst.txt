Installation
============

The code is developed and tested on a machine with 8 Nvidia GPUs with the CentOS Linux 7.4 (Core) operation system. 

.. note::
    We do not recommend installation as root user on your system python.
    Please setup an `Anaconda/Miniconda <https://conda.io/docs/user-guide/install/index.html/>`_ environment and add
    the required packages to the environment.

Please follow the steps below for a successful installation:

#. Create a new conda environment:

    .. code-block:: none

        $ conda create -n py3_torch python=3.7
        $ source activate py3_torch
        $ conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

#. Ensure that at least PyTorch 1.3.0 is installed:

    .. code-block:: none

        $ python -c 'import torch; print(torch.__version__)'
        >>> 1.3.0

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
            >>> 9.2

    #. Ensure that PyTorch and system CUDA versions match:

        .. code-block:: none

            $ python -c 'import torch; print(torch.version.cuda)'
            >>> 9.2

            $ nvcc --version
            >>> 9.2

#. Download and install the package:

    .. code-block:: none

        $ git clone git@github.com:zudi-lin/pytorch_connectomics.git
        $ cd pytorch_connectomics
        $ pip install -r requirements.txt
        $ pip install --editable .

.. note::
    If you meet compilation errors, please check the `TROUBLESHOOTING.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/TROUBLESHOOTING.md>`_.
    It is highly recommended to first play with the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demo>`_ scripts to make sure that
    the installation is correct and also have intial taste of the modules.
