Neuron Segmentation
=====================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of instance segmentation.
The methodology is to first predict the affinity map (the connectivity of each pixel to neighboring pixels) 
with an encoder-decoder ConvNets and then generate the segmentation map using a standard
segmentation algorithm (e.g., watershed).

The evaluation of segmentation results is based on the `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_
and `Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.

.. note::
    Before running neuron segmentation, please take a look at the `demos <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demos>`_
    to get familiar with the datasets.

The main script to run the training and inference is ``pytorch_connectomics/scripts/main.py``. 
The pytorch target affinity generation is :class:`connectomics.data.dataset.VolumeDataset`.

#. Get the dataset:

        .. code-block:: none

            wget http://hp03.mindhackers.org/rhoana_product/dataset/snemi.zip

    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

#. Provide the ``yaml`` configuration file to run training:

    .. code-block:: none

        $ source activate py3_torch
        $ python scripts/main.py --config-file configs/SNEMI-Neuron.yaml

    The configuration file for training is in ``configs/SNEMI-Neuron.yaml`` [`link <https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/SNEMI-Neuron.yaml>`_]. 
    We usualy create a ``datasets/`` folder under ``pytorch_connectomics`` and put the SNEMI dataset there. Please modify the following options according to
    your system configuration and data storage:
 
    - ``IMAGE_NAME``: Name of the volume file (HDF5 or TIFF).
    - ``LABEL_NAME``: Name of the label file (HDF5 or TIFF).
    - ``INPUT_PATH``: Path to both files above.
    - ``OUTPUT_PATH``: Path to store outputs (checkpoints and Tensorboard events).
    - ``NUM_GPUS``: Number of GPUs to use.
    - ``NUM_CPUS``: Number of CPU cores to use (for data loading).

    To run training starting from pretrained weights, add a checkpoint file:

    .. code-block:: none

        $ source activate py3_torch
        $ python scripts/main.py --config-file configs/SNEMI-Neuron.yaml \
        $ --checkpoint /path/to/checkpoint/checkpoint_xxxxx.pth.tar

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir outputs/SNEMI/
                                                                              
#. Run inference on image volumes, The test configuration file can have bigger input, augmentation at inference:

    .. code-block:: none

        $ python scripts/main.py --config-file configs/SNEMI-Neuron.yaml 
        --inference --checkpoint outputs/SNEMI/checkpoint_xxxxx.pth

#. Generate segmentation and run evaluation:

    #. Download the ``waterz`` package:

        .. code-block:: none

            $ git clone git@github.com:zudi-lin/waterz.git
            $ cd waterz
            $ pip install --editable .

    #. Download the ``zwatershed`` package:

        .. code-block:: none

            $ git clone git@github.com:zudi-lin/zwatershed.git
            $ cd zwatershed
            $ pip install --editable .

    #. Generate 3D segmentation and report Rand and VI score using ``waterz``.
    Please see examples at `https://github.com/zudi-lin/waterz <https://github.com/zudi-lin/waterz>`_.
