Neuron Segmentation
=======================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of instance segmentation.
The methodology is to first predict the affinity map of pixels with an encoder-decoder ConvNets and
then generate the segmentation map using a segmentation algorithm (e.g., watershed).

The evaluation of segmentation results is based on the `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_
and `Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.

.. note::
    Before running neuron segmentation, please take a look at the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demo>`_
    to get familiar with the datasets and have a sense of how the affinity graphs look like.

The main script is at ``pytorch_connectomics/scripts/main.py``. The pytorch target affinity generation is :class:`connectomics.data.utils.data_segmentation`.

#. Get the dataset:

        .. code-block:: none

            wget http://hp03.mindhackers.org/rhoana_product/dataset/snemi.zip

    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.


#. Provide the ``yaml`` configuration file to run training:

    .. code-block:: none

        $ source activate py3_torch
        $ python scripts/main.py --config-file configs/SNEMI-Neuron-Train.yaml

    The configuration file for training is in ``configs/SNEMI-Neuron-Train.yaml``, modify the following accordingly:
 
    - ``IMAGE_NAME``: Name of the volume file
    - ``LABEL_NAME``: Name of the label file
    - ``INPUT_PATH``: Path to both files above 
    - ``OUTPUT_PATH``: Path to store outputs (Model and Tensorboard files)
    - ``NUM_GPUS``: Number of GPUs to use
    - ``NUM_CPUS``: Number of CPU cores to use

    To run training starting from pretrained weights, add a checkpoint file:

    .. code-block:: none

        $ source activate py3_torch
        $ python scripts/main.py --config-file configs/SNEMI-Neuron-Train.yaml \
        $ --checkpoint /path/to/checkpoint/checkpoint_30000.pth.tar

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir outputs/SNEMI/
                                                                              
#. Run inference on image volumes, The test configuration file can have bigger input, augmentation at inference:

    .. code-block:: none

        $ python scripts/main.py --config-file configs/SNEMI-Neuron-Test.yaml --inference --checkpoint outputs/SNEMI/volume_xxxxx.pth

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

    #. Generate 3D segmentation and report Rand and VI score using ``waterz``:

        .. code-block:: none

            $

    #. You can also run the jupyter notebook `segmentation.ipynb <https://github.com/zudi-lin/pytorch_connectomics/blob/master/demo/segmentation.ipynb>`_ in
       the demo, which provides more options and visualization.
