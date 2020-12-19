Synaptic Cleft Segmentation
============================

This tutorial provides step-by-step guidance for synaptic cleft detection with `CREMI <https://cremi.org>`_ benchmark datasets.
We consider the task as a semantic segmentation task and predict the synapse pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the synapse detection results is based on the F1 score and average distance. See `CREMI metrics <https://cremi.org/metrics/>`_
for more details.

.. note::

    We preform re-alignment of the original CREMI image stacks and also remove the crack artifacts. Please reverse 
    the alignment before submitting the test prediction to the CREMI challenge.

Script needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. YAML files can be found at ``pytorch_connectomics/configs/``, where stores the common setting for current experiment's configuration. Default config file can be found at ``pytorch_connectomics/connectomics/config/``, where stores all the configuration. The pytorch dataset class of synapses is :class:`connectomics.data.dataset.VolumeDataset`.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget http://rhoana.rc.fas.harvard.edu/dataset/cremi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

    .. note::
        If you use the original CREMI challenge datasets or the data processed by yourself, the file names can be
        different from the default ones. In such case, please change the corresponding entries, including ``IMAGE_NAME``, 
        ``LABEL_NAME`` and ``INPUT_PATH`` in the `CREMI config file <https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/CREMI-Synaptic-Cleft.yaml>`_.

#. Run the main.py script for training. This script can take a list of volumes and conduct training/inference at the same time.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/CREMI-Synaptic-Cleft.yaml

    - ``config-file``: configuration setting for the current experiment.

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run the main.py script for inference:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/CREMI-Synaptic-Cleft.yaml \
          --checkpoint outputs/CREMI_syn_baseline/volume_50000.pth.tar \
          --inference

    - ``config-file``: configuration setting for current experiments.
    - ``inference``: will run inference when given, otherwise will run training instead.
    - ``checkpoint``: the pre-trained checkpoint file for inference.
