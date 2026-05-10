CREMI (Synaptic Cleft Detection)
==================================

This tutorial provides step-by-step guidance for synaptic cleft detection
with `CREMI <https://cremi.org>`_ benchmark datasets. We consider the task
as a semantic segmentation task and predict the synapse pixels with
encoder-decoder ConvNets similar to the models used in affinity prediction
in :doc:`../neuron/snemi3d`. The evaluation of the synapse detection
results is based on the F1 score and average distance. See
`CREMI metrics <https://cremi.org/metrics/>`_ for more details.

    .. note::

        We preform re-alignment of the original CREMI image stacks and also remove the crack artifacts. Please reverse the alignment before submitting the test prediction to the CREMI challenge.

Script needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Modern *YAML* examples live under
``pytorch_connectomics/tutorials/``, with structured defaults under ``pytorch_connectomics/connectomics/config/``.
Synaptic cleft detection uses the Lightning data factory, which selects datasets from
:mod:`connectomics.data.datasets` based on the ``data`` config.

.. figure:: ../../_static/img/cremi_qual.png
    :align: center
    :width: 800px

Qualitative results of the synaptic cleft prediction (red segments) on the CREMI challenge test volumes. The three images from left to right are
cropped from volume A+, B+, and C+, respectively.

1 - Get the dataset
^^^^^^^^^^^^^^^^^^^^^

Download the dataset from the `challenge page <https://cremi.org/>`_, or the Harvard RC server:

.. code-block:: none

    wget http://rhoana.rc.fas.harvard.edu/dataset/cremi.zip

Or execute the following snippet in the root directory:

    .. note::
        If you use the original CREMI challenge datasets or the data processed by yourself, the file names can be different from the default ones. In such case, please change the corresponding entries, including ``IMAGE_NAME``, ``LABEL_NAME`` and ``INPUT_PATH`` in the `CREMI config file <https://github.com/zudi-lin/pytorch_connectomics/blob/master/configs/CREMI-Synaptic-Cleft.yaml>`_.

2 - Run training
^^^^^^^^^^^^^^^^^^

For the CREMI dataset that has multiple volumes, our framework can take a list of volumes and
conduct training/inference at the same time.

.. code-block:: none

    source activate py3_torch
    python -u scripts/main.py \
    --config-base configs/CREMI/CREMI-Base.yaml \
    --config-file configs/CREMI/CREMI-Foreground-UNet.yaml

Or if using multiple GPUs for higher performance:

.. code-block:: none

    source activate py3_torch
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -u -m torch.distributed.run \
    --nproc_per_node=4 --master_port=2345 scripts/main.py --distributed \
    --config-base configs/CREMI/CREMI-Base_multiGPU.yaml \
    --config-file configs/CREMI/CREMI-Foreground-UNet.yaml


3 - Visualize the training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    tensorboard --logdir outputs/CREMI_Binary_UNet

4 - Run inference
^^^^^^^^^^^^^^^^^^

.. code-block:: none

    python -u scripts/main.py \
    --inference --config-base configs/CREMI/CREMI-Base.yaml \
    --config-file configs/CREMI/CREMI-Foreground-UNet.yaml \
    --checkpoint outputs/CREMI_Binary_UNet/volume_100000.pth.tar
