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

            wget http://hp03.mindhackers.org/rhoana_product/dataset/cremi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

#. Run the main.py script for training. This script can take a list of volumes and conduct training/inference at the same time.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u main.py \
          --config-file configs/CREMI-Synaptic-Cleft-Train.yaml \
          --output "outputs/cremi_synapse_train"  

    - config-file: configuration setting for the current experiment.
    - output: the training results saving path.

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run the main.py script for inference:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u main.py \
          --config-file configs/CREMI-Synaptic-Cleft-Train.yaml \
          --output "outputs/cremi_synapse_inference" \
          --inference --checkpoint outputs/cremi_synapse_inference/volume_30000.pth.tar

    - config-file: configuration setting for current experiments.
    - output: the inference results saving path. 
    - inference: will run inference when given, otherwise will run training instead.
    - checkpoint: the pre-trained checkpoint file for inference.
