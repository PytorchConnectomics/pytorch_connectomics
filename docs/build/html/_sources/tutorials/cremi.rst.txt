Synaptic Cleft Detection
=======================

This tutorial provides step-by-step guidance for synaptic cleft detection with `CREMI<https://cremi.org>`_ benchmark datasets.
We consider the task as a semantic segmentation task and predict the synapse pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in 'neuron segmentation<https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>'_. 
The evaluation of the synapse detection results is based on the F1 score and average distance. See `CREMI metrics<https://cremi.org/metrics/>`_
for more details.

.. note::
    We preform re-alignment of the original CREMI image stacks and also remove the crack artifacts. Please reverse 
    the alignment before submitting the test prediction to the CREMI challenge.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--task 2``
when executing the ``train.py`` and ``test.py`` scripts. The pytorch dataset class of synapses is :class:`torch_connectomics.data.dataset.SynapseDataset`.

#. Get the dataset:

    #. Download the original images from our server:

        .. code-block:: none

            wget http://140.247.107.75/rhoana_product/snemi/image/train-input.tif
            wget http://140.247.107.75/rhoana_product/snemi/seg/train-labels.tif
            wget http://140.247.107.75/rhoana_product/snemi/image/test-input.tif

#. Run the training script:

    .. code-block:: none

        $ source activate py3_torch
        $ python -u train.py -t /path/to/cremi/ -dn images/im_A_v2_200.h5@images/im_B_v2_200.h5@images/im_C_v2_200.h5 \
          -ln gt-syn/syn_A_v2_200.h5@gt-syn/syn_B_v2_200.h5@gt-syn/syn_C_v2_200.h5 -o outputs/unetv0 -lr 1e-03 \
          --iteration-total 100000 --iteration-save 10000 -mi 8,256,256 -g 4 -c 4 -b 8 -ac unetv0

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py -t /path/to/cremi/ \
          -dn images/im_A_v2_200.h5@images/im_B_v2_200.h5@images/im_C_v2_200.h5 \
          -mi 8,256,256 -g 4 -c 4 -b 16 -ac unetv0 -lm True -pm outputs/unetv0/volume_50000.pth