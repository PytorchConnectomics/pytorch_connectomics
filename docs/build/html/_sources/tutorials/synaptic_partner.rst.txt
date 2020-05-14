Synaptic Partner Detection
===========================

This tutorial provides step-by-step guidance for synaptic partner detection with the benchmark datasets.
We consider the task as a semantic segmentation task and predict both the pre-synaptic and post-synaptic pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the synapse detection results is based on the F1 score. 

.. note::
    Our segmentation task consists of 2 targets and 3 channels. Targets are background and synapses whereas channels are pre-synapse, post-synapse and background.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``.  The pytorch dataset class of synapses is :class:`torch_connectomics.data.dataset.VolumeDataset`.


#. Dataset examples can be found on the Harvard RC server here:

        Image:

        .. code-block:: none

            /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/gt-syn/syn_0922_im_orig.h5

        Label:

        .. code-block:: none

            /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/gt-syn/syn_0922_seg.h5

#. Run the training script. The training and inference script can take a list of volumes (separated by '@') in either the yaml config file or by command-line arguments.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Partner-Segmentation.yaml

#. Visualize the training progress. More info `here <https://vcg.github.io/newbie-wiki/build/html/computation/machine_rc.html>`_:

    .. code-block:: none

        $ tensorboard --logdir outputs/synaptic_polarity

#. Run inference on image volumes:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Partner-Segmentation.yaml \
          --inference