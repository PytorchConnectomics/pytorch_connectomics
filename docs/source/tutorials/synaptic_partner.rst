Synaptic Partner Segmentation
==============================

This tutorial provides step-by-step guidance for synaptic partner detection with our benchmark datasets. 
We consider the task as a semantic segmentation task and predict both the pre-synaptic and post-synaptic pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the synapse detection results is based on the F1 score. The sparsity and diversity of synapses make the task challenging. 

.. note::
    Our segmentation model uses a model target of three channels. The three channels are **pre-synaptic region**, **post-synaptic region** and **synaptic 
    region** (union of the first two channels), respectively. 

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``.  
The pytorch dataset class of synaptic partners is :class:`connectomics.data.dataset.VolumeDataset`.


#. Dataset examples can be found on the Harvard RC server here:

        Image:

        .. code-block:: none

            /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/gt-syn/syn_0922_im_orig.h5

        Label:

        .. code-block:: none

            /n/pfister_lab2/Lab/vcg_connectomics/cerebellum_P7/gt-syn/syn_0922_seg.h5

#. Run the training script. The training and inference script can take a list of volumes (separated by '@') in either the yaml config file or by command-line arguments.

    .. note::
        By default the path of images and labels are not specified. To 
        run the training scripts, please revise the ``IMAGE_NAME``, ``LABEL_NAME``
        and ``INPUT_PATH`` options in ``configs/Synaptic-Partner-Segmentation.yaml``.
        The options can also be given as command-line arguments without changing of the ``yaml`` configuration files.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Partner-Segmentation.yaml

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Partner-Segmentation.yaml

    .. note::
        We add **higher weights** to the foreground pixels and apply **rejection sampling** to reject samples without synapes during training to heavily penalize
        false negatives. This is beneficial for down-stream proofreading and analysis as correcting false positives is much easier than finding missing synapses in the
        vast volumes.

#. Visualize the training progress. More info `here <https://vcg.github.io/newbie-wiki/build/html/computation/machine_rc.html>`_:

    .. code-block:: none

        $ tensorboard --logdir outputs/synaptic_polarity

#. Run inference on image volumes:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Synaptic-Partner-Segmentation.yaml --inference \
          --checkpoint outputs/synaptic_polarity/volume_xxxxx.pth.tar

    .. note::
        By default the path of images for inference are not specified. Please change 
        the ``INFERENCE.IMAGE_NAME`` option in ``configs/Synaptic-Partner-Segmentation.yaml``.

#. Use `this function <https://github.com/zudi-lin/pytorch_connectomics/blob/master/connectomics/utils/processing/process_syn.py>`_ to convert the probability map into instance/semantic segmentation.