Mitochondria Segmentaion
==========================

This tutorial provides step-by-step guidance for mitochondria segmentation with the EM benchmark datasets released by `Lucchi et al. <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`_.
We consider the task as a semantic segmentation task and predict the mitochondria pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the mitochondria segmentation results is based on the F1 score and Intersection over Union (IoU).

.. note::
    Our input and output sizes are 512 * 512 pixels, respectively, with the input being fed in as a grayscale image and the output being a binary mask, highlighting mitochondria as the positive class.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--task 2``
when executing the ``train.py`` and ``test.py`` scripts. The pytorch dataset class of lucchi data is:class:`torch_connectomics.data.dataset.MitoDataset`.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget https://hp03.mindhackers.org/rhoana_product/dataset/lucchi.zip
    
    For description of the data please check `the author page <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

#. Run the training script.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py \
          --config-file configs/Mitochondria-Segmentation.yaml
          
      .. note::
      The training and inference script can take a Tlist of volumes and conduct training/inference at the same time.

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on test image volumes (change ``LOG-FOLDER``): VOC-test=0.945

    .. code-block:: none

        $ python -u test.py -i /path/to/Lucchi/ -din img/test_im.tif -o outputs/unetv0_mito/result\
          -mi 112,256,256  -g 1 -c 1 -b 1 -ma unet_residual -mf 28,36,48,64,80 -me 0 -moc 1 
          -mpt outputs/unet_res_mito/LOG-FOLDER/volume_59999.pth -mpi 59999 -dp 8,64,64
