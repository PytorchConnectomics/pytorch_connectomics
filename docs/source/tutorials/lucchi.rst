Mitochondria Segmentaion
==========================

This tutorial provides step-by-step guidance for mitochondria segmentation with the EM benchmark datasets released by `Lucchi et al. <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`_.
We consider the task as a semantic segmentation task and predict the mitochondria pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. 
The evaluation of the mitochondria segmentation results is based on the F1 score and Intersection over Union (IoU).

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--task 2``
when executing the ``train.py`` and ``test.py`` scripts. The pytorch dataset class of synapses is :class:`torch_connectomics.data.dataset.MitoDataset`.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget https://hp03.mindhackers.org/rhoana_product/dataset/lucchi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

#. Run the training script. The training and inference script can take a list of volumes and conduct training/inference at the same time.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u train.py -t /path/to/Lucchi/ \
          -dn img/train_im.h5 -ln label/train_label.h5 -o outputs/unetv0_mito -lr 1e-03 \
          --iteration-total 100000 --iteration-save 10000 -mi 8,256,256 -g 4 -c 4 -b 8 \
          -ac unetv0 --task 2 --out-channel 1

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u test.py -t /path/to/Lucchi/ \
          -dn img/test_im.h5 -o outputs/unetv0_mito/result -mi 8,256,256 -g 4 -c 4 -b 32 \
          -ac unetv0 -lm True -pm outputs/unetv0_mito/volume_50000.pth --task 2