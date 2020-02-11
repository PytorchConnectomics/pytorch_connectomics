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

        $ module load cuda/9.0-fasrc02 cudnn/7.0_cuda9.0-fasrc01 boost # on Harvard rc cluster
        $ source activate py3_torch
        $ python -u train.py -t /path/to/Lucchi/ \
          -dn img/train_im.tif -ln label/train_label.tif -o outputs/unet_res_mito\
          -lr 1e-03 --iteration-total 60000 --iteration-save 10000  -g 4 -c 4 -b 4 \
          -mi 112,112,112 -mo 112,112,112 -ma unet_residual -mf 28,36,48,64,80 -me 0 -daz 1 --task 2 -oc 1 -lt 1

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on test image volumes (change ``LOG-FOLDER``): VOC-test=0.945

    .. code-block:: none

        $ python -u test.py -t /path/to/Lucchi/ \
          -dn img/test_im.tif -o outputs/unetv0_mito/result\
          -mi 112,112,112 -mo 112,112,112 -g 4 -c 4 -b 4 -ma unet_residual -mf 28,36,48,64,80 -me 0 -oc 1 
          -pm outputs/unet_res_mito/LOG-FOLDER/volume_59999.pth -tam mean -tan 4 -tsz 112,224,224
