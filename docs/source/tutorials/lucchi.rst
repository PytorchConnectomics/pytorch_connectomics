Mitochondria Segmentation
==========================

.. warning::
    This page will be deprecated soon. Please check `the updated page <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/mito.html#>`_ for a more
    comprehensive mitochondria segmentation tutorial.

This tutorial provides step-by-step guidance for mitochondria segmentation with the EM benchmark datasets released by `Lucchi et al. <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`_.
We consider the task as a semantic segmentation task and predict the mitochondria pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. The evaluation of the mitochondria segmentation results is based on the F1 score and Intersection over Union (IoU).

.. note::
    Different from other EM connectomics datasets used in the tutorials, the dataset released by Lucchi et al. is an isotropic dataset,
    which means the spatial resolution along all three axes is the same. Therefore a completely 3D U-Net and data augmentation along z-x
    and z-y planes besides x-y planes are preferred.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--config-file configs/Lucchi-Mitochondria.yaml`` during training and inference to load the required configurations for this task. 
The pytorch dataset class of lucchi data is :class:`connectomics.data.dataset.VolumeDataset`.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget http://rhoana.rc.fas.harvard.edu/dataset/lucchi.zip
    
    For description of the data please check `the author page <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

#. Run the training script:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Lucchi-Mitochondria.yaml

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on test image volumes. Our model achieves a VOC-test score of 0.945.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Lucchi-Mitochondria.yaml --inference \
          --checkpoint outputs/Lucchi_mito_baseline/volume_50000.pth.tar
