Mitochondria Segmentation
===========================

.. contents::
   :local:

Introduction
-------------

This tutorial has two parts. In the first part, you will learn how to make **pixel-wise class prediction** on the dataset released 
by Lucchi et al. in 2012. In the second part, you will learn how to predict the **instance masks** of individual mitochondrion on the MitoEM
dataset released by Wei et al. in 2020.

Semantic Segmentation
----------------------

This section provides step-by-step guidance for mitochondria segmentation with the EM benchmark datasets released by `Lucchi et al. <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`_.
We consider the task as a semantic segmentation task and predict the mitochondria pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_. The evaluation of the mitochondria segmentation results is based on the F1 score and Intersection over Union (IoU).

.. note::
    Different from other EM connectomics datasets used in the tutorials, the dataset released by Lucchi et al. is an isotropic dataset,
    which means the spatial resolution along all three axes is the same. Therefore a completely 3D U-Net and data augmentation along z-x
    and z-y planes besides x-y planes are preferred.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--config-file configs/Lucchi-Mitochondria.yaml`` during training and inference to load the required configurations for this task. 
The pytorch dataset class of lucchi data is :class:`connectomics.data.dataset.VolumeDataset`.

.. figure:: ../_static/img/lucchi_qual.png
    :align: center
    :width: 800px

    Qualitative results of the model prediction on the mitochondria segmentation dataset released by 
    Lucchi et al., without any post-processing.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget https://hp03.mindhackers.org/rhoana_product/dataset/lucchi.zip
    
    For description of the data please check `the author page <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

#. Run the training script:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Lucchi-Mitochondria.yaml

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on test image volumes. Our model achieves a VOC score of 0.942 on the test set.

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Lucchi-Mitochondria.yaml --inference \
          --checkpoint outputs/Lucchi_mito_baseline/volume_100000.pth.tar

Instance Segmentation
----------------------

This section provides step-by-step guidance for mitochondria segmentation with our benchmark datasets `MitoEM <https://donglaiw.github.io/page/mitoEM/index.html>`_.
We consider the task as 3D instance segmentation task and provide three different confiurations of the model output. 
The model is ``unet_res_3d``, similar to the one used in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_.
The evaluation of the segmentation results is based on the AP-75 (average precision with an IoU threshold of 0.75). 

.. figure:: ../_static/img/mito_complex.png
    :align: center
    :width: 800px

    Complex mitochondria in the MitoEM dataset:(a) mitochondria-on-a-string (MOAS), and (b) dense tangle of touching mitochondria. 
    Those challenging cases are prevalent but not covered in previous labeled datasets.

.. note::
    The MitoEM dataset has two sub-datasets **Rat** and **Human** based on the source of the tissues. Three training configuration files on **MitoEM-Rat** 
    are provided in ``pytorch_connectomics/configs/MitoEM/`` for different learning targets of the model. 

.. note::
    Since the dataset is very large and can not be directly loaded into memory, we use the :class:`connectomics.data.dataset.TileDataset` dataset class that only 
    loads part of the whole volume by opening involved ``.png`` images.

#. Introduction to the dataset:

    On the Harvard RC cluster, the datasets can be found at:

    .. code-block:: none

        /n/pfister_lab2/Lab/vcg_connectomics/mitochondria/miccai2020/rat

    and

    .. code-block:: none

        /n/pfister_lab2/Lab/vcg_connectomics/mitochondria/miccai2020/human
        
    Dataset description

    - ``im``: includes 1,000 single-channel ``.png`` files (**4096x4096**) of raw EM images (with a spatial resolution of **30x8x8** nm).

    - ``mito``: includes 1,000 single-channel ``.png`` files (**4096x4096**) of instance labels.

    - ``*.json``: :class:`Dict` contains paths to ``.png`` files 


#. Configure ``.yaml`` files for different learning targets.

    - ``MitoEM-R-A.yaml``: output 3 channels for affinty prediction.

    - ``MitoEM-R-AC.yaml``: output 4 channels for both affinity and instance contour prediction.

    - ``MitoEM-R-BC.yaml``: output 2 channels for both binary mask and instance contour prediction.


#. Run the training script. 

    .. note::
        By default the path of images and labels are not specified. To 
        run the training scripts, please revise the ``DATASET.IMAGE_NAME``, ``DATASET.LABEL_NAME``, ``DATASET.OUTPUT_PATH``
        and ``DATASET.INPUT_PATH`` options in ``configs/MitoEM-R-*.yaml``.
        The options can also be given as command-line arguments without changing of the ``yaml`` configuration files.

    .. code-block:: none

        $ source activate py3_torch
        $ python -u scripts/main.py --config-file configs/MitoEM-R-A.yaml
        

#. Visualize the training progress. More info `here <https://vcg.github.io/newbie-wiki/build/html/computation/machine_rc.html>`_:

    .. code-block:: none

        $ tensorboard --logdir ``OUTPUT_PATH/xxxxx``

    .. note::
        Tensorboard will create a subdir in OUTPUT_PATH. Substitute **xxxxx** with the subdir name.

#. Run inference on image volumes:

    .. code-block:: none

        $ source activate py3_torch
        $ python -u scripts/main.py \
          --config-file configs/MitoEM-R-A.yaml --inference \
          --checkpoint OUTPUT_PATH/xxxxx.pth.tar

    .. note::
        Please change the ``INFERENCE.IMAGE_NAME`` ``INFERENCE.OUTPUT_PATH`` ``INFERENCE.OUTPUT_NAME`` 
        options in ``configs/MitoEM-R-A.yaml``.
