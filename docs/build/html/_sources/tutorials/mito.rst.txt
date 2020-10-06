Mitochondria Segmentation
===========================

.. contents::
   :local:

Introduction
-------------

`Mitochondria <https://en.wikipedia.org/wiki/Mitochondrion>`__ are the primary energy providers for cell activities, thus essential for metabolism. 
Quantification of the size and geometry of mitochondria is not only crucial to basic neuroscience research, but also informative to 
clinical studies including, but not limited to, bipolar disorder and diabetes.

This tutorial has two parts. In the first part, you will learn how to make **pixel-wise class prediction** on the widely used benchmark
dataset released by `Lucchi et al. <https://ieeexplore.ieee.org/document/6619103>`__ in 2012. In the second part, you will learn how to predict the **instance masks** of 
individual mitochondrion from the large-scale MitoEM dataset released by `Wei et al. <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`__ in 2020.

Semantic Segmentation
----------------------

This section provides step-by-step guidance for mitochondria segmentation with the EM benchmark datasets released by `Lucchi et al. <https://cvlab.epfl.ch/research/page-90578-en-html/research-medical-em-mitochondria-index-php/>`__.
We consider the task as a **semantic segmentation** task and predict the mitochondria pixels with encoder-decoder ConvNets similar to
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

    Download the dataset from our server:

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

#. Run inference on test image volume:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
          --config-file configs/Lucchi-Mitochondria.yaml --inference \
          --checkpoint outputs/Lucchi_mito_baseline/volume_100000.pth.tar

#. Since the ground-truth label of the test set is public, we can run the evaluation locally:

    .. code-block:: python

        from connectomics.utils.evaluation import get_binary_jaccard
        pred = pred / 255. # output is casted to uint8 with range [0,255].
        gt = (gt!==0).astype(np.uint8)
        thres = [0.4, 0.6, 0.8] # evaluate at multiple thresholds.
        scores = get_binary_jaccard(pred, gt, thres)

    The prediction can be further improved by conducting median filtering to remove noise:

    .. code-block:: python

        from connectomics.utils.evaluation import get_binary_jaccard
        from connectomics.utils.processing import binarize_and_median
        pred = pred / 255. # output is casted to uint8 with range [0,255].
        pred = binarize_and_median(pred, size=(7,7,7), thres=0.8)
        gt = (gt!==0).astype(np.uint8)
        scores = get_binary_jaccard(pred, gt) # prediction is already binarized

Our pretained model achieves a foreground IoU and IoU of **0.892** and **0.943** on the test set, respectively. The results are better or on par with
state-of-the-art approaches. Please check `BENCHMARK.md <https://github.com/zudi-lin/pytorch_connectomics/blob/master/BENCHMARK.md>`_  for detailed performance 
comparison and the pre-trained models.

Instance Segmentation
----------------------

This section provides step-by-step guidance for mitochondria segmentation with our benchmark datasets `MitoEM <https://donglaiw.github.io/page/mitoEM/index.html>`_.
We consider the task as 3D **instance segmentation** task and provide three different confiurations of the model output. 
The model is ``unet_res_3d``, similar to the one used in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_.
The evaluation of the segmentation results is based on the AP-75 (average precision with an IoU threshold of 0.75). 

.. figure:: ../_static/img/mito_complex.png
    :align: center
    :width: 800px

    Complex mitochondria in the MitoEM dataset:(a) mitochondria-on-a-string (MOAS), and (b) dense tangle of touching instances. 
    Those challenging cases are prevalent but not covered in previous datasets.

.. note::
    The MitoEM dataset has two sub-datasets **Rat** and **Human** based on the source of the tissues. Three training configuration files on **MitoEM-Rat** 
    are provided in ``pytorch_connectomics/configs/MitoEM/`` for different learning targets of the model. 

.. note::
    Since the dataset is very large and can not be directly loaded into memory, we use the :class:`connectomics.data.dataset.TileDataset` dataset class that only 
    loads part of the whole volume by opening involved ``PNG`` images.

#. Introduction to the dataset:

    On the Harvard RC cluster, the datasets can be found at:

    .. code-block:: none

        /n/pfister_lab2/Lab/vcg_connectomics/mitochondria/miccai2020/rat

    and

    .. code-block:: none

        /n/pfister_lab2/Lab/vcg_connectomics/mitochondria/miccai2020/human

    For the public link of the dataset, check the `project page <https://donglaiw.github.io/page/mitoEM/index.html>`_.
        
    Dataset description:

    - ``im``: includes 1,000 single-channel ``*.png`` files (**4096x4096**) of raw EM images (with a spatial resolution of **30x8x8** nm).

    - ``mito``: includes 1,000 single-channel ``*.png`` files (**4096x4096**) of instance labels.

    - ``*.json``: :class:`Dict` contains paths to ``*.png`` files 


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

        $ tensorboard --logdir ``OUTPUT_PATH/<EXP_DIR_NAME>``

    .. note::
        Our utility functions will create a subdir in OUTPUT_PATH to save the Tensorboard event files. Substitute **<EXP_DIR_NAME>** with your subdir name.

#. Run inference on image volumes:

    .. code-block:: none

        $ source activate py3_torch
        $ python -u scripts/main.py \
          --config-file configs/MitoEM-R-A.yaml --inference \
          --checkpoint OUTPUT_PATH/checkpoint_<ITER_NUM>.pth.tar

    .. note::
        Please change the ``INFERENCE.IMAGE_NAME`` ``INFERENCE.OUTPUT_PATH`` ``INFERENCE.OUTPUT_NAME`` 
        options in ``configs/MitoEM-R-A.yaml``.
