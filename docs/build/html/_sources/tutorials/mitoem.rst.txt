MitoEM Instance Segmentation
=============================

.. warning::
    This page will be deprecated soon.  Please check `the updated page <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/mito.html#>`_ for a more
    comprehensive mitochondria segmentation tutorial.

This tutorial provides step-by-step guidance for mitochondria segmentation with our benchmark datasets `MitoEM <https://donglaiw.github.io/page/mitoEM/index.html>`_.
We consider the task as 3D instance segmentation task and provide three different confiurations of the model output. 
The model is ``unet_res_3d``, similar to the one used in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_.
The evaluation of the segmentation results is based on the AP-75 (average precision with an IoU threshold of 0.75). 

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