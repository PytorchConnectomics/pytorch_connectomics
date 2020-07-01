MitoEM segmentation
==============================

This tutorial provides step-by-step guidance for mitochondria segmentation with our benchmark datasets MitoEM.
We consider the task as 3D instance segmentation task and provide three different confiurations of the model output. 
The model is 'unet_res_3d', similar to the one used in `neuron segmentation <https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>`_.
The evaluation of the segmentation results is based on the AP-75. 

.. note::
    The MitoEM dataset has two sub-datasets **rat** and **human**. Three training scripts ``.yaml`` on **rat** are provided in 
    ``pytorch_connectomics/configs/`` for different outputs of the model.
    The pytorch dataset class of synaptic partners is :class:`connectomics.data.dataset.TileDataset`.


#. Introduction to the dataset:


    Dataset can be found here

    .. code-block:: none

        /n/pfister_lab2/Lab/vcg_connectomics/mitochondria/miccai2020/rat

    or

    .. code-block:: none

        /n/pfister_lab2/Lab/vcg_connectomics/mitochondria/miccai2020/human
        
    Dataset description

    .. note::
        :class:`connectomics.data.dataset.TileDataset` only reads in part of the image files 
        correpoding to given coordinates when running.

    - ``im``: include 1000 2d ``.png`` (**4096x4096**) files of input images (resolution **30x8x8** nm)

    - ``mito``: include 1000 2d ``.png`` (**4096x4096**) files of instance labels

    - ``*.json``: :class:`Dict` contain paths to ``.png`` files 


#. Configure ``.yaml`` file.

    .. note::
        Change or add items in ``.yaml`` with keys in ``connectomics/config/config.py``

        ``.yaml`` files can be found in ``pytorch_connectomics/configs/Mito-EM``

    - ``MitoEM-R-A.yaml``: output 3 channels for affinty prediction

    - ``MitoEM-R-AC.yaml``: output 4 channels for both affinity and instance boundary prediction

    - ``MitoEM-R-BC.yaml``: output 2 channels for both binary and instance boundary prediction


#. Run the training script. 

    .. note::
        By default the path of images and labels are not specified. To 
        run the training scripts, please revise the ``DATASET.IMAGE_NAME``, ``DATASET.LABEL_NAME``, ``DATASET.OUTPUT_PATH``
        and ``DATASET.INPUT_PATH`` options in ``configs/MitoEM-R-*.yaml``.
        The options can also be given as command-line arguments without changing of the ``yaml`` configuration files.

    .. code-block:: none

        $ source activate py3_torch
        $ python -u scripts/main.py \
          --config-file configs/MitoEM-R-A.yaml
        

#. Visualize the training progress. More info `here <https://vcg.github.io/newbie-wiki/build/html/computation/machine_rc.html>`_:

    .. code-block:: none

        $ tensorboard --logdir ``OUTPUT_PATH/xxxxx``

    .. note::
        Tensorboard will create a subdir in OUTPUT_PATH. Substitute **xxxxx** with that subdir.

#. Run inference on image volumes:

    .. code-block:: none

        $ source activate py3_torch
        $ python -u scripts/main.py \
          --config-file configs/MitoEM-R-A.yaml --inference \
          --checkpoint OUTPUT_PATH/xxxxx.pth.tar

    .. note::
        Please change the ``INFERENCE.IMAGE_NAME`` ``INFERENCE.OUTPUT_PATH`` ``INFERENCE.OUTPUT_NAME`` 
        options in ``configs/MitoEM-R-A.yaml``.