Neuron Segmentation
=======================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of instance segmentation.
The methodology is to first predict the affinity map of pixels with an encoder-decoder ConvNets and 
then generate the segmentation map using a segmentation algorithm (e.g. watershed). 

.. note::
    Before running neuron segmentation, please take a look at the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demo>`_
    to get familiar with the datasets and have a sense of how the affinity graphs look like.

#. Get the dataset:

    #. Download the original images from our server:
        .. code-block:: none

            wget http://140.247.107.75/rhoana_product/snemi/image/train-input.tif
            wget http://140.247.107.75/rhoana_product/snemi/seg/train-labels.tif
            wget http://140.247.107.75/rhoana_product/snemi/image/test-input.tif

    #. Store the data into `HDF5` format (take train-input.tif as example):
        .. code-block:: python

            import h5py
            import imageio

            train_image = imageio.volread('train-input.tif')

            fl = h5py.File('train_image.h5', 'w')
            fl.create_dataset('main', data=train_image)
            fl.close()

#. Run the training script:
    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3 python rain.py -t /path/to/snemi/
          -dn train_image.h5 -ln train_label.h5 -o outputs/unetv3 -lr 1e-03 \
          --iteration-total 100000 --iteration-save 10000 -mi 18,160,160 \
          -g 4 -c 4 -b 8 -ac unetv3

#. Visualize the training progress:
    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes:


#. Gnerate segmentation and run evaluation:
    #. Download the waterz package:
        .. code-block:: none
        
            $ git clone git@github.com:zudi-lin/waterz.git
            $ cd waterz
            $ pip install --editable . 