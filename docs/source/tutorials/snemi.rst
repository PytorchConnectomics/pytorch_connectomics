Neuron Segmentation
=======================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of instance segmentation.
The methodology is to first predict the affinity map of pixels with an encoder-decoder ConvNets and 
then generate the segmentation map using a segmentation algorithm (e.g. watershed). 

The evaluation of segmentation results is based on the `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_
and `Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.

.. note::
    Before running neuron segmentation, please take a look at the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demo>`_
    to get familiar with the datasets and have a sense of how the affinity graphs look like.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. The pytorch dataset class for neuron segmentation
is :class:`torch_connectomics.data.dataset.AffinityDataset`.

#. Get the dataset:

    #. Download the dataset from our server:

        .. code-block:: none

            wget http://hp03.mindhackers.org/rhoana_product/dataset/snemi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

    #. Store the data into ``HDF5`` format (take ``train-input.tif`` as example):

        .. code-block:: python
            :linenos:

            import h5py
            import imageio

            train_image = imageio.volread('train-input.tif')

            fl = h5py.File('train_image.h5', 'w')
            fl.create_dataset('main', data=train_image)
            fl.close()

#. Run the training script:

    .. code-block:: none

        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python rain.py -t /path/to/snemi/ \
          -dn train_image.h5 -ln train_label.h5 -o outputs/unetv3 -lr 1e-03 \
          --iteration-total 100000 --iteration-save 10000 -mi 18,160,160 \
          -g 4 -c 4 -b 8 -ac unetv3

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes:

    .. code-block:: none

        $ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test.py -t /path/to/snemi/ \
          -dn train_image.h5 -o outputs/unetv3/result -mi 18,160,160 -g 4 \
          -c 4 -b 8 -ac unetv3 -lm True -pm outputs/unetv3/volume_50000.pth


#. Gnerate segmentation and run evaluation:

    #. Download the ``waterz`` package:

        .. code-block:: none

            $ git clone git@github.com:zudi-lin/waterz.git
            $ cd waterz
            $ pip install --editable . 

    #. Download the ``zwatershed`` package:

        .. code-block:: none

            $ git clone git@github.com:zudi-lin/zwatershed.git
            $ cd zwatershed
            $ pip install --editable . 

    #. Generate 3D segmentation and report Rand and VI score using ``waterz``:

        .. code-block:: none

            $ python evaluation.py -pd /path/to/snemi/aff_pred.h5 -gt /path/to/snemi/seg_gt.h5 --mode 1

    #. You can also run the jupyter notebook `segmentation.ipynb <https://github.com/zudi-lin/pytorch_connectomics/blob/master/demo/segmentation.ipynb>`_ in 
       the demo, which provides more options and visualization.