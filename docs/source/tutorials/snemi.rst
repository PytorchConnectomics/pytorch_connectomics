Neuron Segmentation
=======================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of instance segmentation.
The methodology is to first predict the affinity map of pixels with an encoder-decoder ConvNets and 
then generate the segmentation map using a segmentation algorithm (e.g., watershed). 

The evaluation of segmentation results is based on the `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_
and `Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.

.. note::
    Before running neuron segmentation, please take a look at the `demo <https://github.com/zudi-lin/pytorch_connectomics/tree/master/demo>`_
    to get familiar with the datasets and have a sense of how the affinity graphs look like.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. The pytorch target affinity generation is :class:`connectomics.data.utils.data_segmentation`.

#. Get the dataset:

        .. code-block:: none

            wget http://hp03.mindhackers.org/rhoana_product/dataset/snemi.zip
    
    For description of the data please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.


#. Run the training script:

    .. code-block:: none

        $ source activate py3_torch
        $ python scripts/train.py -i /{path-to-snemi}/ -o outputs/unetv3 -din train-input.tif -dln train-labels.tif \
            -lr 1e-03 --iteration-total 100000 --iteration-save 10000 \
            -mi 18,160,160 -ma unet_residual_3d -moc 3 \
            -to 2 -lo 1 -wo 1 -g 4 -c 4 -b 8 

    - Data: i/o/din/dln (input folder/output folder/train volume/train label)
    - Optimization: lr/iteration-total/iteration-save (learning rate/total #iterations/#iterations to save)
    - Model: mi/ma/moc (input size/architecture/#output channel)
    - Loss: to/lo/wo (target option/loss option/weight option)
    - System: g/c/b (#GPU/#CPU/batch size)

#. Visualize the training progress:

    .. code-block:: none

        $ tensorboard --logdir runs

#. Run inference on image volumes (min over 4-aug):

    .. code-block:: none

        $ python scripts/test.py -i /{path-to-snemi}/ \
          -din train-input.tif -mi 116,256,256 -g 4 -c 4 -b 4 \
          -ma unet_residual_3d -mpt outputs/unetv3/{log-folder}/volume_100000.pth -mpi 99999 -dp 8,64,64 -tam min -tan 4 


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
