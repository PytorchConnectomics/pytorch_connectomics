Synaptic Cleft Detection
=======================

This tutorial provides step-by-step guidance for synaptic cleft detection with `CREMI<https://cremi.org>`_ benchmark datasets.
We consider the task as a semantic segmentation task and predict the synapse pixels with encoder-decoder ConvNets similar to
the models used in affinity prediction in 'neuron segmentation<https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html>'_. 
The evaluation of the synapse detection results is based on the F1 score and average distance. See `CREMI metrics<https://cremi.org/metrics/>`_
for more details.

.. note::
    We preform re-alignment of the original CREMI image stacks. Please reverse the alignment before submitting the test 
    prediction to the CREMI challenge.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/scripts/``. Need to pass the argument ``--task 1``
when executing the ``train.py`` and ``test.py`` scripts. The pytorch dataset class of synapses is :class:`torch_connectomics.data.dataset.SynapseDataset`.

#. Get the dataset:

    #. Download the original images from our server:

        .. code-block:: none

            wget http://140.247.107.75/rhoana_product/snemi/image/train-input.tif
            wget http://140.247.107.75/rhoana_product/snemi/seg/train-labels.tif
            wget http://140.247.107.75/rhoana_product/snemi/image/test-input.tif

    #. Store the data into ``HDF5`` format (take train-input.tif as example):

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