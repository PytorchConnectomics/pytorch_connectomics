Neuron Segmentation
=====================

This tutorial provides step-by-step guidance for neuron segmentation with SENMI3D benchmark datasets.
Dense neuron segmentation in electronic microscopy (EM) images belongs to the category of **instance segmentation**.
The methodology is to first predict the affinity map (the connectivity of each pixel to neighboring pixels)
with an encoder-decoder ConvNets and then generate the segmentation map using a standard
segmentation algorithm (e.g., watershed).

The evaluation of segmentation results is based on the `Rand Index <https://en.wikipedia.org/wiki/Rand_index>`_
and `Variation of Information <https://en.wikipedia.org/wiki/Variation_of_information>`_.

.. tip::

    Before running neuron segmentation, please take a look at the `notebooks <https://github.com/zudi-lin/pytorch_connectomics/tree/master/notebooks>`_
    to get familiar with the datasets and available utility functions in this package.

The main script to run the training and inference is ``pytorch_connectomics/scripts/main.py``.
The pytorch target affinity generation is :class:`connectomics.data.dataset.VolumeDataset`.

Neighboring affinity learning
-------------------------------

The affinity value between two neighboring pixels (voxels) is 1 if they belong to the same instance and 0 if
they belong to different instances or at least one of them is a background pixel (voxel). An affinity map can
be regarded as a more informative version of boundary map as it contains the affinity to two directions in 2D inputs and
three directions (`z`, `y` and `x` axes) in 3D inputs.

.. figure:: ../_static/img/snemi_affinity.png
    :align: center
    :width: 800px

The figure above shows examples of EM images, segmentation and affinity map from the SNEMI3D dataset. Since the
3D affinity map has 3 channels, we can visualize them as RGB images.

1 - Get the data
^^^^^^^^^^^^^^^^^^

.. code-block:: none

    wget http://rhoana.rc.fas.harvard.edu/dataset/snemi.zip

For description of the SNEMI dataset please check `this page <https://vcg.github.io/newbie-wiki/build/html/data/data_em.html>`_.

.. note::

    Since for a region with dense masks, most affinity values are 1, in practice, we usually widen the instance border (erode the instance mask)
    to deal with the class imbalance problem and let the model make more conservative predictions to prevent merge error. This is done by
    setting ``MODEL.LABEL_EROSION = 1``.

2 - Run training
^^^^^^^^^^^^^^^^^^

Provide the ``yaml`` configuration files to run training:

.. code-block:: none

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1234 scripts/main.py --distributed \
    --config-base configs/SNEMI/SNEMI-Base.yaml \
    --config-file configs/SNEMI/SNEMI-Affinity-UNet.yaml

The configuration files for training can be found in ``configs/SNEMI/``.
We usually create a ``datasets/`` folder under ``pytorch_connectomics`` and put the SNEMI dataset there.
Please modify the following options according to your system configuration and data storage:

- ``IMAGE_NAME``: name of the 3D image file (HDF5 or TIFF)
- ``LABEL_NAME``: name of the 3D label file (HDF5 or TIFF)
- ``INPUT_PATH``: directory path to both input files above
- ``OUTPUT_PATH``: path to save outputs (checkpoints and Tensorboard events)
- ``NUM_GPUS``: number of GPUs
- ``NUM_CPUS``: number of CPU cores (for data loading)

.. tip::

    By default, we use multi-process distributed training with one GPU per process (and multiple CPUs for data loading).
    The model is wrapped with `DistributedDataParallel <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ (DDP).
    For more benefits of DDP, check `this tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_.
    Please note that official synchronized batch normalization (SyncBN) in PyTorch is only supported with DDP.

We also support `data parallel <https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html>`_ (DP) training.
If the training command above does not work for your system, please use:

.. code-block:: none

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -u scripts/main.py \
    --config-base configs/SNEMI/SNEMI-Base.yaml \
    --config-file configs/SNEMI/SNEMI-Affinity-UNet.yaml

DDP training is our default settings because features like automatic mixed-precision training and synchronized batch
normalization are better supported for DDP. Besides, DP usually has an imbalanced GPU memory usage.

3 - Run training with pretrained model (*optional*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

(*Optional*) To run training starting from pretrained weights, add a checkpoint file:

.. code-block:: none

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=1234 scripts/main.py --distributed \
    --config-base configs/SNEMI/SNEMI-Base.yaml \
    --config-file configs/SNEMI/SNEMI-Affinity-UNet.yaml \
    --checkpoint /path/to/checkpoint/checkpoint_xxxxx.pth.tar

4 - Visualize the training progress
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use Tensorboard to visualize the training process. Specify ``--logdir`` with your own experiment directory, which can be different
from the default one.

.. code-block:: none

    tensorboard --logdir outputs/SNEMI_UNet/

To visualize the training process and generate a **public link** to share the results with collaborators, we
use `tensorboard dev <https://tensorboard.dev/>`_. Similar to local visualization, we specify ``--logdir`` with the experiment
directory (which can be different from the default one).

.. code-block:: none

    tensorboard dev upload --logdir outputs/SNEMI_UNet/

Please refer this `example <https://colab.research.google.com/github/tensorflow/tensorboard/blob/master/docs/tbdev_getting_started.ipynb#scrollTo=oKW8V5chyx6e>`_ Google Colab
notebook for a step-by-step tutorial. Please also note that Tensorboard Dev `does not suppport <https://github.com/tensorflow/tensorboard/issues/3585/>`_ images
in the visualization with public link as of 12 October, 2021.

5 - Inference of affinity map
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run inference on image volumes (add ``--inference``). During inference the model can use larger batch sizes or take bigger inputs.
Test-time augmentation is also applied by default. We do not use distributed data-parallel during inference as the back-propagation
is not needed.

.. code-block:: none

    python -u scripts/main.py --config-base configs/SNEMI/SNEMI-Base.yaml \
    --config-file configs/SNEMI/SNEMI-Affinity-UNet.yaml --inference \
    --checkpoint outputs/SNEMI_UNet/checkpoint_100000.pth

6 - Get segmentation
^^^^^^^^^^^^^^^^^^^^^^

The last step is to generate segmentation (with external post processing packages) and run
evaluation. First download the ``waterz`` package:

.. code-block:: none

    git clone git@github.com:zudi-lin/waterz.git
    cd waterz
    pip install --editable .

Follow the instructions on the repo to install the ``waterz`` package. We will use the ``waterz.waterz`` API to generate segmentation from the affinity maps. The API takes in as arguments.

- ``Affinities``. This is the affinity map generated by our model. The values in the affinity map is expected to be between ``affinity_threshold[0]`` and ``affnity_threshold[1]``. It is recommended to keep the affinity values between 0 and 1 but the values in the affinity map prediicted by the model are between 0 and 255. Hence before using the affinity maps divide each value by 255.
- ``segmentation_thresholds``. This is an array of segmentation threshold values. Recommended values are ``[0.1,0.3,0.6]``. The API will produce a segmentation volume for each segmentation threshold in the array.
- ``merge_function``. The function that will be used while merging the nodes of the region adjacency graph. Recommended value for this parameter is  ``"aff50_his256"``.
- ``aff_threshold``. The values in the affinity maps will be constrained to lie between these thresholds. Recommended values are ``[0.05,0.995]``.
- ``gt``. This is the ground truth for the predicted affinity maps. This will be used to evaluate the segmentation result. If ground truth is not available, this parameter is supposed to be ``None``.If the ground truth is available, the API prints the ``Rand`` and ``VI`` score. 

.. code-block:: none

    import waterz
    import numpy as np

    # affinities is a [3,depth,height,width] numpy array of float32
    affinities = ... # model prediction

    affinities = affinities/255 
    # The affinity values in the model prediction are in the interval [0,255] and the affinity thresholds provided constraint them 
    # in the interval [0.05,0.995] hence we divide it by 255 in order to scale it.

    # evaluation: vi/rand
    seg_gt = None #segmentation ground truth. If available, the prediction is evaluated against this ground truth and Rand and VI scores are produced.

    aff_thresholds = [0.05, 0.995]
    seg_thresholds = [0.1, 0.3, 0.6]
    
    seg = waterz.waterz(affinities, seg_thresholds, merge_function='aff50_his256',                                
              aff_threshold=aff_thresholds, gt=seg_gt)

    # seg will be an array of shape [3,depth,height,width]. Since there are 3 segmentation thresholds, we get a result of shape 
    # [depth,height,width] for each threshold.
 
Download the ``zwatershed`` package (optional):

.. code-block:: none

    git clone git@github.com:zudi-lin/zwatershed.git
    cd zwatershed
    pip install --editable .

Generate 3D segmentation and report Rand and VI score using ``waterz``. Please see examples at `https://github.com/zudi-lin/waterz <https://github.com/zudi-lin/waterz>`_.

Long-range affinity learning
-------------------------------

ToDo