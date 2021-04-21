Data Loading
=============

Data Augmentation
------------------

Since many semi-supervised and unsupervised learning tasks do not require labels, the only key required in our 
data augmentor is ``'image'``. Let's look at an example for using an augmentation pipeline on input images:

.. code-block:: python

    from connectomics.data.augmentation import *
    tranforms = [
        Rescale(p=0.8),
        MisAlignment(displacement=16, 
                     rotate_ratio=0.5, 
                     p=0.5),
        CutBlur(length_ratio=0.6,
                down_ratio_min=4.0,
                down_ratio_max=8.0,
                p=0.7),
    ]
    augmentor = Compose(tranforms,
                        input_size = (8, 256, 256))
    
    sample = {'image': image}
    augmented = augmentor(sample)

Then the augmented data can be retrived using the corresponding key. Our augmentor can also apply the same set 
of transformations to the input images and all other specified targets. For example, under the supervised
segmentation setting, an label image/volume contains the segmentation masks and a valid mask indicating the
annotated regions are required. We provide the ``additional_targets`` option to handle those targets:

.. code-block:: python

    from connectomics.data.augmentation import *
    additional_targets = {'label': 'mask', 
                          'valid_mask': 'mask'}

    tranforms = [
        Rescale(p=0.8,
                additional_targets=additional_targets),
        MisAlignment(displacement=16, 
                     rotate_ratio=0.5, 
                     p=0.5,
                     additional_targets=additional_targets),
        CutBlur(length_ratio=0.6,
                down_ratio_min=4.0,
                down_ratio_max=8.0,
                p=0.7,
                additional_targets=additional_targets),
    ]
    augmentor = Compose(tranforms,
                        input_size = (8, 256, 256),
                        additional_targets=additional_targets)
    
    sample = {'image': image, 
              'label': label,
              'valid_mask': valid_mask}
    augmented = augmentor(sample)

.. tip::

    Each addition target need to be specified with a name (*e.g.*, ``'valid_mask'``) and a target type (``'img'`` or ``'mask'``). Some augmentations are only
    applied to ``'img'``, and augmentations for both ``'img'`` and ``'mask'`` will use different interpolation modes for them.

.. note::

    The ``'image'`` key in the examples above is to indicate the **name** of the sample, which means other keys can be used
    to retrive corresponding samples in augmentation. However, the ``'img'`` and ``'mask'`` values indicate the **type** of 
    a sample, therefore only the two values can be recognized by the augmentor.   

The ``'label'`` key in ``'mask'`` target type is used by default in the configuration file as most of the tutorial examples belong to the supervised 
training category. For model training with partially annotated dataset under the supervised setting, we need to add:

.. code-block:: yaml

    AUGMENTOR:
      ADDITIONAL_TARGETS_NAME: ['label', 'valid_mask']
      ADDITIONAL_TARGETS_TYPE: ['mask', 'mask']

Each transformation class is associated with an ``ENABLED`` key. To turn off a specific transformation (*e.g.*, mis-alignment), set:

.. code-block:: yaml

    AUGMENTOR:
      MISALIGNMENT: 
        ENABLED: False

Rejection Sampling
-------------------

Rejection sampling in the dataloader is applied for the following two purposes:

**1 - Adding more attention to sparse targets**

For some datasets/tasks, the foreground mask is sparse in the volume (*e.g.*, `synapse detection <../tutorials/synapse.html>`_). 
Therefore we perform reject sampling to decrease the ratio of (all completely avoid) regions without foreground pixels. 
Such a design lets the model pay more attention to the foreground pixels to alleviate false negatives (but may introduce
more false positives). There are two corresponding hyper-parameters in the configuration file:

.. code-block:: yaml

    DATASET:
        REJECT_SAMPLING:
        SIZE_THRES: 1000
        P: 0.95

The ``SIZE_THRES: 1000`` key-value pair means that if a random volume contains more than 1,000 non-background voxels, then
the volume is considered as a foreground volume and is returned by the rejection sampling function. If it contains less
than 1,000 voxels, the function will reject it with a probability ``P: 0.95`` and sample another volume. ``SIZE_THRES`` is
set to -1 by default to disable the rejection sampling.

**2 - Handling partially annotated data**

Some datasets are only partially labeled, and the unlabeled region should not be considered in loss calculation. In that case,
the user can specify the data path to the valid mask using the ``DATASET.VALID_MASK_NAME`` option. The valid mask volume should
be of the same shape as the label volume with non-zero values denoting annotated regions. A sampled volume with a valid ratio
less than 0.5 will be rejected by default.


TileDataset
------------

Large-scale volumetric datasets (*e.g.,* `MitoEM <https://mitoem.grand-challenge.org>`_) are usually stored as individual 
tiles (*i.e.*, 2D patches). Directly loading them as a single array into the memory for training and inference is infeasible. 
Therefore we designed the :class:`connectomics.data.dataset.TileDataset` class that reads the paths of the tiles and 
construct tractable chunks for processing. To use this dataset class, the user needs to prepare a `JSON` file which contains
the information of the dataset. An example for the MitoEM dataset can be 
found `here <https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/master/configs/MitoEM/im_train.json>`_.
