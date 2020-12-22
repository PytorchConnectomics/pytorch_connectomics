Data Loading
=============

.. contents::
   :local:

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

.. note::

    Each addition target need to be specified with a name (*e.g.*, ``'valid_mask'``) and a target type (``'img'`` or ``'mask'``). Some augmentations are only
    applied to ``'img'``, and augmentations for both ``'img'`` and ``'mask'`` will use different interpolation modes for them.

The ``'label'`` key in ``'mask'`` target type is used by default in the configuration file as most of the tutorial examples belong to the supervised 
training category. For model training with partially annotated dataset under the supervised setting, we need to add:

.. code-block:: yaml

    AUGMENTOR:
      ADDITIONAL_TARGETS_NAME: ['label', 'valid_mask']
      ADDITIONAL_TARGETS_TYPE: ['mask', 'mask']

Rejection Sampling
-------------------

TileDataset
------------