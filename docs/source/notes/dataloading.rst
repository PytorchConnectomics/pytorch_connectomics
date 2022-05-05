Data Loading
==============

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
construct tractable chunks for processing. To use this dataset class, the user needs to prepare a **JSON** file which contains
the information of the dataset. An example for the MitoEM dataset can be 
found `here <https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/master/configs/MitoEM/im_train.json>`_.
Below is a list of (incomplete) configurations exclusive for *TileDataset*:

.. code-block:: yaml

    DATASET:
      DO_CHUNK_TITLE: 1 # set to 1 to use TileDataset (default is 0)
      DATA_CHUNK_NUM: [2, 4, 4] # split the large volume into chunks
      DATA_CHUNK_ITER: 5000 # (training) number of iterations for a chunk

Suppose the input volume is of size (2000,6400,6400) in `(z,y,x)` order, setting ``DATASET.DATA_CHUNK_NUM = [2,4,4]`` will
split the `z` axis by 2 and `x` and `y` axes by 4, so that the process can handle (500,1600,1600) chunks sequentially, which 
is more manageable. The actual chunk size can be larger due to overlap sampling (only for training) and padding.

.. note::

    When using padding, the coordinate range of a chunk can have negative numbers, *e.g.*, ``[-4, 104, -64, 864, -64, 864]``, or
    numbers that are larger than the whole volume size, which is not an error. Those regions are padded so that the size of 
    sampled chunks stay unchanged.

Below is a Python snippet for creating the JSON file for a new dataset of size (2000,6400,6400), which are stored as 
2000 individual PNG images of size (6400,6400).

.. code-block:: python

    import json
    data_path = "path/to/images"
    n_images = 2000

    data_dict = {}
    data_dict["ndim"] = 1
    data_dict["dtype"] = "uint8"
    data_dict["image"] = [data_path + "im%04d.png" % idx for idx in range(n_images)]
    data_dict["height"] = 6400
    data_dict["width"] = 6400
    data_dict["depth"] = n_images
    data_dict["tile_ratio"] = 1
    data_dict["n_columns"] = 1
    data_dict["n_rows"] = 1
    data_dict["tile_st"] = [0,0]
    data_dict["tile_size"] = 6400

    js_path = 'tile_dataset.json'
    with open(js_path, 'w') as fp:
        json.dump(data_dict, fp)

Please note that the paths to **all** images are given as a list to ``data_dict["dtype"]``. For even larger datasets where
each slice is saved as multiple non-overlapping patches, ``data_dict["dtype"]`` is assumed to have the following format:

.. code-block:: json

    {
        "image": [
            "path/to/images/0000/{row}_{column}.png",
            "path/to/images/0001/{row}_{column}.png",
            "path/to/images/0002/{row}_{column}.png",
            ...
            "path/to/images/2000/{row}_{column}.png",
        ],
        "n_columns": 4,
        "n_rows": 4,
    }

Each slice uses a folder named by the *z* index. The name **{row}_{column}.png** in the JSON file is just a placeholder, 
and there is no need to give an exact input number. For the case above, each 2D slice is saved as 4x4 patches, so the real
images files in each *path/to/images/xxxx/* directory should be *0_0.png*, *1_0.png* until *3_3.png*.

Handling 2D Data
------------------

We design two ways to run inference for a trained 2D model. The first way is to directly load a 3D volume, but the inference
pipeline will predict each slice one-by-one and stack them back to a 3D volume. For representations depend on the dimension of
the inputs (*e.g.*, affinity map has three channels for 3D masks but only two channels for 2D masks), the number of output
channels is consistent with the 2D model. The second way is to directly load 2D PNG or TIFF images. Below are the configurations
for streaming 2D inputs at inference time:

.. code-block:: yaml

    DATASET:
      DO_2D: True # use 2d models
      LOAD_2D: True # load 2d images
    INFERENCE:
      IMAGE_NAME: datasets/test_path.txt
      IS_ABSOLUTE_PATH: True
      DO_SINGLY: True

Please note that the `test_path.txt` should be a list of absolute paths like the example below to avoid ambiguity:

.. code-block::

    /data/test/slice_0001.png
    /data/test/slice_0002.png
    /data/test/slice_0003.png
    ...
    /data/test/slice_0004.png

Additionally, ``INFERENCE.DO_SINGLY = True`` will let the pipeline process and save each input image separately, to
avoid loading all files into memory at the same time. The useful Linux command to get the absolute paths of all PNG
images in a folder is:

.. code-block:: console

    ls -d $(pwd -P)/*.png > path.txt
