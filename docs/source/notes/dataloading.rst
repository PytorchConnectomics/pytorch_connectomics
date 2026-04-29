Data Loading
==============

Data Augmentation
------------------

PyTorch Connectomics uses MONAI dictionary transforms for augmentation. The common path is to
configure augmentations in YAML and let the Lightning data factory build the transform pipeline:

.. code-block:: python

    from connectomics.config import load_config
    from connectomics.data.augmentation import build_train_transforms

    cfg = load_config("tutorials/minimal.yaml")
    transforms = build_train_transforms(cfg, keys=["image", "label"], skip_loading=True)

    sample = {"image": image, "label": label}
    augmented = transforms(sample)

For custom pipelines, compose MONAI transforms with the connectomics-specific ``*d``
dictionary transforms:

.. code-block:: python

    from monai.transforms import Compose, RandFlipd
    from connectomics.data.augmentation import RandCutBlurd, RandMisAlignmentd

    transforms = Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandMisAlignmentd(keys=["image", "label"], prob=0.5, displacement=16),
        RandCutBlurd(keys=["image"], prob=0.7, length_ratio=0.6),
    ])

    sample = {"image": image, "label": label}
    augmented = transforms(sample)

The standard keys are ``image``, ``label``, ``label_aux``, and ``mask``. Spatial transforms
that receive multiple keys sample one random transform and apply it consistently to every
specified key.

Augmentations are configured under ``data.augmentation``:

.. code-block:: yaml

    default:
      data:
        augmentation:
          profile: aug_standard
          misalignment:
            enabled: true
            prob: 0.5
            displacement: 16

Each transform has an ``enabled`` flag. To turn off a specific transformation, set:

.. code-block:: yaml

    default:
      data:
        augmentation:
          misalignment:
            enabled: false

Rejection Sampling
-------------------

Rejection sampling in the dataloader is applied for the following two purposes:

**1 - Adding more attention to sparse targets**

For some datasets/tasks, the foreground mask is sparse in the volume (*e.g.*, `synapse detection <../tutorials/synapse.html>`_).
Therefore we perform reject sampling to decrease the ratio of (all completely avoid) regions without foreground pixels.
Such a design lets the model pay more attention to the foreground pixels to alleviate false negatives (but may introduce
more false positives). Configure rejection sampling under ``data.dataloader``:

.. code-block:: yaml

    default:
      data:
        dataloader:
          reject_sampling:
            size_thres: 1000
            p: 0.95

The ``size_thres: 1000`` key-value pair means that if a random volume contains more than 1,000 non-background voxels, then
the volume is considered as a foreground volume and is returned by the rejection sampling function. If it contains less
than 1,000 voxels, the function will reject it with a probability ``p: 0.95`` and sample another volume. ``size_thres`` is
set to -1 by default to disable the rejection sampling.

**2 - Handling partially annotated data**

Some datasets are only partially labeled, and the unlabeled region should not be considered in loss calculation. In that case,
the user can specify the data path to the valid mask using ``data.train.mask`` and ``data.val.mask``. The valid mask volume should
be of the same shape as the label volume with non-zero values denoting annotated regions. A sampled volume with a valid ratio
less than 0.5 will be rejected by default.


Filename and Lazy Datasets
--------------------------

The old ``TileDataset`` path has been removed. Large datasets now use one of the
current dataset implementations exported from :mod:`connectomics.data.datasets`:

- :class:`connectomics.data.datasets.CachedVolumeDataset` for volumes that fit in RAM.
- :class:`connectomics.data.datasets.LazyH5VolumeDataset` and
  :class:`connectomics.data.datasets.LazyZarrVolumeDataset` for crop-on-read HDF5/Zarr
  training without preloading the full volume.
- :class:`connectomics.data.datasets.MonaiFilenameDataset` for pre-tiled PNG/TIFF-style
  file lists.

For filename-based datasets, prepare a JSON file with image and label paths:

.. code-block:: python

    import json
    from pathlib import Path

    root = Path("path/to/dataset")
    n_images = 2000
    data_dict = {
        "base_path": str(root),
        "images": [f"images/im{idx:04d}.png" for idx in range(n_images)],
        "masks": [f"labels/seg{idx:04d}.png" for idx in range(n_images)],
    }

    js_path = "filename_dataset.json"
    with open(js_path, 'w') as fp:
        json.dump(data_dict, fp)

Then select the filename dataset in the Hydra config:

.. code-block:: yaml

    default:
      data:
        train:
          dataset_type: filename
          json: filename_dataset.json
          image_key: images
          label_key: masks
          split_ratio: 0.9

For large HDF5 or Zarr volumes, prefer lazy crop-on-read instead of file tiling:

.. code-block:: yaml

    default:
      data:
        dataloader:
          use_lazy_h5: true
          # or: use_lazy_zarr: true
          patch_size: [128, 128, 128]

The Lightning data factory chooses the concrete dataset from these config fields:

.. code-block:: python

    from connectomics.config import load_config
    from connectomics.training.lightning import create_datamodule

    cfg = load_config("tutorials/minimal.yaml")
    datamodule = create_datamodule(cfg)


Handling 2D Data
------------------

We design two ways to run inference for a trained 2D model. The first way is to directly load a 3D volume, but the inference
pipeline will predict each slice one-by-one and stack them back to a 3D volume. For representations depend on the dimension of
the inputs (*e.g.*, affinity map has three channels for 3D masks but only two channels for 2D masks), the number of output
channels is consistent with the 2D model. The second way is to directly load 2D PNG or TIFF images. Below are the configurations
for streaming 2D inputs at inference time:

.. code-block:: yaml

    test:
      data:
        test:
          dataset_type: filename
          json: datasets/test_files.json
        dataloader:
          patch_size: [1, 256, 256]

The filename JSON should list every input image:

.. code-block:: json

    {
      "base_path": "/data/test",
      "images": [
        "slice_0001.png",
        "slice_0002.png",
        "slice_0003.png",
        "slice_0004.png"
      ]
    }

The useful Linux command to list PNG images in a folder is:

.. code-block:: console

    ls -d $(pwd -P)/*.png > path.txt
