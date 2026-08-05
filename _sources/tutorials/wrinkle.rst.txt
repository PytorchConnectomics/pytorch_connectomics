Wrinkle (2D Binary Segmentation)
==================================

This tutorial covers 2D binary segmentation of thin wrinkle / crease
structures from EM images using ``tutorials/wrinkle/common.yaml``. The
recipe is the v2 port of the v1 ``misc/Wrinkle-Deeplab-Binary-2D.yaml``
config.

    .. note::
        The v1 config used DeepLabv3c with a ResNet-101 backbone. The
        modern PyTC v2 architecture registry does not ship a DeepLab
        wrapper, so this port substitutes a 2D MONAI BasicUNet at the
        same input size (1 × 385 × 385). Loss, augmentation, and
        schedule are otherwise faithful. If wrinkle structures benefit
        from atrous-conv-style capacity, swap in a different backbone
        rather than tuning here.

Goal
----

The pipeline pins the following setup (encoded in
``tutorials/wrinkle/common.yaml``):

- **Input** ``[1, 385, 385]`` patches; 2D segmentation served through
  the 3D pipeline with depth 1.
- **Model** MONAI BasicUNet, filters ``[32, 64, 128, 256, 512]``, group
  normalization. Single-channel output.
- **Label** ``binary`` target with ``dilation: 3`` — the wrinkle
  annotations are single-voxel thin, so they are widened before being
  used as a target.
- **Loss** ``WeightedBCEWithLogitsLoss`` (weight 1.0) plus ``DiceLoss``
  (weight 0.5).
- **Augmentation** ``aug_light`` profile with ``smooth.enabled: false``
  — smoothing deforms thin wrinkle structures.
- **Sampling** rejection sampling at ``size_thres=100``, ``p=1.0`` to
  bias training toward patches that contain wrinkles (foreground is
  sparse).
- **Optimization** ``warmup_cosine_lr`` profile, AdamW @ ``lr=1e-3``,
  ``weight_decay=0.01``, 100 k steps × 1000 steps/epoch,
  ``precision=16-mixed``, gradient clip 1.0.
- **Inference** sliding window 1 × 385 × 385, 50 % overlap, bump
  blending, reflect padding; sigmoid on the output channel.
- **Metrics** ``jaccard`` and ``dice``.

1 - Get the data
^^^^^^^^^^^^^^^^

The dataset is a directory of paired image / label PNGs. The config
expects:

.. code-block:: text

    datasets/Wrinkle/
        train/
            images/**/*.png       # raw EM images
            wrinkles/**/*.png     # binary wrinkle masks
        test_path.txt             # list of test images (absolute or
                                  # relative paths)

Edit ``train.data.train.path`` / ``test.data.test.path`` in
``common.yaml`` if you stage data elsewhere.

2 - Run training
^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/wrinkle/common.yaml

The config sets ``system.profile: all-gpu-cpu``, so PyTC fans out
across every visible GPU. Override at the CLI if needed:

.. code-block:: bash

    python scripts/main.py --config tutorials/wrinkle/common.yaml \
        system.num_gpus=2 data.dataloader.batch_size=8

Training schedule:

- **Step-based**: 100 k optimizer steps, cosine decay after warmup.
- ``checkpoint.monitor=train_loss_total_epoch``, ``save_top_k=3``.
- Outputs land in ``outputs/wrinkle_unet2d/<timestamp>/``.

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard wrinkle_unet2d

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode:

.. code-block:: bash

    python scripts/main.py --config tutorials/wrinkle/common.yaml \
        --mode test \
        --checkpoint outputs/wrinkle_unet2d/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference**. Sliding window 1 × 385 × 385, 50 % overlap, bump
   blending, reflect padding. Sigmoid on the cleft / wrinkle channel.
   Saves the raw probability map as ``test_im_prediction.h5`` under
   ``outputs/wrinkle_unet2d/<timestamp>/test/``.
2. **Decoding**. The wrinkle mask is the binarized probability; no
   instance-level decoder is run for this task.
3. **Evaluation**. Jaccard and Dice against the ground-truth wrinkle
   masks listed in ``test_path.txt``.

4 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals:

- **Training loss** is the BCE + Dice sum; both terms drop together as
  the foreground mask becomes well-calibrated. The label dilation
  hides single-voxel labels, so the early loss is dominated by the
  Dice term until the model learns the dilated structure.
- **Inference** is fast on 2D inputs; expect well under a minute per
  large image on a single A100/H100.
- **Jaccard / Dice** is the headline metric. The original DeepLab
  baseline on this dataset achieved ~0.7 Dice; the BasicUNet
  substitute should land in a similar range under the canonical
  schedule. If the gap is material, the v2 follow-up is to register a
  DeepLab-style architecture or swap to a transformer backbone.
