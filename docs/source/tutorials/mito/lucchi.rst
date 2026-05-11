Lucchi++ (Semantic Segmentation)
==================================

.. include:: _intro.rst

This tutorial reproduces binary mitochondria segmentation on the Lucchi++
EM benchmark using ``tutorials/mito_lucchi++.yaml``. The task is treated
as **semantic segmentation** — predict the mitochondria foreground mask
with an encoder-decoder network. Evaluation is the Jaccard / IoU score.

The dataset was released by Lucchi et al. and is isotropic at 5 nm
across all three axes, so the recipe uses a fully-3D MedNeXt with
isotropic 112³ patches.

Goal
----

The pipeline pins the following setup:

- **Input** ``[112, 112, 112]`` patches, isotropic 5 × 5 × 5 nm.
- **Model** MedNeXt-S, kernel size 3, 3D, no deep supervision.
- **Pipeline** ``pipeline_profile: binary`` (single foreground channel).
- **Dataloader** cached profile, batch size 8, ``aug_strong``
  augmentation profile.
- **Optimization** ``warmup_cosine_lr`` profile, AdamW @
  ``lr=1e-3``, ``weight_decay=0.01``, 150 epochs × 1000 steps,
  ``precision=16-mixed``, gradient clip 1.0.
- **Inference** sliding window 112³ with 50 % overlap, bump blending,
  ``sw_batch_size=8``, TTA enabled with all-axis flips.
- **Metric** ``jaccard``.

Each of these is encoded directly in ``tutorials/mito_lucchi++.yaml``;
do not change them in passing.

1 - Get the data
^^^^^^^^^^^^^^^^

Lucchi++ is the relabeled version of the original Lucchi 2012 dataset
released by Casser et al.; download from the EPFL CVLab page or your
local mirror. After unpacking you should have HDF5 volumes:

.. code-block:: text

    datasets/lucchi++/
        train_im.h5
        train_mito.h5
        test_im.h5
        test_mito.h5

The config reads from ``datasets/lucchi++/`` relative to the repo
root. Edit the ``train.data.train`` and ``test.data.test`` blocks in
``tutorials/mito_lucchi++.yaml`` if you stage data elsewhere.

For the upstream description see the
`EPFL CVLab page <https://www.epfl.ch/labs/cvlab/data/data-em/>`_.

2 - Run training
^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/mito_lucchi++.yaml

The config sets ``system.profile: all-gpu-cpu``, so PyTC fans out
across every visible GPU. Override at the CLI if needed:

.. code-block:: bash

    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        system.num_gpus=4 data.dataloader.batch_size=4

Training schedule:

- **Epoch-based**: 150 epochs × 1000 steps = 150 k optimizer steps.
- ``warmup_cosine_lr`` profile: linear warmup, then cosine decay.
- ``checkpoint.monitor=train_loss_total_epoch`` (no held-out
  validation split — Lucchi++ is small and the public test split is
  used for final reporting).
- Image previews logged every 10 epochs to TensorBoard.

Outputs land in ``outputs/mito_lucchi++/<timestamp>/`` (the
``save_path`` baked into ``train.monitor.checkpoint``).

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard mito_lucchi++

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode against the trained checkpoint:

.. code-block:: bash

    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        --mode test \
        --checkpoint outputs/mito_lucchi++/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference**. Sliding window 112³ with 50 % overlap, bump
   blending, ``sw_batch_size=8``. TTA is on by default
   (``flip_axes: all``), so Lucchi++ is predicted with 8× flip
   augmentations averaged. Saves the raw foreground probability as
   ``test_im_prediction.h5`` in
   ``outputs/mito_lucchi++/<timestamp>/results_step=<N>/``.
2. **Decoding**. The ``binary`` pipeline profile keeps the
   probability map without further post-processing (foreground mask is
   thresholded inside evaluation).
3. **Evaluation**. Jaccard / IoU against
   ``datasets/lucchi++/test_mito.h5``; the result is written next to
   the prediction.

To disable TTA (faster but slightly weaker), override:

.. code-block:: bash

    python scripts/main.py --config tutorials/mito_lucchi++.yaml \
        --mode test --checkpoint <ckpt> \
        inference.test_time_augmentation.enabled=false

4 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals:

- **Training loss** drops sharply through the warmup (~5 epochs), then
  descends slowly through cosine decay. With MedNeXt-S on 112³
  isotropic patches the loss usually plateaus after epoch ~80.
- **Inference** is fast on Lucchi++ (165 × 1024 × 768 test volume)
  with TTA: tens of seconds on an A100/H100, low single-digit minutes
  on an L40S.
- **Jaccard / IoU** lands in the same ballpark as the published
  benchmarks for this dataset; the dominant lever beyond training
  duration is whether TTA is enabled.

For a multi-task variant that adds a signed distance transform head,
see the sibling configs under ``tutorials/`` (``mito_betaseg.yaml`` and
``mito_betaseg_banis_v{0,1,2}.yaml`` use this style on a different
dataset).
