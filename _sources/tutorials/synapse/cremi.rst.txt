CREMI (Synaptic Cleft Detection)
==================================

.. include:: _intro.rst

This tutorial reproduces synaptic cleft detection on the
`CREMI Challenge <https://cremi.org>`_ dataset using
``tutorials/syn_cremi.yaml``. The task is binary semantic segmentation
of cleft pixels with an anisotropic RSUNet, evaluated with the CREMI
distance metric and Jaccard.

Goal
----

The pipeline pins the following setup (encoded in
``tutorials/syn_cremi.yaml``):

- **Input** ``[18, 256, 256]`` patches at native CREMI resolution
  ``40 × 4 × 4`` nm.
- **Model** RSUNet with anisotropic down-sampling
  (``down_factors: [[1,2,2]] * 4``, ``depth_2d=1``,
  ``kernel_2d=[1,3,3]``), filters ``[32, 64, 96, 128, 160]``, batch
  norm, ELU activation. Single-channel cleft output.
- **Loss** ``WeightedBCEWithLogitsLoss`` (weight 1.0) plus
  ``DiceLoss`` (weight 1.0, sigmoid input).
- **Augmentation** flip + rotate + elastic + intensity (Gaussian
  noise, intensity shift, contrast) + EM-specific
  (``misalignment``, ``missing_section``, ``motion_blur``) — the
  full ``aug_em_*`` suite enabled inline.
- **Sampling** rejection sampling at ``size_thres=1000``, ``p=0.95``
  to focus on patches with synaptic clefts; preloaded cache for
  train and val.
- **Optimization** AdamW @ ``lr=1e-3``, ``weight_decay=0.01``,
  cosine to ``min_lr=1e-5`` over ``max_steps=150_000``,
  ``precision=bf16-mixed``, EMA enabled with ``decay=0.999`` and
  ``warmup_steps=500`` (``validate_with_ema=true``).
- **Inference** sliding window 18 × 256 × 256, 50 % overlap, bump
  blending, reflect padding; sigmoid activation on the cleft
  channel; TTA enabled with ``flip_axes: all``, ``ensemble_mode: mean``.
- **Metrics** ``cremi_distance`` and ``jaccard``.

1 - Get the data
^^^^^^^^^^^^^^^^

Download from the
`CREMI challenge page <https://cremi.org/>`_ or the Harvard RC mirror:

.. code-block:: bash

    mkdir -p datasets/corrected && cd datasets/corrected
    wget http://rhoana.rc.fas.harvard.edu/dataset/cremi.zip
    unzip cremi.zip

The config expects re-aligned, crack-corrected versions of the original
CREMI volumes:

.. code-block:: text

    datasets/corrected/
        im_A.h5,  im_B.h5,  im_C.h5         # training volumes
        syn_A.h5, syn_B.h5, syn_C.h5        # cleft labels
        im_A+.h5, im_B+.h5, im_C+.h5        # held-out test (no labels)

    .. note::

        The training pipeline reads the re-aligned volumes (``im_*.h5``
        without the ``+``); the ``+`` variants are the held-out
        challenge test set used only for submission. We perform
        re-alignment of the original CREMI image stacks and remove the
        crack artifacts. Reverse the alignment before submitting test
        predictions to the CREMI challenge.

2 - Run training
^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/syn_cremi.yaml

The config does not pin ``system.num_gpus``, so it inherits the
``all-gpu-cpu`` default. Override at the CLI if needed:

.. code-block:: bash

    python scripts/main.py --config tutorials/syn_cremi.yaml \
        system.num_gpus=4

Training schedule:

- **Step-based**: ``max_steps=150_000``, ``n_steps_per_epoch=1280``.
- Cosine LR over 150 k steps to ``min_lr=1e-5``.
- EMA shadow weights with ``decay=0.999``;
  ``validate_with_ema=true`` so checkpoint metrics are computed on the
  EMA model.
- ``checkpoint.monitor=train_loss_total_epoch``, ``save_top_k=3``,
  saved every 10 epochs.

Outputs land in
``outputs/rsunet_cremi_synapse_cleft/<timestamp>/``.

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard rsunet_cremi_synapse_cleft

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode:

.. code-block:: bash

    python scripts/main.py --config tutorials/syn_cremi.yaml \
        --mode test \
        --checkpoint outputs/rsunet_cremi_synapse_cleft/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference**. Sliding window 18 × 256 × 256, 50 % overlap, bump
   blending, reflect padding. TTA on (``flip_axes: all``,
   ``ensemble_mode: mean``) — predictions averaged across all flip
   variants. Sigmoid on the cleft channel. The raw probability map
   is saved as ``test_im_prediction.h5`` under
   ``outputs/rsunet_cremi_synapse_cleft/<timestamp>/results/``,
   ``save_dtype: float32``.
2. **Decoding**. The cleft mask is the binarized probability; no
   instance-level decoder is run for this task.
3. **Evaluation**. ``cremi_distance`` (the official CREMI cleft
   metric) and Jaccard against ``syn_*.h5``.

For CREMI challenge submission, switch ``test.data.test`` to point at
the held-out volumes (``im_A+.h5`` / ``im_B+.h5`` / ``im_C+.h5``) — the
config has the lines commented in for easy switching. The
``+`` volumes have no public labels, so ``evaluation.enabled`` should
be set to ``false`` for that run; the produced HDF5 should be
re-aligned back to the original CREMI coordinate system before
uploading.

4 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

- **Training loss** is the BCE + Dice sum; both terms drop together
  as the cleft mask becomes well-calibrated. EMA validation generally
  produces a slightly better Jaccard than non-EMA late in training.
- **Inference** on a 1250 × 1250 × 125 CREMI volume with TTA takes
  several minutes on a single A100/H100; without TTA it is roughly
  8× faster.
- **CREMI distance** is the headline metric for the challenge; under
  the canonical pin set it lands in the same ballpark as published
  RSUNet baselines after the full 150 k-step schedule completes.
