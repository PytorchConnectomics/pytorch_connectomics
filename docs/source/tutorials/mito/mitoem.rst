MitoEM (Instance Segmentation)
================================

This tutorial reproduces 3D mitochondria **instance segmentation** on
the
`MitoEM <https://donglaiw.github.io/page/mitoEM/index.html>`_ dataset
released by
`Wei et al. <https://donglaiw.github.io/paper/2020_miccai_mitoEM.pdf>`__
in 2020. The recipe lives under ``tutorials/mitoEM/`` with two
dataset-specific entry points (``R.yaml`` for MitoEM-Rat,
``H.yaml`` for MitoEM-Human, ``HR.yaml`` for joint training) sharing
``common.yaml``.

The pipeline is multi-task: predict short-range affinity, long-range
affinity (radius 5), and a skeleton-aware EDT head, then decode with a
distance-watershed step. Evaluation uses Adapted Rand and Variation of
Information.

Goal
----

The pipeline pins the following setup (encoded in
``tutorials/mitoEM/common.yaml`` and inherited by ``R.yaml`` /
``H.yaml`` / ``HR.yaml``):

- **Input** ``[32, 256, 256]`` patches at native MitoEM resolution
  ``30 × 8 × 8`` nm.
- **Model** MedNeXt-M, kernel size 3, ``checkpoint_style:
  outside_block``, three output heads:

  - ``aff_r1`` — 3-channel short-range affinity at offsets
    ``(0, 0, 1) / (0, 1, 0) / (1, 0, 0)``;
  - ``aff_r5`` — 3-channel long-range affinity at offsets
    ``(0, 0, 5) / (0, 5, 0) / (5, 0, 0)``;
  - ``sdt`` — 1-channel skeleton-aware EDT head.

- **Loss** per-channel BCE on each affinity head plus a SmoothL1
  (``tanh: true``) on the EDT head, balanced by ``uncertainty``
  loss-balancing.
- **Augmentation** ``aug_em_neuron_fast`` profile with rotations on
  all three axes.
- **Optimization** ``warmup_cosine_lr`` profile, 200 epochs × 1000
  steps, ``accumulate_grad_batches=4``, ``precision=bf16-mixed``.
- **Inference** sliding window 32 × 256 × 256, ``sw_batch_size=1``,
  50 % overlap, bump blending, replicate-padding mode; head set to
  ``aff_r1`` for the saved primary output.
- **Decoder** ``decode_distance_watershed`` over the EDT channel
  (``distance_channels=[6]``, ``distance_threshold=[0.5, 0]``,
  ``min_seed_size=100``, ``min_instance_size=50``).
- **Metric** ``adapted_rand`` + ``voi``.

1 - Get the data
^^^^^^^^^^^^^^^^

The MitoEM dataset is publicly available at the
`project page <https://donglaiw.github.io/page/mitoEM/index.html>`_ and
the `MitoEM Challenge <https://mitoem.grand-challenge.org/>`_. On the
lab cluster it is staged at:

.. code-block:: text

    /projects/weilab/dataset/mito/mitoEM/
        EM30-R/                  # rat
            im_train.h5,  mito_train-v2.h5
            im_val.h5,    mito_val-v2.h5
            im_test.h5,   mito_test-v2.h5
        EM30-H/                  # human
            (same layout)

Each split is a 4096 × 4096 × {400|100|500} HDF5 stack at
30 × 8 × 8 nm. The ``train.data.root_path`` field in
``common.yaml`` points at this directory; override at the CLI if you
stage data elsewhere.

The test labels for MitoEM challenge submission are not publicly
released; ``mito_test-v2.h5`` here refers to the locally maintained
v2 labels for offline development.

2 - Run training
^^^^^^^^^^^^^^^^

Pick the dataset variant and run:

.. code-block:: bash

    conda activate pytc

    # MitoEM-Rat
    python scripts/main.py --config tutorials/mitoEM/R.yaml

    # MitoEM-Human
    python scripts/main.py --config tutorials/mitoEM/H.yaml

    # Joint (rat + human in the same training run)
    python scripts/main.py --config tutorials/mitoEM/HR.yaml

The config sets ``system.num_gpus: -1`` and ``system.num_workers: -1``,
so PyTC fans out across every visible GPU.

Training schedule:

- ``max_epochs=200``, ``n_steps_per_epoch=1000`` → 200 k optimizer
  steps total.
- ``accumulate_grad_batches=4`` with ``batch_size=1`` per GPU →
  effective batch size 4 × num_gpus.
- ``checkpoint.monitor=val_loss_total`` with ``mode=min``,
  ``save_top_k=3``.
- Image previews on the ``aff_r1`` head every 10 epochs.

Outputs land in
``outputs/mitoem30{r,h,hr}_mednext_sdt_multitask/<timestamp>/``.

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard mitoem30r_mednext_sdt_multitask

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode:

.. code-block:: bash

    python scripts/main.py --config tutorials/mitoEM/R.yaml \
        --mode test \
        --checkpoint outputs/mitoem30r_mednext_sdt_multitask/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference**. Sliding window 32 × 256 × 256, 50 % overlap, bump
   blending, ``padding_mode=replicate``. The primary head ``aff_r1``
   is selected at save time, and per-channel sigmoid is applied. The
   raw 7-channel multi-head prediction is saved as
   ``test_im_prediction.h5``.
2. **Decoding**. ``decode_distance_watershed`` runs on the EDT
   channel (channel 6), seeded at distance > 0.5, growing until the
   distance hits 0, with seeds < 100 voxels and instances < 50 voxels
   filtered out. The fast EDT path is enabled with
   ``edt_parallel=8``.
3. **Evaluation**. Adapted Rand and Variation of Information against
   the test labels; written next to the segmentation.

To swap in the ``aff_r5`` head as the primary inference output:

.. code-block:: bash

    python scripts/main.py --config tutorials/mitoEM/R.yaml \
        --mode test --checkpoint <ckpt> \
        inference.model.head=aff_r5

4 - Submitting to the MitoEM Challenge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Grand Challenge accepts segmentation HDF5 volumes. After ``--mode
test`` produces the segmentation under ``outputs/.../results_step=<N>/``,
follow the formatting rules at
https://mitoem.grand-challenge.org/ and submit. Performance on the
challenge test split is only computable on the Grand Challenge website
because public ground truth is not released for that split.

Per-volume offline evaluation on the **validation** split (provided in
``EM30-{R,H}/mito_val-v2.h5``) uses the same ``adapted_rand + voi``
metrics described above; just point ``test.data.test`` at the val
volumes.

5 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals:

- **Training loss** has three components (``aff_r1``, ``aff_r5``,
  ``sdt``) and uncertainty-balanced weights. The
  ``train_loss_term_*_weighted`` scalars logged in TensorBoard are the
  most informative — uncertainty balancing typically pushes the
  ``aff_r5`` term down faster than ``aff_r1`` because the long-range
  task is harder.
- **Validation loss** is checked at every epoch boundary; the
  best-3 checkpoints by ``val_loss_total`` are kept.
- **Inference** on the 4096 × 4096 × 500 test volume is the dominant
  cost; expect roughly 1-2 hours on a single A100/H100 with
  ``sw_batch_size=1``.
- **Decoder threshold** (``distance_threshold[0]``) is the primary
  knob for over- / under-segmentation. The default 0.5 is a
  reasonable starting point; lower (e.g. 0.3) yields more seeds.
- **Adapted Rand** below ~0.05 on the validation split is in the
  ballpark of the published MitoEM-Rat baseline. The challenge uses
  AP-75 (average precision at IoU 0.75), which is computed by the
  Grand Challenge submission system.
