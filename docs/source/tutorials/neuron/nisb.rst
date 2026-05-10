NISB / BANIS
==============

This tutorial reproduces the BANIS ``train_base_long`` neuron-segmentation
result on the NISB dataset using
``tutorials/neuron_nisb/base_banis.yaml``. The target is a 9 nm, 6-channel
affinity model (MedNeXt-S / kernel 3) trained for 200 k steps on
128-cube patches, decoded with affinity-threshold connected components,
and evaluated with the NERL skeleton metric.

Reference:

- Codebase: `StructuralNeurobiologyLab/banis
  <https://github.com/StructuralNeurobiologyLab/banis>`_ — the upstream
  BANIS pipeline this configuration mirrors. Affinity offset semantics
  (``affinity_mode: banis``, source-voxel storage, ``edge_offset: 0``),
  patched-inference geometry (128³ windows, 50 % overlap,
  ``snap_to_edge`` boundary handling, distance-transform blending), and
  the ``train_base_long`` schedule are all matched against this repo.

If you only need a different starting point, the same directory ships
several variants:

- ``base_banis.yaml`` — the canonical reproduction (this page).
- ``base_banis_v1.yaml`` / ``v2.yaml`` / ``v3.yaml`` — successive
  variations of the base config (different model size / target shape).
- ``*_erosion2.yaml`` — same configs with label erosion 2 to widen
  instance borders.
- ``base_banis_chunk.yaml`` / ``base_banis_crop.yaml`` — chunked /
  cropped data variants for low-memory machines.
- ``common.yaml`` — MedNeXt-B with affinity + signed distance transform
  at 40 nm; an alternative starting point, *not* the BANIS reproduction.

Goal
----

Match the BANIS ``train_base_long`` setup faithfully:

- 9 nm isotropic-XY input, ``[128, 128, 128]`` patches.
- 6-channel affinity target: short-range ``(1, 0, 0) / (0, 1, 0) / (0, 0, 1)``
  plus long-range at 10 voxels, in source-voxel BANIS convention.
- MedNeXt-S, kernel size 3, ``WeightedBCEWithLogitsLoss``, no deep
  supervision.
- AdamW @ ``lr=1e-3``, ``weight_decay=0.01``, cosine to ``min_lr=0``
  over 200 000 steps, ``precision=16-mixed``, ``batch_size=4`` per GPU.
- Sliding-window inference with 128-cube windows, 50 % overlap,
  distance-transform blending, snap-to-edge boundary handling. Output is
  ``float16``.
- Decoder: ``decode_affinity_cc`` (numba) at threshold 0.75, with the
  short-range affinities (channels 0-2) only.
- Metric: ``nerl`` against ``skeleton.pkl``.

Each of these is encoded directly in
``tutorials/neuron_nisb/base_banis.yaml``; do not change them in passing.

1 - Get the data
^^^^^^^^^^^^^^^^^^

The NISB ``base`` split is staged on the lab cluster at:

.. code-block:: text

    /projects/weilab/dataset/nisb/base/
        train/seed{0..N}/data.zarr
        val/seed100/data.zarr
        test/seed101/data.zarr  +  skeleton.pkl

Each ``data.zarr`` is XYZ-ordered (matching upstream BANIS); the config
keeps it that way and the decoder consumes channels in axis-0, axis-1,
axis-2 order. ``skeleton.pkl`` is required for NERL evaluation on
``test/seed101``.

If you are working off-cluster, edit the ``data.train.path`` /
``data.val.path`` / ``data.test.path`` entries in
``base_banis.yaml`` (under the ``train:``, ``test:``, and ``tune:``
sections) to point at your local copy.

2 - Run training
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml

The config sets ``system.num_gpus: -1`` and ``system.num_workers: -1``,
so PyTC will fan out across every visible GPU and use the auto-planned
worker count. Override at the CLI if needed:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
        system.num_gpus=4 data.dataloader.batch_size=2

The training schedule is **step-based**, not epoch-based:

- ``max_steps: 200000``, ``n_steps_per_epoch: 5000``,
  ``val_check_interval: 5000`` — validation every 5 k steps.
- ``val_steps_per_epoch: 100`` — matches BANIS
  ``limit_val_batches=100``.
- Cosine LR over 200 k steps to 0.
- Checkpoints saved every 50 k steps to
  ``outputs/nisb_base_banis/<timestamp>/checkpoints/``.

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard nisb_base_banis

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode with the trained checkpoint. This invokes
the inference, decoding, and evaluation stages back-to-back:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
        --mode test \
        --checkpoint outputs/nisb_base_banis/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference** (``connectomics.inference.stage``). Runs the BANIS-style
   sliding-window: 128³ windows, 50 % overlap, ``snap_to_edge``
   boundary handling, distance-transform blending, ``sw_device=cuda``,
   ``output_device=cpu``. Saves the raw 6-channel affinity as
   ``test_im_prediction.h5`` under
   ``outputs/nisb_base_banis/<timestamp>/results_step=<N>/``. Output
   dtype is ``float16``.
2. **Decoding** (``connectomics.decoding.stage``). Selects the
   short-range affinities (channels 0–2 in XYZ order), optionally masks
   them by ``affinity_mask_path``, then runs ``decode_affinity_cc`` via
   the numba backend at threshold 0.75 with ``edge_offset: 0`` (BANIS
   source-voxel convention).
3. **Evaluation** (``connectomics.evaluation.stage``). Computes NERL
   against ``test/seed101/skeleton.pkl`` and writes the metrics file
   alongside the segmentation.

If you do not yet have the affinity mask referenced under
``decoding.affinity_mask_path``, drop or override that line — the
mask is optional and the decoder will treat all voxels as valid:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
        --mode test --checkpoint <ckpt> \
        decoding.affinity_mask_path=null

4 - Tune the decoder threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The decoder threshold (0.75 in the canonical config) is the most
impactful single parameter. ``--mode tune`` runs an Optuna search on
the held-out *training* seed (``seed0``) using the ``cupy`` backend for
speed:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
        --mode tune \
        --checkpoint outputs/nisb_base_banis/<timestamp>/checkpoints/last.ckpt

Configuration (under the ``tune:`` block):

- TPE sampler with 4 startup trials, 4 total trials.
- Search space: ``threshold ∈ [0.4, 0.9]`` step 0.1.
- Single objective: maximize ``nerl``.
- Study persisted as ``nisb_base_banis_cc_tuning``;
  ``load_if_exists: true`` so subsequent runs append trials.

5 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals during reproduction:

- **Training loss** (``train_loss_total_epoch``) should drop steadily
  through the first ~50 k steps; cosine LR makes the late phase a
  long, slow refinement rather than another sharp drop.
- **Validation loss** is reported at 5 k-step intervals. The default
  checkpoint monitor is ``val_loss_total`` (mode ``min``), top-3 saved.
- **Inference speed** with ``sw_batch_size: 8``, 50 % overlap, and
  ``output_dtype: float16`` is the dominant cost; expect minutes per
  ``seed`` volume on a single A100/H100, hours on a single L40S.
- **Decode threshold** has a clean unimodal curve over ``[0.4, 0.9]``
  on validation NERL; the canonical 0.75 is a reasonable starting
  point but worth re-running ``--mode tune`` per checkpoint.

For the underlying mechanics (affinity learning, watershed-style
post-processing), see :doc:`snemi3d` — the same affinity-then-decode
pipeline applies.
