LICONN
========

.. include:: _intro.rst

This tutorial trains and evaluates the v3 (per-channel BCE + EMA,
label erosion=2, MedNeXt-L / kernel 3) affinity model on the **LICONN**
volume of the NISB neuron-segmentation benchmark. It reuses the same
6-channel affinity target and BANIS-style decoder as
:doc:`NISB <nisb>`, but points data paths at the LICONN split and
shows how to QC the raw prediction and tighten the decoder mask before
final NERL evaluation.

The driving config is ``tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml``,
which inherits everything from ``base_banis_v3_erosion2.yaml`` and only
overrides the LICONN data paths and the per-split ``affinity_mask_path``.

Goal
----

Reproduce the LICONN training/eval run that mirrors the upstream BANIS
``train_base_long`` schedule:

- 9 nm isotropic-XY input, ``[128, 128, 128]`` patches.
- 6-channel affinity target (short-range + long-range), BANIS
  source-voxel convention, ``edge_offset: 0``.
- MedNeXt-L, kernel size 3, per-channel ``BCEWithLogitsLoss``, no deep
  supervision, EMA validation (decay 0.999, warmup 500 steps), label
  erosion=2.
- AdamW @ ``lr=1e-3``, cosine to 0 over 200 000 steps,
  ``precision=16-mixed``, batch size 2 per GPU.
- Sliding-window inference (128³ windows, 50 % overlap,
  ``snap_to_edge``, distance-transform blending, ``output_dtype: float16``).
- Decoder: ``decode_affinity_cc`` (numba) on channels 0-2; threshold
  tuned with Optuna on ``val/seed5``.
- Metric: NERL against ``test/seed6/skeleton.pkl``.

1 - Get the data
^^^^^^^^^^^^^^^^^^

The LICONN split is staged on the lab cluster at:

.. code-block:: text

    /projects/weilab/dataset/nisb/liconn/
        train/seed{0..4}/data.zarr   (img, seg)
        val/seed5/data.zarr   +  skeleton.pkl
        test/seed6/data.zarr  +  skeleton.pkl

Each ``data.zarr`` is XYZ-ordered (same layout as the base NISB split);
the config keeps that convention end-to-end. ``skeleton.pkl`` is
required for NERL evaluation on both val (used by ``--mode tune``) and
test (used by ``--mode test``).

If you are working off-cluster, edit the ``data.train.path`` /
``data.val.path`` / ``data.test.path`` entries in
``liconn_banis_v3_erosion2.yaml`` (under the ``train:``, ``test:``, and
``tune:`` sections) to point at your local copy.

2 - Run training
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py \
        --config tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml

The training schedule (inherited via ``_base_:``) is step-based:

- ``max_steps: 200000``, ``n_steps_per_epoch: 5000``,
  ``val_check_interval: 5000`` — validation every 5 k steps.
- ``val_steps_per_epoch: 100`` (matches BANIS ``limit_val_batches=100``).
- Cosine LR over 200 k steps to ``min_lr=0``.
- Checkpoints saved every 50 k steps to
  ``outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/checkpoints/``.

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard nisb_liconn_banis_v3_erosion2

3 - Raw inference + default decode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run ``--mode test`` to invoke inference → decoding → evaluation back-to-back
on ``test/seed6``:

.. code-block:: bash

    python scripts/main.py \
        --config tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml \
        --mode test \
        --checkpoint outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/checkpoints/step=00200000.ckpt

What happens, in order:

1. **Inference** writes the raw 6-channel affinity to
   ``outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/test_step=00200000/seed6/raw_x1_ch0-1-2.h5``
   (``float16``, XYZ-ordered).
2. **Decoding** selects channels 0-2, applies the
   ``decoding.affinity_mask_path`` (see step 4) if present, and runs
   ``decode_affinity_cc`` (numba) at threshold 0.75.
3. **Evaluation** computes NERL against ``test/seed6/skeleton.pkl`` and
   writes ``eval_decoded_x1_ch0-1-2_*.txt`` next to the segmentation.

If you have not yet built the affinity mask, override that path to
``null`` for the first pass — the decoder will treat every voxel as
valid:

.. code-block:: bash

    python scripts/main.py \
        --config tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml \
        --mode test --checkpoint <ckpt> \
        test.decoding.affinity_mask_path=null

4 - Build an affinity mask
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LICONN slabs typically have a handful of bad z-planes at the top and
bottom of the cube (residual saturation / staining artifacts) that
poison the affinity prediction. A per-voxel mask drops those planes
and the image-border halo before decoding.

The two-step QC + mask workflow ships in ``dev/nisb/``:

.. code-block:: bash

    # 4a. Scan the saved prediction, decide low_z / high_z, write a report.
    python dev/nisb/check_aff.py \
        --pred outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/test_step=00200000/seed6/raw_x1_ch0-1-2.h5 \
        --img  /projects/weilab/dataset/nisb/liconn/test/seed6/data.zarr/img \
        --out-md   dev/nisb/check_aff_report_liconn_seed6.md \
        --mask-out outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/test_step=00200000/seed6/affinity_mask.h5

    # 4b. Build the affinity mask h5 from the report's frontmatter.
    python dev/nisb/build_aff_mask.py \
        --report dev/nisb/check_aff_report_liconn_seed6.md

``check_aff.py`` streams the (C=3, X, Y, Z) ``float16`` prediction one
z-slab at a time (peak RSS ~6 GB on an 83 GB volume) and writes a
markdown report whose frontmatter records ``img``, ``out``, ``low_z``,
``high_z``, ``bg_thresh``, and ``border_width``. ``build_aff_mask.py``
reads that frontmatter and writes the h5 mask at ``--out``.

Point ``test.decoding.affinity_mask_path`` at that file (the
LICONN config already does this for ``seed6``):

.. code-block:: yaml

    test:
      decoding:
        affinity_mask_path: outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/test_step=00200000/seed6/affinity_mask.h5

5 - Tune the decoder threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The decoder threshold is the most impactful single parameter. Run an
Optuna grid search on the held-out ``val/seed5`` skeleton:

.. code-block:: bash

    python scripts/main.py \
        --config tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml \
        --mode tune \
        --checkpoint outputs/nisb_liconn_banis_v3_erosion2/<timestamp>/checkpoints/step=00200000.ckpt

Configuration (inherited from ``base_banis_v3_erosion2.yaml`` via the
``tune:`` block):

- GridSampler over ``threshold ∈ {0.40, 0.46, …, 0.94}`` (step 0.06).
- Single objective: maximize ``nerl``.
- Study persisted as ``nisb_base_banis_cc_tuning`` with
  ``load_if_exists: true`` — re-running appends trials to the existing
  ``.db`` instead of starting over.

The tuner reuses any cached prediction h5 under
``tune_step=*/predictions/seed5/`` so a second pass skips inference and
only re-runs the decode + NERL step per threshold. If a previous run
was killed mid-trial (SLURM TIMEOUT), the in-progress threshold is
released back to the grid on resume.

6 - Apply the best threshold on test
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once tuning is done, re-run step 3 with the best threshold pulled from
the study:

.. code-block:: bash

    python scripts/main.py \
        --config tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml \
        --mode test --checkpoint <ckpt> \
        test.decoding.threshold=<best_threshold_from_tune>

Reference behavior
^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals during reproduction:

- ``raw_x1_ch0-1-2.h5`` on LICONN ``test/seed6`` is ~83 GB
  (compare ~53 GB on base NISB ``test/seed101``); the LICONN cube
  contains more non-zero predictions and compresses less.
- LICONN NERL at threshold 0.75 (default) on ``test/seed6`` is much
  lower (~4 %) than on base NISB; tune on ``val/seed5`` before
  reporting numbers. Expect the optimum to sit around 0.70.
- ``check_aff.py`` typically reports ``low_z=30``, ``high_z=2220`` on
  ``test/seed6`` — most of the cube is usable, but the border halo
  (``border_width=32``) and the first/last few z-planes are not.

For the underlying mechanics, see :doc:`nisb` — the same BANIS
affinity-then-decode pipeline applies; this page just specializes data
paths and adds the LICONN-specific affinity-mask step.
