SNEMI3D
=========

.. include:: _intro.rst

This tutorial reproduces a DeepEM-style neuron segmentation result on the
SNEMI3D challenge dataset using
``tutorials/neuron_snemi/neuron_snemi.yaml``. It is a modernization of
the affinity-learning recipe from Lee et al. 2017 — same short-range
affinity target and waterz-based agglomeration, but a MedNeXt backbone
and current optimization and stability tricks.

References:

- Paper: `Superhuman Accuracy on the SNEMI3D Connectomics Challenge
  <https://arxiv.org/abs/1706.00120>`_ (Lee et al., 2017).
- Codebase: `seung-lab/DeepEM <https://github.com/seung-lab/DeepEM>`_.
- Config directory: ``tutorials/neuron_snemi/`` (see its ``README.md``
  for the variant table).

Goal
----

The pipeline pins the following setup for SNEMI3D:

- **Input** ``[32, 160, 160]`` patches, anisotropic spacing ``30 × 6 × 6``
  nm; pad ``[16, 80, 80]`` for symmetric inference context.
- **Model** MedNeXt-S, kernel size 3, 3D, with deep supervision.
- **Target** 12-channel affinity (``pipeline_profile: aff12``): three
  nearest-neighbor edges plus nine long-range auxiliaries, in
  ``affinity_mode: deepem`` convention with label ``erosion: 1``. At
  inference only channels 0-2 are decoded.
- **Split** DeepEM's own: top 80 z-slices train, bottom 20 validation.
- **Optimization** profile ``warmup_cosine_lr``, 200 epochs × 1000
  steps, batch 12 per GPU with ``accumulate_grad_batches=4``,
  ``bf16-mixed``, EMA (``decay=0.999``, validated with EMA weights).
- **Inference** sliding window 32 × 160 × 160, ``sw_batch_size=4``, 50 %
  overlap, bump blending, TTA on;
  ``crop_pad=[15, 16, 79, 80, 79, 80]`` puts the affinity output back on
  the original image support after padding and the ``deepem``
  destination-index shift.
- **Decoder** ``decoding_waterz`` template at ``thresholds=0.5``,
  ``merge_function=aff85_his256``, ``aff_threshold=[0.1, 0.999]``, with
  dust merge enabled.
- **Metric** Adapted Rand (``adapted_rand``).

Each of these is encoded directly in
``tutorials/neuron_snemi/neuron_snemi.yaml``; do not change them in
passing. Four sibling configs are provided for comparison:

- ``neuron_snemi_efficient.yaml`` — same recipe without deep supervision
  or gradient accumulation, on a 100 × 200-step schedule.
- ``neuron_snemi_v1.yaml`` — the efficient variant with an explicit
  AdamW ``lr=5e-4`` and stronger elastic / motion-blur augmentation.
- ``neuron_snemi_sdt.yaml`` — 9-channel affinity plus a skeleton-aware
  signed distance transform.
- ``neuron_snemi_sdt_multitask.yaml`` — the same target split across four
  heads with uncertainty loss balancing.

This page covers ``neuron_snemi.yaml`` only. No pretrained SNEMI3D
checkpoint ships with the tutorial, so training is a prerequisite for
the later steps.

1 - Get the data
^^^^^^^^^^^^^^^^^^

SNEMI3D is published on Zenodo as `record 7142003
<https://zenodo.org/records/7142003>`_ (DOI
`10.5281/zenodo.7142003 <https://doi.org/10.5281/zenodo.7142003>`_,
CC-BY-4.0, ``snemi.zip``, 185.6 MiB, md5
``3d25a7025f66698f33c7850ace885939``):

.. code-block:: bash

    mkdir -p datasets/SNEMI
    curl -L 'https://zenodo.org/records/7142003/files/snemi.zip?download=1' -o /tmp/snemi.zip
    unzip -j /tmp/snemi.zip -d datasets/SNEMI   # -j flattens the image/ and seg/ folders

That archive holds the three volumes the challenge released:

.. code-block:: text

    datasets/SNEMI/
        train-input.tif       # 100 slices, anisotropic 30 × 6 × 6 nm
        train-labels.tif      # dense neuron instance labels
        test-input.tif        # held-out volume

The challenge never published the **test** labels, so the Zenodo archive
alone cannot score ``--mode test`` or ``--mode tune``. The PyTC mirror
repacks the same three volumes (byte-identical) together with a
``test-labels.h5`` for offline evaluation, already flattened into the
layout the config expects:

.. code-block:: bash

    just download snemi          # 190.0 MiB from huggingface.co/pytc/tutorial

.. code-block:: text

    datasets/SNEMI/
        train-input.tif
        train-labels.tif
        test-input.tif
        test-labels.h5        # 100 × 1024 × 1024 uint16, 333 instances

The config reads from ``datasets/SNEMI/`` relative to the repo root.
Paths under ``train.data.train``, ``test.data.test``, and
``tune.data.val`` in ``neuron_snemi.yaml`` can be edited if you stage
data elsewhere.

2 - Run training
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml

The config sets ``system.profile: all-gpu-cpu``, so PyTC uses every
visible GPU. Override at the CLI if needed:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        system.num_gpus=4 data.dataloader.batch_size=6

Training schedule:

- **Epoch-based**: ``max_epochs=200``, ``n_steps_per_epoch=1000``, with
  ``accumulate_grad_batches=4`` on a per-GPU batch of 12.
- ``warmup_cosine_lr`` profile: linear warmup, then cosine decay;
  ``bf16-mixed`` precision, gradient clip 1.0, EMA weights used for
  validation.
- ``checkpoint.monitor=val_loss_total``, ``save_top_k=3`` — the
  bottom-20 z-slices of the training volume are held out as the
  validation split (``split_enabled: true``), so a real validation loss
  is available even though SNEMI3D has no public test labels.
- Image previews logged every 10 epochs.

Outputs land in ``outputs/neuron_snemi/<timestamp>/`` (the run base is
derived from the YAML stem, not from ``experiment_name``).

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard neuron_snemi

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode against the trained checkpoint. This
exercises inference, waterz decoding, and adapted-Rand evaluation
end-to-end:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        --mode test \
        --checkpoint outputs/neuron_snemi/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference** (``connectomics.inference.stage``). Sliding window
   32 × 160 × 160, ``sw_batch_size=4``, 50 % overlap, bump blending,
   symmetric pad of ``[16, 80, 80]``. Test-time augmentation is **on**
   by default: 16 unique variants (all-axis flips × 90° xy rotations)
   combined with ``ensemble_mode: min``. ``crop_pad`` puts the affinity
   back on the original image support. Saves the raw 12-channel
   affinity as ``test_im_prediction.h5`` in the checkpoint-derived
   output directory (see below).

2. **Decoding** (``connectomics.decoding.stage``). Selects the
   nearest-neighbor affinities (``select_channel: [0, 1, 2]``; the
   long-range edges are training auxiliaries only), then runs waterz
   with the DeepEM-style settings:

   - ``merge_function: aff85_his256``
   - ``aff_threshold: [0.1, 0.999]``
   - ``thresholds: 0.5``
   - ``channel_order: xyz``
   - dust merge ON (``dust_merge_size=800``,
     ``dust_merge_affinity=0.3``, ``dust_remove_size=600``)

3. **Evaluation** (``connectomics.evaluation.stage``). Computes
   Adapted Rand against ``datasets/SNEMI/test-labels.h5``.

The combined output (segmentation + metrics) lands in the checkpoint's
own run directory under ``test_<ckpt tag>/`` — for the command above,
``outputs/neuron_snemi/<timestamp>/test_last/``; a step checkpoint gives
``test_step=00050000/``. ``--mode tune`` writes to ``tune_<ckpt tag>/``
alongside it.

TTA is the dominant inference cost; disable it for a fast pass:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        --mode test --checkpoint <ckpt> \
        inference.test_time_augmentation.enabled=false

4 - Tune the decoder
^^^^^^^^^^^^^^^^^^^^^^

The waterz threshold and merge function dominate downstream Rand error.
``--mode tune`` runs an Optuna search with adapted Rand as the
objective:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        --mode tune \
        --checkpoint outputs/neuron_snemi/<timestamp>/checkpoints/last.ckpt

Configuration (under the ``tune:`` block):

- ``profile: tune_waterz`` (TPE sampler, study persisted as
  ``snemi_waterz_tuning``).
- 25 trials, 300 s timeout each.
- Search space:

  - ``merge_function`` ∈ ``{aff85_his256, aff75_his256, aff50_his256,
    aff25_his256, aff15_his256}``
  - ``thresholds`` ∈ ``[0.1, 0.9]`` step 0.1
  - ``aff_threshold[0]`` ∈ ``[0.0, 0.5]`` step 0.1
  - ``aff_threshold[1]`` ∈ ``[0.7, 1.0]`` step 0.1

The search reuses the same checkpoint and saved affinity; only the
decode + evaluate stages run per trial, so each trial is fast. Chain the
selected parameters into a test decode with ``--mode tune-test``.

SNEMI3D has no separate validation volume, so ``tune.data.val`` points
at the test volume — parameters are selected on the same data they are
reported on. Swap in the commented-out ``train-input.tif`` /
``train-labels.tif`` lines under ``tune.data.val`` for a clean split.

5 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals during reproduction:

- **Training loss** (``train_loss_total_epoch``) drops sharply through
  the warmup phase, then descends slowly through cosine decay;
  ``val_loss_total`` on the bottom-20 z-slices is the checkpoint
  selector.
- **Inference** is fast on SNEMI3D (a 100 × 1024 × 1024 volume) because
  of the small sliding-window grid, but the 16× TTA multiplies it:
  expect a couple of minutes on a single A100/H100 and roughly an order
  of magnitude more on an L40S.
- **Adapted Rand** is the headline number. The single best lever is
  ``thresholds`` followed by ``merge_function``; ``aff_threshold``
  boundaries matter mostly at low (<0.05) or high (>0.99) settings.

For the underlying mechanics (affinity learning, waterz post-processing
internals), see the
`DeepEM repository <https://github.com/seung-lab/DeepEM>`_ and the
`paper <https://arxiv.org/abs/1706.00120>`_.
