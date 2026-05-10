SNEMI3D
=========

This tutorial reproduces the DeepEM-style neuron segmentation result on
the SNEMI3D challenge dataset using
``tutorials/neuron_snemi/neuron_snemi.yaml``. It is a modernization of
the affinity-learning recipe from Lee et al. 2017, with current
optimization and stability tricks but the same short-range affinity
target and waterz-based agglomeration.

References:

- Paper: `Superhuman Accuracy on the SNEMI3D Connectomics Challenge
  <https://arxiv.org/abs/1706.00120>`_ (Lee et al., 2017).
- Codebase: `seung-lab/DeepEM <https://github.com/seung-lab/DeepEM>`_.

Goal
----

The pipeline pins the following BANIS-equivalent setup for SNEMI3D:

- **Input** ``[16, 224, 224]`` patches, anisotropic spacing ``30 × 6 × 6``
  nm; pad ``[8, 128, 128]`` for symmetric inference context.
- **Model** RSUNet (Recursive Symmetric UNet, the DeepEM architecture).
- **Target** 12-channel affinity (``aff12``): short-range plus long-range
  (the ``pipeline_profile: aff12`` in ``all_profiles.yaml``). At inference
  we keep only channels 0-2 (axis-0/1/2 short-range) for waterz.
- **Optimization** profile ``warmup_cosine_lr``, 100 epochs × 1000
  steps/epoch.
- **Inference** sliding window 16 × 224 × 224, ``sw_batch_size=16``;
  ``crop_pad=[7, 8, 127, 128, 127, 128]`` puts the affinity output back on
  the original image support after symmetric padding.
- **Decoder** ``decoding_waterz`` template at ``thresholds=0.5``,
  ``merge_function=aff85_his256``, ``aff_threshold=[0.1, 0.999]``, plus
  dust merge / best-buddy / one-sided post-processing — the standard
  DeepEM-style agglomerative watershed.
- **Metric** Adapted Rand (``adapted_rand``).

Each of these is encoded directly in
``tutorials/neuron_snemi/neuron_snemi.yaml``; do not change them in
passing. Two sibling configs are also provided for comparison:

- ``neuron_snemi_sdt.yaml`` — affinity + signed distance transform.
- ``neuron_snemi_sdt_multitask.yaml`` — joint multi-task variant.

This page covers ``neuron_snemi.yaml`` only.

1 - Get the data
^^^^^^^^^^^^^^^^^^

The challenge data is available from the
`SNEMI3D challenge page <http://brainiac2.mit.edu/SNEMI3D/>`_ or the
Harvard RC mirror:

.. code-block:: bash

    mkdir -p datasets/SNEMI && cd datasets/SNEMI
    wget http://rhoana.rc.fas.harvard.edu/dataset/snemi.zip
    unzip snemi.zip

After unpacking, you should have:

.. code-block:: text

    datasets/SNEMI/
        train-input.tif       # 100 slices, anisotropic 30 × 6 × 6 nm
        train-labels.tif      # dense neuron instance labels
        test-input.tif        # held-out volume (no public labels)
        test-labels.h5        # provided locally for offline evaluation

The config reads from ``datasets/SNEMI/`` relative to the repo root.
Paths under ``train.data.train`` and ``test.data.test`` in
``neuron_snemi.yaml`` can be edited if you stage data elsewhere.

2 - Run training
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml

The config sets ``system.profile: all-gpu-cpu``, so PyTC uses every
visible GPU. Override at the CLI if needed:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        system.num_gpus=4 data.dataloader.batch_size=2

Training schedule:

- **Epoch-based**: ``max_epochs=100``, ``n_steps_per_epoch=1000`` →
  100 k optimizer steps total.
- ``warmup_cosine_lr`` profile: linear warmup, then cosine decay.
- ``checkpoint.monitor=train_loss_total_epoch``, ``save_top_k=3`` (no
  validation loss is monitored — SNEMI3D has no public test labels and
  the training labels are dense, so the recipe reports the
  best-train-loss epochs rather than holding out a validation split).
- Image previews logged every 10 epochs.

Monitor with TensorBoard:

.. code-block:: bash

    just tensorboard rsunet_snemi_lee2017_modern

The output directory is keyed off ``experiment_name``, so you'll see
``outputs/rsunet_snemi_lee2017_modern/<timestamp>/...``.

3 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the combined ``test`` mode against the trained checkpoint. This
exercises inference, waterz decoding, and adapted-Rand evaluation
end-to-end:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        --mode test \
        --checkpoint outputs/rsunet_snemi_lee2017_modern/<timestamp>/checkpoints/last.ckpt

What happens, in order:

1. **Inference** (``connectomics.inference.stage``). Sliding window
   16 × 224 × 224, ``sw_batch_size=16``, symmetric pad of
   ``[8, 128, 128]`` (cropped back via ``crop_pad`` after prediction so
   the saved affinity occupies the original image support). Model runs
   on GPU; default activations from the model wrapper. Saves the raw
   12-channel affinity as ``test_im_prediction.h5`` in
   ``outputs/.../results_step=<N>/``.

2. **Decoding** (``connectomics.decoding.stage``). Selects the
   short-range affinities (channels 0-2; ``aff=1`` neighborhood), then
   runs waterz with the DeepEM-style settings:

   - ``merge_function: aff85_his256``
   - ``aff_threshold: [0.1, 0.999]``
   - ``thresholds: 0.5``
   - dust merge ON (``dust_merge_size=800``,
     ``dust_merge_affinity=0.3``, ``dust_remove_size=600``)
   - best-buddy on, ``one_sided_threshold=0.8``,
     ``one_sided_min_size=100``

3. **Evaluation** (``connectomics.evaluation.stage``). Computes
   Adapted Rand against ``datasets/SNEMI/test-labels.h5``.

The combined output (segmentation + metrics) lands under
``outputs/.../results_step=<N>/``.

To switch to the long-range affinity selection (``aff=3`` in DeepEM),
override at the CLI:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        --mode test --checkpoint <ckpt> \
        inference.model.select_channel='[6, 9, 4]'

Test-time augmentation (8× via flips + 90° rotations in xy) is
disabled by default in the config; flip
``inference.test_time_augmentation.enabled=true`` for the
``patch_first_local`` flow used by DeepEM.

4 - Tune the decoder
^^^^^^^^^^^^^^^^^^^^^^

The waterz threshold and merge function dominate downstream Rand error.
``--mode tune`` runs an Optuna search on the ``test`` volume (since
SNEMI3D has no separate validation volume) with adapted Rand as the
objective:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_snemi/neuron_snemi.yaml \
        --mode tune \
        --checkpoint outputs/rsunet_snemi_lee2017_modern/<timestamp>/checkpoints/last.ckpt

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
decode + evaluate stages run per trial, so each trial is fast.

5 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

A few sanity-check signals during reproduction:

- **Training loss** (``train_loss_total_epoch``) drops sharply through
  the warmup phase, then descends slowly through cosine decay. With
  RSUNet on the 12-channel affinity target it usually plateaus by
  epoch ~60.
- **Inference** is fast on SNEMI3D (a 100×1024×1024 volume) because of
  the small sliding-window grid; expect well under a minute per
  inference on a single A100/H100, low single-digit minutes on an
  L40S.
- **Adapted Rand** is the headline number; under the canonical pin
  set it should land in the same range as the DeepEM paper after
  threshold tuning. The single best lever is ``thresholds`` followed
  by ``merge_function``; ``aff_threshold`` boundaries matter mostly at
  low (<0.05) or high (>0.99) settings.

For the underlying mechanics (affinity learning, waterz post-processing
internals), see the
`DeepEM repository <https://github.com/seung-lab/DeepEM>`_ and the
`paper <https://arxiv.org/abs/1706.00120>`_.
