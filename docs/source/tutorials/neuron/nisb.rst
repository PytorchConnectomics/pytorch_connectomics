NISB / BANIS
==============

The NISB benchmark (sometimes referred to as **BANIS** in the configs) is a
larger, anisotropic neuron-segmentation dataset. The task formulation is the
same as SNEMI3D — predict an affinity map (optionally augmented with a
signed distance transform), then run watershed-style post-processing to get
instance segmentation.

PyTC ships reproduction targets for several NISB / BANIS configurations
under ``tutorials/neuron_nisb/``:

- ``base_banis.yaml`` / ``base_banis_chunk.yaml`` / ``base_banis_crop.yaml``
  — base BANIS training (MedNeXt-S, k=3, 128-cube, 200k steps, 6-channel
  affinity, 9 nm).
- ``base_banis_v1.yaml`` / ``base_banis_v2.yaml`` / ``base_banis_v3.yaml``
  — variants of the base config.
- ``*_erosion2.yaml`` — same configs with label erosion 2 to widen instance
  borders for class-imbalance handling.
- ``common.yaml`` — the recommended starting point: MedNeXt-B with affinity
  + SDT (40 nm).

1 - Get the data
^^^^^^^^^^^^^^^^^^

The NISB dataset is consumed at the path baked into the configs. Update the
``data.train`` / ``data.val`` / ``data.test`` entries in the YAML you pick
to point at your local copy.

2 - Run training
^^^^^^^^^^^^^^^^^^

The standard PyTC entry point is the same as for SNEMI3D:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/common.yaml

Override anything from the CLI:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/common.yaml \
        data.dataloader.batch_size=4 optimization.max_epochs=200

3 - Inference and decoding
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``--mode test --checkpoint <ckpt>`` to run the combined inference +
decode + evaluate pipeline:

.. code-block:: bash

    python scripts/main.py --config tutorials/neuron_nisb/common.yaml \
        --mode test --checkpoint outputs/.../checkpoints/last.ckpt

The decode stage uses waterz against the predicted affinity map; the
evaluation stage reports Rand and VOI on the held-out volume.

    .. note::

        For SDT-based decoding, see the ``decoding`` block in ``common.yaml``;
        for plain affinity, see ``base_banis.yaml``.

For the underlying mechanics (affinity learning, waterz post-processing),
see :doc:`snemi3d` — the same pipeline applies.
