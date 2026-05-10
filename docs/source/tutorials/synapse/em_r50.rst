EM-R50 (Synaptic Polarity Detection)
======================================

This tutorial covers synaptic **polarity** detection — predicting
separated pre-synaptic and post-synaptic masks so the signal flow
between neurons can be traced. The dataset was released by
`Lin et al. (2020) <http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630103.pdf>`__
from Layer II/III of the primary visual cortex of an adult rat.

Unlike the cleft-detection task, polarity detection requires
distinguishing individual synapses *and* assigning a side (pre /
post) to each. The model outputs three channels — pre-synaptic,
post-synaptic, and synaptic (union) — and a connected-components
post-processor groups them into per-synapse instances.

    .. note::
        A dedicated EM-R50 tutorial config does not yet ship under
        ``tutorials/`` in the modern (v2/v3) PyTC layout. The decoder
        and target-generation building blocks are present
        (``polarity2instance`` decoder; ``seg_to_polarity`` target),
        but a canonical end-to-end YAML is a follow-up. The recipe
        below shows how to assemble one from the existing CREMI
        cleft-detection config (``tutorials/syn_cremi.yaml``) plus
        the polarity-specific pieces.

Goal
----

The intended pipeline:

- **Model output** 3 channels — pre-synaptic, post-synaptic, and
  synaptic-union.
- **Target generation** ``seg_to_polarity`` over instance labels;
  set ``exclusive=true`` if pre / post should never overlap (uses
  softmax) or ``false`` for the standard non-exclusive BCE setup
  (channels predicted independently with sigmoid).
- **Loss** ``WeightedBCEWithLogitsLoss`` with rejection sampling on
  the foreground (synapses are sparse), or ``WeightedCE`` for the
  exclusive variant.
- **Decoder** ``polarity2instance`` — connected-components on the
  synaptic-union channel, then voting on pre / post per instance.
- **Metric** an IoU-based F1 score on instance-matched pairs.

1 - Get the data
^^^^^^^^^^^^^^^^

.. code-block:: bash

    mkdir -p datasets/jwr15 && cd datasets/jwr15
    wget http://rhoana.rc.fas.harvard.edu/dataset/jwr15_synapse.zip
    unzip jwr15_synapse.zip

2 - Building a v2 config (template)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Start from ``tutorials/syn_cremi.yaml`` and apply three changes:

(a) **Switch the model output to 3 channels** and use a non-exclusive
BCE loss (or ``WeightedCE`` if you prefer exclusive masks):

.. code-block:: yaml

    default:
      model:
        out_channels: 3
        loss:
          losses:
          - function: WeightedBCEWithLogitsLoss
            weight: 1.0
            kwargs: {reduction: mean}
            pred_slice: "0:3"
            target_slice: "0:3"

(b) **Drive target generation through the polarity helper.** In your
``data.label_transform`` block, generate the 3-channel polarity
target from instance labels by post-processing your label volume
through ``connectomics.data.processing.target.seg_to_polarity``
(invoke from a small data-preparation script ahead of training, or
add it as a custom processing step). Disable the cleft binary path
that ships in ``syn_cremi.yaml``.

(c) **Add the polarity decoder** under ``decoding.steps``:

.. code-block:: yaml

    decoding:
      steps:
      - name: polarity2instance
        kwargs:
          # default: non-exclusive (independent BCE channels);
          # set true if you trained with WeightedCE / softmax.
          exclusive: false

The rest of the CREMI config (RSUNet, anisotropic 18 × 256 × 256
patches, sliding-window inference, EMA, augmentation profile) carries
over.

    .. tip::
        Synapses are sparse, so add **rejection sampling** to focus
        training on patches that contain foreground:

        .. code-block:: yaml

            data:
              dataloader:
                reject_sampling:
                  size_thres: 1000
                  p: 0.95

        This is what ``syn_cremi.yaml`` already does for cleft
        detection; keep it for polarity training.

3 - Run training
^^^^^^^^^^^^^^^^

Once your config is assembled (e.g. saved as
``tutorials/syn_em_r50.yaml``):

.. code-block:: bash

    conda activate pytc
    python scripts/main.py --config tutorials/syn_em_r50.yaml

4 - Inference, decoding, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python scripts/main.py --config tutorials/syn_em_r50.yaml \
        --mode test \
        --checkpoint outputs/<exp>/<timestamp>/checkpoints/last.ckpt

The decoding stage runs ``polarity2instance``, which converts the
3-channel probability map into per-synapse instance masks with a
recorded side (pre vs. post). For programmatic use:

.. code-block:: python

    from connectomics.decoding import polarity2instance
    # volume: (3, D, H, W) prediction, channels = [pre, post, syn]
    instances = polarity2instance(volume)

The ``exclusive=True`` variant interprets the channels as a softmax
output over {background, pre, post} and uses a different connected-
component path; pass ``exclusive=True`` to ``polarity2instance`` if
your model was trained with ``WeightedCE``.

5 - Reference behavior
^^^^^^^^^^^^^^^^^^^^^^^^

- **Training loss** is dominated by the synaptic-union channel
  early on; the pre / post channels improve as the model learns
  spatial directionality, typically after the first cosine
  decay phase. Rejection sampling plus higher foreground weights
  is essential — without it, pre and post collapse to all-zeros.
- **Inference** is identical to the CREMI flow (same RSUNet, same
  sliding window). The decoder runs in CPU and is fast.
- **Reporting** the published EM-R50 F1 is computed against the
  Lin et al. 2020 evaluation script. Comparable code lives in the
  upstream paper repo; contact the maintainers if you intend to
  benchmark formally.
