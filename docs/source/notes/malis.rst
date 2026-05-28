MALIS Loss
==========

The ``MalisLoss`` (`connectomics.models.losses.malis.MalisLoss`) is a
constrained structured loss for 3D affinity prediction, used as a
companion term to per-channel BCE in neuron-segmentation configs such
as ``tutorials/neuron_nisb/base_banis+_malis.yaml``.

This page documents two knobs that significantly affect performance
and correctness:

- ``malis_crop_size`` — random sub-volume crop on the MALIS forward.
- ``label_transform.emit_gt_seg`` — pass the eroded GT segmentation
  through to MalisLoss to skip the per-step connected-components
  call and to preserve global instance IDs under cropping.

The Cost of Full-Volume MALIS
-----------------------------

MALIS computes per-edge weights via two maximin-tree passes over the
affinity graph; for each sample these passes are single-threaded
CPU work that the GPU forward/backward path waits on. On the BANIS
production config (MedNeXt-L, batch 2, 128³ patch), the cost is large:

.. list-table:: BANIS production config — measured step rate
   :header-rows: 1

   * - Configuration
     - Run
     - it/s
     - sec/step
     - h/epoch (5000 steps)
   * - BCE only (no MALIS)
     - slurm 2442858 / 2442857
     - ~0.71
     - ~1.4
     - ~1.95
   * - Full-volume MALIS (original)
     - slurm 2487040
     - ~0.17
     - ~5.9
     - ~7.3
   * - MALIS with ``malis_crop_size: 64``
     - slurm 2505814
     - ~0.78
     - ~1.3
     - ~1.78

The "Full-volume MALIS" row is the original implementation. Adding
MALIS makes each epoch ~3.5× slower than BCE-only on this hardware.

``malis_crop_size`` — Random Sub-Volume Crop
--------------------------------------------

Setting ``malis_crop_size: K`` (or ``[Kz, Ky, Kx]``) instructs
MalisLoss to apply a single random ``K × K × K`` crop to ``pred``,
``target``, and ``mask`` on each forward call before computing MALIS.
The crop origin is shared across the batch and resampled every step,
so over many iterations the model still sees MALIS supervision
covering the whole patch in expectation.

The reduction in MALIS volume is **cubic in the crop ratio**. At
``K = 64`` on a ``128³`` patch the cropped volume is ``1/8`` of the
original, and CPU MALIS work drops by roughly the same factor.

YAML usage:

.. code-block:: yaml

    model:
      loss:
        losses:
          - function: PerChannelBCEWithLogitsLoss
            weight: 1.0
            kwargs: { auto_pos_weight: true, max_pos_weight: 10.0 }
          - function: MalisLoss
            weight: 1.0
            pred_slice: "0:3"
            target_slice: "0:3"
            kwargs:
              nhood: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
              sigmoid: true
              reduction: mean
              malis_crop_size: 64   # or e.g. [32, 64, 64] for anisotropic

Default ``malis_crop_size: null`` (or omitting the key) preserves
the original full-volume behavior bit-for-bit.

**Measured speedup vs original (slurm 2505814 vs 2487040):**

- it/s: 0.17 → 0.78 — **~4.6× faster per step**.
- hours/epoch (5000 steps): 7.3 → 1.78 — **~4.1× faster wall-clock**.
- Total training (200 k steps, 40 epochs at 5000/epoch): ~12 days →
  ~3 days.

**Caveat.** Cropping reduces the spatial volume MalisLoss sees per
step. The model is still trained on the full patch by the BCE term,
but the structured MALIS signal is restricted to the cropped sub-volume
on each step. Convergence behaviour on your dataset should be
validated empirically before committing to a final crop size.

``emit_gt_seg`` — Skip the Per-Step CC, Fix Cropped Topology
------------------------------------------------------------

Without ``emit_gt_seg``, MalisLoss reconstructs the GT segmentation
from the GT affinity tensor each step:

.. code-block:: text

    gt_seg = connected_components_affgraph(target > 0.5, nhood)

This is wasteful (the data pipeline already produced the segmentation
upstream), but more importantly it interacts badly with
``malis_crop_size``: when a single GT instance spans the crop boundary,
the CC of the cropped affinities labels its pieces as **distinct
components**, and MALIS then injects spurious negative-constraint
edges inside one true instance.

Setting ``label_transform.emit_gt_seg: true`` adds a small
``CopyItemsd`` MONAI transform immediately after the existing
``SegErosionInstanced`` step, snapshotting the post-augmentation,
post-erosion segmentation as ``batch["gt_seg"]``. The loss
orchestrator forwards it to MalisLoss, which:

- Skips the per-step ``connected_components_affgraph`` call.
- Crops the supplied ``gt_seg`` at the same origin as ``pred`` /
  ``target`` / ``mask``, preserving each instance's global label
  inside the crop window.

YAML usage:

.. code-block:: yaml

    default:
      data:
        label_transform:
          erosion: 2
          emit_gt_seg: true   # opt-in; pairs with MalisLoss

Default ``emit_gt_seg: false`` preserves the legacy CC-recompute
behavior bit-for-bit; configs without MalisLoss are unaffected.

**Deep supervision.** When deep supervision is active, the loss
orchestrator forces ``gt_seg=None`` for every head (DS lower heads
work on downsampled targets that ``gt_seg`` cannot match
label-correctly with the same cheap transform). MalisLoss in a DS
config falls back to the CC-recompute path; the BANIS production
configs use ``deep_supervision: false`` and are unaffected.

Combined Speedup
----------------

Both knobs compose cleanly. The crop is the dominant speedup; the
``gt_seg`` passthrough adds a small additional CPU saving (~5–10 % of
the remaining MALIS overhead — the per-step CC is removed but the
two maximin-tree passes still dominate) on top of correctness.

.. list-table:: Cumulative impact relative to the original
   :header-rows: 1

   * - Configuration
     - Speedup vs original
     - Correctness fix
   * - Full-volume MALIS (original)
     - 1.0×
     - —
   * - ``malis_crop_size: 64``
     - ~4.6× (measured)
     - —
   * - ``malis_crop_size: 64`` + ``emit_gt_seg: true``
     - ~4.6× plus a few % (estimated)
     - Preserves global instance IDs under crop;
       removes spurious negative-constraint edges
       at cropped instance boundaries.

The correctness fix is the primary motivation for
``emit_gt_seg``; the speed benefit is secondary.

See Also
--------

- ``tutorials/neuron_nisb/base_banis+_malis.yaml`` — production
  config with both knobs enabled.
- ``lib/malis/INVESTIGATION.md`` — internal notes on GPU MALIS
  candidates and algorithm-level speedups beyond what this PR
  ships.
- The MalisLoss reference: Turaga *et al.*, *Maximin learning of
  image segmentation*, NIPS 2009.
