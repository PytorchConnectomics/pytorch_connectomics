# Task

Implement geometrically correct test-time augmentation (TTA) for directional affinity outputs.
The current implementation spatially reverses flips and rotations but treats every output
channel as a scalar field. That is incorrect for affinity channels: a spatial transform changes
the direction represented by each channel and may also change which endpoint stores the edge.

This is one incremental CCC change. It must fix both the standard whole-view TTA path and the
patch-first local TTA path through one canonical affinity inverse. Do not implement a
Zebrafinch-only hard-coded channel swap or a second ad hoc TTA loop.

## Motivation and measured evidence

The bug is in `connectomics/inference/tta.py`:

1. Input augmentation is applied as spatial flips followed by `torch.rot90`.
2. Prediction inversion applies the inverse rotation followed by the inverse flips.
3. No affinity-channel permutation or edge re-anchoring occurs before ensemble aggregation.

CUDA diagnostics on an L40S established the required behavior:

- With a perfect source-indexed affinity oracle and one axis-0 flip, current spatial-only TTA
  produced 2,064 mismatched voxels; offset-aware inversion produced exactly 0.
- With an XY quarter-turn, current spatial-only TTA produced 4,270 mismatched voxels;
  channel permutation plus direction re-anchoring produced exactly 0.
- On the trained BANIS/LiCONN checkpoint, flipping spatial axis `c` required correcting affinity
  channel `c` with a `-1` roll along that axis. The relevant-channel mean absolute differences
  versus the unaugmented prediction changed as follows:

  | flip | channel | spatial-only | corrected |
  |---|---:|---:|---:|
  | axis 0 | 0 | 0.05243 | 0.02093 |
  | axis 1 | 1 | 0.04389 | 0.01986 |
  | axis 2 | 2 | 0.04037 | 0.01971 |

  The other two channels for each flip were best with no roll. The residual after correction is
  ordinary learned-model non-equivariance; the extra error is the geometry bug.

`lib/DeepEM/deepem/test/fwd_utils.py::revert_flip` is a useful semantic reference: it swaps
directional channels for XY transpose and shifts the affinity channel affected by a flip.
However, it must not be copied literally because DeepEM hard-codes x/y/z unit offsets, uses
destination-indexed affinity, and zero-fills invalid faces.

Existing generated Zebrafinch `r10ttamin` artifacts were deleted because they compared
misregistered edges. They must not be recreated until this contract passes.

## Goal

Make TTA output-aware:

- scalar and other non-directional channels retain their existing spatial inversion behavior;
- configured affinity channels are transformed according to their configured offset vectors;
- both `affinity_mode: banis` and `affinity_mode: deepem` use the correct storage convention;
- r1, r10, arbitrary supported signed offsets, flips, and 90-degree rotations are handled by
  the same offset-vector logic;
- invalid wrapped faces are excluded from mean/min/max aggregation rather than inserted as zero;
- standard, patch-first, selected-channel, named-head, and distributed-sharded TTA agree on the
  same semantics.

The implementation must be generic. Do not hard-code `channel == spatial_axis`, three channels,
unit offsets, or a Zebrafinch array-order label such as XYZ/ZYX. Configured offsets are positional:
offset component `i` refers to spatial tensor axis `i`.

## Normative affinity model

For an offset vector `t`, an affinity value describes an undirected edge between two voxels but
is stored at one endpoint:

- `banis` is source-indexed: `A_t(v)` describes the edge `(v, v + t)` and is stored at `v`.
- `deepem` is destination-indexed: `A_t(v)` describes the edge `(v - t, v)` and is stored at `v`.

The configured affinity offsets and mode are canonical and must come from the existing helpers in
`connectomics/data/processing/affinity.py`, notably:

- `resolve_affinity_channel_groups_from_cfg`;
- `resolve_affinity_mode_from_cfg`;
- `resolve_affinity_offsets_from_kwargs` where needed.

Inference is allowed to depend on `data` under the repository dependency contract. Reuse these
helpers rather than creating a second parser for offsets or `affinity_mode`.

## Normative transform contract

### Transform order

The current augmentation order is part of the contract:

```text
x_aug = rotate(flips(x), k)
```

The spatial inverse is therefore:

```text
pred_original_frame = flips(inverse_rotate(pred_aug, k))
```

Offset vectors must undergo the linear part of exactly that same inverse, in the same order.
Tensor dimensions include batch and channel prefixes; offset-vector dimensions do not.

For each source affinity channel with configured augmented-frame offset `o`:

1. Compute its original-frame directed offset `d` using the inverse spatial transform.
2. Within the same configured affinity group, find the unique target channel whose configured
   offset `t` satisfies either `d == t` or `d == -t`.
3. Prefer the exact-direction match `d == t`. It requires channel placement only.
4. A sign-reversed match `d == -t` requires channel placement plus edge re-anchoring as below.
5. The mapping for a group must be bijective. Duplicate/ambiguous offsets, a missing transformed
   counterpart, or a mapping that targets one channel twice must raise a clear `ValueError`
   before expensive volume inference where practical. Never silently fall back to scalar TTA.

This offset mapping naturally implements channel swaps under transpose/rotation and direction
changes under flips or quarter-turns. It must work for offsets such as `(10, 0, 0)` by moving ten
voxels, not one.

### Sign-reversed edge re-anchoring

After spatial inversion, suppose a source channel represents `d == -t`, where `t` is the target
channel's configured offset.

For BANIS source-indexed affinity:

```text
Q_t(v) = P_-t(v + t)
roll shifts = -t
```

For DeepEM destination-indexed affinity:

```text
Q_t(v) = P_-t(v - t)
roll shifts = +t
```

Using PyTorch's roll convention, `roll(input, shifts=s)` gives `output(v) = input(v - s)`.
Wrapped values are invalid and must never contribute to an ensemble:

- a positive roll component invalidates the leading `s` voxels on that axis;
- a negative roll component invalidates the trailing `abs(s)` voxels;
- for a multi-axis offset, the valid region is the intersection of the valid ranges on all axes.

The implementation may avoid `torch.roll` and copy between explicit slices, but the result and
validity region must be identical to these equations.

### Preprocessing and channel selection

Directional correction occurs on raw model outputs in the original spatial frame, before
`apply_preprocessing` applies channel-specific activations and `inference.model.select_channel`.
This ensures activations and selectors refer to canonical output channels rather than augmented
directions.

Only affinity channels are directionally remapped. Other channels in a stacked tensor or another
head receive the ordinary spatial inverse and remain in their original channel positions.

The output-to-target mapping must cover:

- single-tensor models aligned to the stacked label channels;
- named heads using `ModelHeadConfig.target_slice`;
- per-head inference used by merged-head saving;
- selection of r1 (`[0, 1, 2]`) or r10 (`[3, 4, 5]`) after the raw six-channel affinity output
  has been corrected.

If a configured affinity output exists but its raw output channels cannot be mapped
unambiguously to label channels, fail with an actionable error. If the configuration has no
affinity output, preserve existing scalar TTA behavior without requiring affinity metadata.

## Validity-aware ensemble contract

Invalid wrapped faces are missing observations, not affinity zero. Do not use DeepEM's
zero-fill-and-count-every-view behavior.

For every output channel and voxel:

- `mean` computes the sum of valid contributions divided by the number of valid contributions;
- `min` and `max` ignore invalid contributions completely;
- a location with zero valid contributions raises a clear runtime error rather than emitting
  NaN, infinity, or a fabricated zero.

The identity view normally guarantees coverage, but the code must not assume the user configured
one. Per-channel ensemble modes must retain their existing meaning.

Streaming accumulation is required. Do not retain every full-volume prediction merely to call a
NaN-aware reduction. The no-affinity/no-invalidity path should retain its current fast behavior.
Validity may be represented with explicit valid slices rather than a full boolean tensor. Avoid
an `O(num_views * volume)` memory increase; use the smallest safe persistent counting or
slice-based representation consistent with correctness.

Apply pointwise activation only once per view. Preserve the configured final output dtype and do
not let validity sentinels leak into saved predictions.

## Standard TTA path

For standard eager/sliding inference:

1. Run one complete augmented prediction through `network_fn`.
2. Apply the shared spatial/directional inverse to the complete view.
3. Apply activation/channel selection in canonical channel order.
4. Stream the view into the validity-aware ensemble accumulator.

Directional correction at the complete-view level means there are no artificial internal patch
faces; only the transformed view's real outer face is invalidated.

## Patch-first local TTA path

Patch-first TTA currently duplicates the spatial inverse inside its ROI loop. Replace that
duplication with the same canonical inverse used by standard TTA.

Because directional re-anchoring invalidates a face of each ROI prediction, patch blending must
also be validity-aware:

1. Correct each raw ROI prediction before writing it to the sliding accumulator.
2. Multiply the value contribution and blending weight by that channel's validity.
3. Maintain an appropriate per-augmentation/per-channel denominator when affinity validity
   differs by channel; the current one-channel shared weight accumulator is insufficient for
   this case.
4. Overlapping neighboring ROIs may fill an interior face invalidated in one ROI.
5. After ROI blending, expose whether each channel/voxel received a valid contribution from that
   view, then exclude uncovered locations from the cross-view ensemble.

Do not introduce zero-valued stripes at ROI boundaries. For a deterministic perfect oracle,
patch-first and standard TTA must agree exactly, including r10 and boundary behavior.

The implementation must not worsen the existing asymptotic retention of per-view patch-first
accumulators unless the planner explicitly replaces it with a less memory-intensive equivalent.

## Distributed TTA sharding

Distributed-sharded TTA must reduce both prediction statistics and validity information:

- mean channels reduce valid sums and valid counts, then divide on rank zero;
- min/max channels reduce while ignoring invalid entries on every rank;
- a rank whose local augmentation shard has no valid value for a location must not bias the
  global result with zero or an extremum sentinel;
- final zero-coverage validation occurs after the global reduction;
- non-zero ranks retain the existing skip-postprocessing behavior.

The current scalar-count reduction is insufficient when affinity faces have per-voxel counts.
Retain the existing scalar fast path when every contribution is valid.

## Ownership and implementation constraints

- One canonical implementation owns spatial inversion, affinity offset transformation, channel
  placement, sign re-anchoring, and validity description. Both standard and patch-first paths
  call it.
- A focused module such as `connectomics/inference/tta_affinity.py` is acceptable if it keeps
  `tta.py` readable, but the CCC plan decides the smallest coherent decomposition.
- Keep orchestration in `connectomics/inference`; do not move TTA into training or decoding.
- Reuse current config and model-output helpers. Do not add a compatibility shim, duplicate
  offset parser, or hard-coded Zebrafinch special case.
- Prefer automatic behavior derived from declared affinity targets. Do not add a user switch that
  allows the same configured affinity channels to be silently treated as scalars.
- Do not add dependencies.
- Do not create commits during the CCC run.
- Preserve strict config behavior and public API snapshots unless a deliberately new public API
  is justified by the plan.
- Preserve existing input-transform enumeration, deduplication, masking, empty-cache behavior,
  output dtype, and scalar/non-affinity results.

## Required tests

Add focused tests, preferably in `tests/unit/test_inference_tta_affinity.py`, using a deterministic
perfect affinity oracle generated from a synthetic segmentation. Correctness tests must compare
the final values, not only shapes or channel indices.

### Exact geometry tests

For both `banis` and `deepem`:

1. Generate canonical affinities from a non-symmetric synthetic segmentation.
2. Apply the configured input augmentation to the segmentation.
3. Generate perfect affinities in the augmented frame using the same configured offsets/mode.
4. Run the production TTA inverse and validity handling.
5. Require exact equality to the canonical affinities at every valid location.

Cover at least:

- each single-axis flip;
- multiple simultaneous flips;
- each relevant 90-degree rotation (`k=1,2,3`) and channel permutation;
- a combined flip plus rotation, exercising the required transform order;
- r1 and r10 offsets;
- the expected leading/trailing invalid face for BANIS versus DeepEM;
- a signed or multi-axis offset if such offsets are accepted by the existing parser;
- bijection/missing-counterpart validation failures.

Use spatial shapes larger than twice the largest tested offset and choose a segmentation whose
affinities vary spatially so an off-by-one or wrong-channel result cannot pass accidentally.

### Integration/regression tests

Also cover:

- standard TTA end-to-end through `TTAPredictor.predict` with a perfect oracle;
- patch-first TTA exact agreement with standard TTA over overlapping ROIs, with no internal seams;
- `mean`, `min`, `max`, and per-channel ensemble-mode handling at invalid faces;
- `select_channel` for r1 and r10 after raw-channel correction;
- a mixed affinity plus scalar output tensor;
- a named affinity head with `target_slice` and a non-affinity head left untouched;
- distributed reduction of per-voxel valid counts/statistics, using the repository's existing
  distributed-test pattern or a focused mocked reduction where a multiprocess test is impractical;
- unchanged scalar TTA behavior when no affinity target is configured;
- unchanged no-TTA and identity-only fast paths.

The exact geometry suite must run on CPU. A CUDA smoke test may be added when CUDA is available,
but CI correctness must not depend on a GPU or Zebrafinch data.

## Documentation cleanup

After the implementation is proven:

- update stale comments in `dev/nisb/liconn/tta_probe.py` and related diagnostic scripts that say
  repository TTA lacks affinity correctness;
- document that r10 is displaced by its configured offset and that invalid faces are
  validity-weighted;
- keep the Zebrafinch TTA YAML recipes disabled or clearly non-authoritative until a corrected
  local result is generated. Do not regenerate large inference artifacts in this CCC task.

## Out of scope

- Running full-volume Zebrafinch inference or recreating deleted `r10ttamin` artifacts.
- Claiming that TTA improves NERL. This task establishes geometric correctness; performance and
  decoder-threshold tuning are separate experiments.
- Implementing ABISS tuning, nuclei masks, glia classification, r10 anchor segmentation, global
  linking, or error correction.
- Changing the set or order of TTA augmentation combinations.
- Adding arbitrary-angle interpolation-based rotations.
- Changing affinity target generation or decoder edge conventions.

## Success criteria

1. A perfect affinity oracle is exactly invariant after every tested augmentation/inverse for
   BANIS and DeepEM, including r10 and combined transforms.
2. Channel permutation and endpoint re-anchoring are derived from configured offsets; there are
   no x/y/z or channel-0/1/2 special cases in production code.
3. Invalid faces never enter mean/min/max as zero and zero-valid-count output is rejected.
4. Standard and patch-first TTA share one inverse and agree exactly in the oracle integration test.
5. Distributed TTA produces the same result as unsharded TTA with validity-varying affinity
   contributions.
6. Scalar/non-affinity TTA tests remain green and retain their fast path.
7. r1/r10 channel selection and named-head target mapping are tested and correct.
8. Existing public API, mask behavior, augmentation enumeration, and tutorial config validation
   do not regress.
9. No full-volume output artifact is generated by the implementation task.

## Verification

At minimum, run in the `pytc` conda environment:

```bash
conda run -n pytc pytest -q \
  tests/unit/test_inference_tta_affinity.py \
  tests/unit/test_inference_tta_masking.py
conda run -n pytc python scripts/validate_tutorial_configs.py \
  --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'
```

Run changed-file formatting and static checks required by `AGENTS.md`:

```bash
conda run -n pytc black --check <changed_python_files>
conda run -n pytc isort --check-only <changed_python_files>
conda run -n pytc flake8 --max-line-length=100 <changed_python_files>
conda run -n pytc mypy --config-file .github/mypy_changed.ini <changed_python_files>
```

Run broader focused inference tests selected by the CCC plan when private helper signatures or
distributed reduction behavior change. Report every skipped or unavailable check explicitly.

