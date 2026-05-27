# Plan v1

## Summary

Revises plan_v0 in response to the codex review. Three majors corrected:
(a) C2's alignment must subsample the accumulator's per-Z `means` by
`z_stride` before computing `interior_mean`, because `_per_z_scan`
applies `k_edge` to *sampled* rows; (b) C7's interior-outlier test is
replaced with a head-fill design that actually distinguishes windowed
vs full-curve behavior; (c) C5 is dropped — `AffinityQCAccumulator.finalize`
keeps its `img=` keyword to preserve the public API. Two minors
refined: C1's error wording explicitly names the rank-0 contract, and
C4's preflight rule exempts configs that already have
`decoding.affinity_mask_path` wired so the existing short-circuit path
stays usable. The three questions from the review are answered
explicitly in `## Risks and Questions`.

## Scope

Unchanged from plan_v0 in terms of files touched. C5 is removed from
the change list. C2/C4/C7 are reshaped per review. C1/C3/C6 stand.

In scope:

- `connectomics/decoding/qc/affinity.py` — finalize alignment (C2
  revised), dead-branch removal (C3), `update` docstring tightening
  (C6), zero-update error wording (C1b).
- `connectomics/training/lightning/test_pipeline.py` — rank-0 guard
  around `finish_streaming_qc` (C1a).
- `connectomics/runtime/preflight.py` — conditional `image_path`
  requirement (C4 revised).
- `tests/unit/test_decoding_qc_affinity.py` — three new tests, one
  reshaped (C7 revised).

Still out of scope:

- Numerical-stability rewrite of `E[X²] − E[X]²`.
- `_xy_border_rows` structural refactor.
- Splitting `run_affinity_qc` into smaller functions.
- Touching the dev/nisb CLI wrappers.
- Removing the streaming short-circuit in `run_affinity_qc`.

## Proposed Changes

### C1 [major] Distributed rank-0 guard for `finish_streaming_qc`

Unchanged from plan_v0 in spirit. Two pieces:

1. **C1a (test_pipeline.py)** — wrap the `finish_streaming_qc(...)`
   call in `if rank == 0:`. Reuse the same
   `torch.distributed.is_available()/is_initialized()/get_rank()` probe
   already present 7 lines below. Pseudocode:

   ```text
   should_finalize_qc = True
   if torch.distributed.is_available() and torch.distributed.is_initialized():
       if torch.distributed.get_world_size() > 1 and torch.distributed.get_rank() != 0:
           should_finalize_qc = False
   if qc_acc is not None and should_finalize_qc:
       finish_streaming_qc(module.cfg, qc_acc)
   ```

2. **C1b (affinity.py)** — refine the existing
   `AffinityQCAccumulator.finalize` error so the *all-zero* case prints
   a distinct, debuggable message:

   ```text
   if (n == 0).all():
       raise ValueError(
           "AffinityQCAccumulator.finalize: no slabs were folded in. "
           "In distributed runs, finish_streaming_qc must only be "
           "called on rank 0."
       )
   if (n == 0).any():
       missing = int((n == 0).sum())
       raise ValueError(
           f"AffinityQCAccumulator.finalize: {missing} Z slices were never updated"
       )
   ```

   The existing partial-update message is preserved verbatim so the
   test asserting that substring stays green.

### C2 [major, REVISED] Align streaming Z-cut with post-save (sampled interior)

The codex review correctly noted that aligning only the head/tail loop
bounds is insufficient: post-save's `interior_mean` is computed from
`_per_z_scan`'s **sampled** rows (one row per `z_stride` raw slices),
while the accumulator stores every raw z. With default `z_stride=10`,
`k_edge=20`, post-save trims 200 raw z from each end; the streaming
finalize as written trims only 20 raw z.

Revised fix: in `AffinityQCAccumulator.finalize`, derive the same
sampled curve before computing `interior_mean`, then apply the
windowed cut scan:

```text
z_stride = params.z_stride
sampled = means[::z_stride]                                # (S, C)
sampled_z_idx = np.arange(0, Z, z_stride, dtype=np.int64)  # mirrors _per_z_scan
ke = params.k_edge
if len(sampled) > 2 * ke + 1:
    interior = sampled[ke:-ke]
else:
    interior = sampled
interior_mean = interior.mean(axis=0)
cutoff = interior_mean - params.drift_thresh

# Windowed head scan: first z in [0, refine_window) with mean >= cutoff.
head_end = min(params.refine_window, Z)
low_z = head_end
for z in range(head_end):
    if bool((means[z] >= cutoff).all()):
        low_z = z
        break

# Windowed tail scan: last z in [Z - refine_window, Z) with mean >= cutoff.
tail_start = max(0, Z - params.refine_window)
high_z = tail_start
last_ok = -1
for z in range(tail_start, Z):
    if bool((means[z] >= cutoff).all()):
        last_ok = z
high_z = last_ok + 1 if last_ok >= 0 else tail_start
```

This now matches the post-save semantics exactly for the cutoff and
the cut points: `interior_mean` uses the same sampled population
(`means[::z_stride]`), `k_edge` lives in sampled-row units in both
paths, and the head/tail scan windows are identical.

Note: the report's `g_mean`/`g_std` stats remain computed over all
streamed voxels in the accumulator (i.e. the full volume), because the
accumulator has that data for free. These are informational only and
do not affect `low_z`/`high_z`. The post-save path samples ~1/z_stride
of voxels for the same stats. This intentional informational drift is
called out in the report's "Volume health" prose.

### C3 [major] Remove dead-branch in `finish_streaming_qc`

Unchanged from plan_v0. The `else:` branch claiming a Z-only fallback
unconditionally raises. Replace with an early check:

```text
image_path, mask_path, report_path = _resolve_qc_paths(cfg, qc_cfg)
if not image_path:
    raise ValueError(
        "streaming affinity_qc requires decoding.affinity_qc.image_path; "
        "no in-memory prediction is retained to size the (X, Y, Z) mask."
    )
```

Then fold the existing `if image_path: build_affinity_mask(...)` block
into a straight call (no `else`). Net deletion: ~14 lines of dead code.

### C4 [minor, REVISED] Preflight: require `image_path` when not pre-wired

The original plan_v0 rule could reject a usable config that pre-sets
`decoding.affinity_mask_path` (e.g. for a decode-only re-run where the
streaming finalize has already populated the mask on a prior run).
Plan_v0's wording said "require `image_path` when streaming"
unconditionally; revise to:

```text
if mode == "streaming":
    image_path_set = bool(getattr(qc_cfg, "image_path", "") or "")
    existing_mask = bool(getattr(decoding_cfg, "affinity_mask_path", "") or "")
    if not image_path_set and not existing_mask:
        raise ValueError(
            "decoding.affinity_qc.mode='streaming' requires either "
            "decoding.affinity_qc.image_path (to build the mask post-stitch) "
            "or a pre-populated decoding.affinity_mask_path (decode-only re-run)."
        )
```

The existing strategy check (`strategy == 'chunked'`) stays. This
preserves the `run_affinity_qc` short-circuit at the same time it
makes the common-case error fast-fail at preflight.

### C5 [WITHDRAWN] Keep `AffinityQCAccumulator.finalize(img=...)` kwarg

Per the reviewer, `AffinityQCAccumulator` is exported via
`connectomics.decoding.qc.__all__`, so `finalize` is part of the public
API in practice. Plan_v1 keeps the keyword accepted (even though the
streaming path does not consume it). The hardcoded `border_rows`
message stays. If a future task introduces an XY-border check at
finalize time, the parameter is already there.

### C6 [minor] Tighten `AffinityQCAccumulator.update` docstring

Unchanged from plan_v0. Update docstring to spell out the two layouts:

```text
update(slab, *, z_offset, z_axis=-1)
    slab: array-like (C, *spatial).
    z_axis: which axis of slab is Z. Defaults to -1 (post-save (C, X, Y, Z)
            convention). Streaming callers from the chunked stitcher pass
            z_axis=1 because slabs there are (C, Zslab, Y, X).
```

No signature change.

### C7 [REVISED] Three new unit tests with corrected synthetic

Reviewer noted that the proposed interior-outlier synthetic at
`(0, 1, 30, 31, 58, 59)` does not exercise the C2 bug — the full-curve
finalize would still pick `low_z=2` and `high_z=58` because there are
good slices both before and after `30/31`. The pre-fix vs post-fix
divergence appears when **all** slices in the head window (or tail
window) are bad with the first good slice immediately past the window.

Revised tests:

1. **`test_streaming_finalize_matches_post_save_when_head_window_fully_bad`**

   ```text
   Build (C=3, X=8, Y=8, Z=60) with z=0..32 bad (zeros) and z=33..59 good (0.6).
   Use params: z_stride=1, k_edge=4, refine_window=30, drift_thresh=0.05.
   Both scan_prediction and accumulator.finalize must return low_z=30, high_z=60.
   Spot-check (manual, not in the test): revert C2's subsampling fix → this test
   still passes because z_stride=1; revert the windowed loop bounds → streaming
   would return low_z=33, the test fails.
   ```

   A second variant exercises the `z_stride > 1` path:

   ```text
   Same volume but params: z_stride=5, k_edge=4, refine_window=30.
   Reverting the subsampling fix (so the accumulator's interior_mean still
   averages every-z) MAY produce a different cutoff if the bad region's
   contribution to a wide-window interior changes; either way the test
   asserts both paths agree.
   ```

   Combining these as parameterized cases gives the alignment guarantee
   under both stride regimes.

2. **`test_streaming_accumulator_finalize_all_zero_message`** — Build
   an accumulator, call `finalize` immediately. Assert the error
   message matches "no slabs were folded in. In distributed runs,
   finish_streaming_qc must only be called on rank 0."

3. **`test_preflight_streaming_requires_image_path_or_existing_mask`** —
   Load `tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml`, set
   `decoding.affinity_qc.enabled=True`, `mode='streaming'`,
   `inference.strategy='chunked'`, leave both `image_path` and
   `affinity_mask_path` empty → expect `ValueError` matching
   "image_path|affinity_mask_path". Then set `affinity_mask_path` to
   a dummy path and re-call `validate_runtime_coherence` → must not
   raise.

The existing
`test_streaming_accumulator_matches_one_shot_scan` synthetic stays as
the head-and-tail symmetric base case.

## Files and Areas

Same file list as plan_v0; C5 row removed.

| File | Section / Range | Change |
|---|---|---|
| `connectomics/decoding/qc/affinity.py` | `AffinityQCAccumulator.finalize` (≈ lines 109–164) | C2 revised: subsample by `z_stride` before `interior_mean`; windowed head/tail scan. C1b zero-update error wording. C5 withdrawn (keep `img=` kwarg). |
| `connectomics/decoding/qc/affinity.py` | `finish_streaming_qc` (≈ lines 562–622) | C3 early `image_path` check; remove dead `else` branch. `finalize(img=img)` continues to pass `img` through (no semantic effect today). |
| `connectomics/decoding/qc/affinity.py` | `AffinityQCAccumulator.update` (≈ lines 79–107) | C6 docstring only. |
| `connectomics/training/lightning/test_pipeline.py` | around line 749–750 | C1a: rank-0 guard. |
| `connectomics/runtime/preflight.py` | `validate_runtime_coherence` `affinity_qc` block (≈ lines 289–306) | C4 revised: require `image_path` OR pre-set `affinity_mask_path`. |
| `tests/unit/test_decoding_qc_affinity.py` | append | C7 revised three tests (head-fill synthetic; zero-update error; preflight). |

Estimated change size: ~65 net lines (≈ +35 logic, +30 tests).

## Verification Plan

Run after each implementation slice:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
python -m pytest tests/unit/test_decoding_qc_affinity.py \
                 tests/unit/test_v3_guardrails.py \
                 tests/unit/test_v2_boundaries.py \
                 tests/unit/test_decoding_pipeline.py -q
python scripts/validate_tutorial_configs.py
```

Acceptance:

1. All four pytest targets pass.
2. Tutorial validator reports 14 canonical OK, 2 skipped custom workflows.
3. `test_streaming_finalize_matches_post_save_when_head_window_fully_bad`
   exercises the C2 fix concretely. Manual spot-check: temporarily
   revert C2's windowed loop in `finalize` and confirm the test fails;
   restore.
4. `test_preflight_streaming_requires_image_path_or_existing_mask`
   exercises both the failure and exemption paths.
5. The existing
   `test_run_affinity_qc_streaming_short_circuit_when_mask_already_wired`
   continues to pass.

Coordinator-side static check:

- `git diff` on `connectomics/inference/` must not introduce any
  `from connectomics.decoding` import.
- `AffinityQCAccumulator.finalize` signature unchanged (still accepts
  `img=` kwarg).

## Risks and Questions

Direct answers to the codex reviewer's questions:

1. **Q.** Should streaming use post-save's sampled `z_stride`
   semantics exactly, including report stats?
   **A.** For the **cutoff and cut points** (the decoding-relevant
   outputs): yes — C2 revised subsamples by `z_stride` so they agree
   numerically with post-save. For **report-only stats** (`g_mean`,
   `g_std`, `g_min`, `g_max`): no — the accumulator has the full
   volume and producing more accurate aggregate stats is a free win.
   This intentional divergence is informational-only and noted in
   plan_v1.

2. **Q.** Should preflight allow `mode='streaming'` without
   `image_path` when `affinity_mask_path` is already set?
   **A.** Yes — C4 revised exempts this case. This preserves the
   `run_affinity_qc` short-circuit for decode-only re-runs.

3. **Q.** Is `AffinityQCAccumulator.finalize` part of the public API?
   **A.** Yes, per the `__all__` export. C5 is withdrawn; the `img=`
   keyword stays accepted (and unused) on the streaming path.

Residual risks:

- **R.** C2's subsampling approach assumes `len(means) == Z` (the
  accumulator always stores per-raw-z stats). If a future change
  decimates the accumulator, the subsampling math must follow. Not a
  concern today.

- **R.** C7's parameterized stride test increases the test count by
  one variant. Acceptable — the matrix is two cases, both fast on the
  small synthetic.

- **R.** C1a's rank probe in `test_pipeline.py` duplicates logic that
  lives 7 lines below. A small follow-up could factor a
  `_should_finalize_local_artifacts(module)` helper. Out of scope.

## Changes Since Previous Plan Version

- **C2 reshaped (response to plan_v0_review major):** The accumulator's
  `means` must be subsampled by `z_stride` before computing
  `interior_mean`, not only loop-bounded. Plan_v0 aligned the loops but
  left the interior population mismatched, so the cutoff value could
  still diverge. Plan_v1 subsamples in `finalize`.
- **C7 reshaped (response to plan_v0_review major):** The interior-
  outlier synthetic does not exercise the C2 bug because the full-curve
  scan accepts isolated bad slices when good slices flank them. Plan_v1
  uses a head-fill synthetic (all slices in head window bad, first
  good slice past the window) that distinguishes pre-fix from post-fix
  output. Added a `z_stride > 1` parameterization to exercise the C2
  subsampling fix specifically.
- **C5 withdrawn (response to plan_v0_review major):** The reviewer
  noted `AffinityQCAccumulator.finalize` is exported, so its signature
  is public. Plan_v1 keeps the `img=` kwarg accepted (unused on the
  streaming path) and the change list shrinks by one item.
- **C4 reshaped (response to plan_v0_review minor):** Preflight must
  not reject configs that pre-set `decoding.affinity_mask_path`, since
  `run_affinity_qc` short-circuits through those. The new rule
  requires `image_path` OR `affinity_mask_path`.
- **C1b wording (response to plan_v0_review minor):** The all-zero
  accumulator error now points the user explicitly at the rank-0
  contract for `finish_streaming_qc`.
- **Reviewer questions answered explicitly in `## Risks and Questions`.**
