# Plan v2

## Summary

Terminal plan. Revises plan_v1 in response to its review. Two majors and
one minor addressed:

- **C7 swap:** drop the brittle `z_stride=5` head-fill divergence test
  in favor of a direct `interior_mean` numerical-equality test between
  `scan_prediction` and `AffinityQCAccumulator.finalize`. This is the
  deterministic regression guard for the C2 subsampling fix.
- **C4 → C4 + C4b:** preserve preflight's `image_path OR
  affinity_mask_path` rule, but **also** make `begin_streaming_qc`
  return `None` when `decoding.affinity_mask_path` is already
  non-empty. That keeps the runtime path consistent with preflight:
  pre-wired mask → no accumulator built → chunked stitcher folds no
  slabs → no `finish_streaming_qc` call → decode-stage short-circuit
  serves the pre-wired path.
- **C2 report-array contract:** the streaming `AffinityQCReport`
  exposes raw per-Z `z_idx`/`means`/`stds` (i.e. `len() == Z`), not
  the sampled subset used internally for `interior_mean`. Documented
  in the report dataclass docstring and in the markdown report prose.

C1, C3, C5 (withdrawn), and C6 carry over from plan_v1 unchanged.

## Scope

Same files as plan_v1; nothing added.

In scope:

- `connectomics/decoding/qc/affinity.py` — `finalize` subsampling +
  windowed scan; `begin_streaming_qc` honors pre-wired mask;
  `finish_streaming_qc` early `image_path` check; docstrings.
- `connectomics/training/lightning/test_pipeline.py` — rank-0 guard
  around `finish_streaming_qc`.
- `connectomics/runtime/preflight.py` — `image_path OR
  affinity_mask_path` rule.
- `tests/unit/test_decoding_qc_affinity.py` — three new tests
  (interior_mean equality, zero-update error, preflight rules).

Out of scope (unchanged from plan_v0/v1):

- Numerical-stability rewrite.
- `_xy_border_rows` structural refactor.
- Splitting `run_affinity_qc`.
- Removing legacy `dev/nisb/` CLIs.

## Proposed Changes

### C1 [major] Rank-0 guard for `finish_streaming_qc` — UNCHANGED FROM v1

Pseudocode for `test_pipeline.py` (gating the existing call):

```text
should_finalize_qc = True
if torch.distributed.is_available() and torch.distributed.is_initialized():
    if torch.distributed.get_world_size() > 1 and torch.distributed.get_rank() != 0:
        should_finalize_qc = False
if qc_acc is not None and should_finalize_qc:
    finish_streaming_qc(module.cfg, qc_acc)
```

Plus in `AffinityQCAccumulator.finalize`, refine the error:

```text
if (n == 0).all():
    raise ValueError(
        "AffinityQCAccumulator.finalize: no slabs were folded in. "
        "In distributed runs, finish_streaming_qc must only be called on rank 0."
    )
if (n == 0).any():
    missing = int((n == 0).sum())
    raise ValueError(
        f"AffinityQCAccumulator.finalize: {missing} Z slices were never updated"
    )
```

### C2 [major] Align streaming Z-cut with post-save — UNCHANGED FROM v1; report-array contract clarified

Algorithm (`finalize` body, in order):

```text
z_stride = params.z_stride
ke = params.k_edge
# raw per-Z curve (kept on the report for diagnostic value)
means_raw = (self._sum / n[:, None]).astype(np.float32)  # (Z, C)
var_raw   = self._sq / n[:, None] - (self._sum / n[:, None]) ** 2
stds_raw  = np.sqrt(np.maximum(var_raw, 0)).astype(np.float32)  # (Z, C)
z_idx_raw = np.arange(self.Z, dtype=np.int64)
# sampled population, mirrors `_per_z_scan(pred, z_stride)`
sampled = means_raw[::z_stride]                                 # (S, C)
if len(sampled) > 2 * ke + 1:
    interior = sampled[ke:-ke]
else:
    interior = sampled
interior_mean = interior.mean(axis=0)
cutoff = interior_mean - params.drift_thresh
# windowed head scan over raw z
head_end = min(params.refine_window, self.Z)
low_z = head_end
for z in range(head_end):
    if bool((means_raw[z] >= cutoff).all()):
        low_z = z
        break
# windowed tail scan over raw z
tail_start = max(0, self.Z - params.refine_window)
last_ok = -1
for z in range(tail_start, self.Z):
    if bool((means_raw[z] >= cutoff).all()):
        last_ok = z
high_z = last_ok + 1 if last_ok >= 0 else tail_start
```

**Report-array contract (new clarification):** The returned
`AffinityQCReport` carries `z_idx=z_idx_raw` (length Z) and
`means=means_raw`/`stds=stds_raw` (shape `(Z, C)`). Post-save
`scan_prediction` returns sampled rows (shape `(ceil(Z/z_stride), C)`).
This intentional informational difference is:

- **No effect on `low_z`/`high_z`** — both use sampled `interior_mean`
  and raw windowed scan; numerically equivalent.
- **Informational benefit** for streaming consumers (full curve at no
  extra cost).
- **Documented** in `AffinityQCReport`'s docstring with a one-line
  note: "Streaming finalize populates these arrays at raw stride 1;
  post-save scan populates at stride `z_stride`. `interior_mean` and
  cutoff math agree between paths."

### C3 [major] Remove dead-branch in `finish_streaming_qc` — UNCHANGED FROM v1

Early `image_path` check at top of `finish_streaming_qc`; drop the
`else:` branch. Net deletion: ~14 lines.

### C4 [minor] Preflight rule — UNCHANGED FROM v1

Require `image_path OR affinity_mask_path` when `mode='streaming'`.

### C4b [minor, NEW] `begin_streaming_qc` honors pre-wired mask

To eliminate the preflight/runtime mismatch identified in plan_v1_review,
add a short-circuit at the head of `begin_streaming_qc`:

```text
def begin_streaming_qc(cfg, *, channel_count, z_extent):
    decoding_cfg = _cfg_get(cfg, "decoding")
    qc_cfg = _cfg_get(decoding_cfg, "affinity_qc")
    if not _cfg_get(qc_cfg, "enabled", False):
        return None
    if _cfg_get(qc_cfg, "mode", "post_save") != "streaming":
        return None
    # NEW: skip streaming setup when a mask is already wired.
    if _cfg_get(decoding_cfg, "affinity_mask_path", "") or "":
        return None
    return AffinityQCAccumulator(channel_count, z_extent, _params_from_cfg(qc_cfg))
```

With this change, the path is:

- Pre-wired mask + streaming config → `begin_streaming_qc` returns
  `None` → chunked stitcher's `qc_streaming_callback=None` → no slabs
  folded → `qc_acc is None` in `test_pipeline.py` → `finish_streaming_qc`
  not called → decode stage's `run_affinity_qc` sees the pre-wired
  `affinity_mask_path`, returns it from the streaming short-circuit.
- Fresh streaming run → `begin_streaming_qc` returns an accumulator →
  slabs accumulate → `finish_streaming_qc` writes mask + wires path →
  decode stage finds the wired path and applies it.

### C5 [WITHDRAWN] — UNCHANGED FROM v1

Keep `AffinityQCAccumulator.finalize(img=...)` keyword accepted.

### C6 [minor] Tighten `AffinityQCAccumulator.update` docstring — UNCHANGED FROM v1

### C7 [REVISED AGAIN] Three new unit tests; replace divergence synthetic with direct equality

Reviewer noted that the `z_stride=5` head-fill divergence test is not
deterministic. Replace with a direct `interior_mean` equality test
(numerically asserts the C2 subsampling fix), keep the head-fill test
at `z_stride=1` (catches the windowed-loop bound bug), and add a
preflight test that covers both the rejection and exemption branches.

1. **`test_streaming_interior_mean_matches_post_save`** (replaces the
   `z_stride=5` variant):

   ```text
   import numpy as np
   rng = np.random.default_rng(0)
   pred = (rng.random((3, 8, 8, 100)).astype(np.float32) * 0.4 + 0.3)
   # Non-trivial; per-z means vary so sampled vs full interior diverge.
   params = AffinityQCParams(z_stride=5, k_edge=4, refine_window=30,
                             drift_thresh=0.05)
   ref = scan_prediction(pred, img=None, params=params)
   acc = AffinityQCAccumulator(channel_count=3, z_extent=100, params=params)
   acc.update(np.moveaxis(pred, -1, 1), z_offset=0, z_axis=1)
   streamed = acc.finalize()
   np.testing.assert_allclose(streamed.interior_mean, ref.interior_mean,
                              atol=1e-5)
   assert streamed.low_z == ref.low_z
   assert streamed.high_z == ref.high_z
   ```

   This passes only when the streaming `interior_mean` is computed from
   `means[::z_stride]`. Reverting C2's subsampling makes the
   `assert_allclose` fail on this volume (the populations differ).

2. **`test_streaming_finalize_matches_post_save_when_head_window_fully_bad`**
   (kept from plan_v1):

   ```text
   Z=60, refine_window=10, drift_thresh=0.05, z_stride=1, k_edge=4.
   pred = 0.6 everywhere except z=0..12 zeroed.
   Both scan_prediction and accumulator.finalize must return low_z=10,
   high_z=60.
   ```

   Catches a regression of the windowed loop bounds. With `z_stride=1`
   the subsampling fix is also exercised but degenerates to the full
   `means` array (sampled == means), so the test is robust either way.

3. **`test_streaming_accumulator_finalize_all_zero_message`** (kept
   from plan_v1):

   Build accumulator, call `finalize` immediately, assert error message
   matches "no slabs were folded in. In distributed runs,
   finish_streaming_qc must only be called on rank 0."

4. **`test_preflight_streaming_image_path_rules`** (combines the two
   branches into one test):

   ```text
   cfg = load_config("tutorials/neuron_nisb/liconn_banis_v3_erosion2.yaml")
   cfg.decoding.affinity_qc.enabled = True
   cfg.decoding.affinity_qc.mode = "streaming"
   cfg.inference.strategy = "chunked"
   # 4a: no image_path, no mask path → reject.
   with pytest.raises(ValueError, match="image_path|affinity_mask_path"):
       validate_runtime_coherence(cfg)
   # 4b: existing affinity_mask_path → accept.
   cfg.decoding.affinity_mask_path = "/tmp/preexisting.h5"
   validate_runtime_coherence(cfg)  # must not raise
   # 4c: image_path set instead → accept.
   cfg.decoding.affinity_mask_path = ""
   cfg.decoding.affinity_qc.image_path = "/tmp/dummy.zarr"
   validate_runtime_coherence(cfg)  # must not raise
   ```

5. **`test_begin_streaming_qc_skips_when_mask_already_wired`** (new for
   C4b):

   ```text
   from types import SimpleNamespace
   qc = SimpleNamespace(enabled=True, mode="streaming", ...all params...)
   decoding = SimpleNamespace(affinity_qc=qc,
                              affinity_mask_path="/tmp/dummy.h5")
   cfg = SimpleNamespace(decoding=decoding, inference=None)
   assert begin_streaming_qc(cfg, channel_count=3, z_extent=40) is None
   ```

## Files and Areas

| File | Section / Range | Change |
|---|---|---|
| `connectomics/decoding/qc/affinity.py` | `AffinityQCAccumulator.finalize` | C2 subsampling + windowed scan; C1b error wording; report-array contract documented |
| `connectomics/decoding/qc/affinity.py` | `AffinityQCReport` dataclass | docstring note on raw vs sampled arrays in streaming finalize |
| `connectomics/decoding/qc/affinity.py` | `begin_streaming_qc` | C4b short-circuit on pre-wired `affinity_mask_path` |
| `connectomics/decoding/qc/affinity.py` | `finish_streaming_qc` | C3 early `image_path` check, dead-branch removal |
| `connectomics/decoding/qc/affinity.py` | `AffinityQCAccumulator.update` | C6 docstring only |
| `connectomics/training/lightning/test_pipeline.py` | around line 749–750 | C1a rank-0 guard |
| `connectomics/runtime/preflight.py` | `affinity_qc` block | C4 rule |
| `tests/unit/test_decoding_qc_affinity.py` | append | C7 five new tests (interior_mean equality, head-window-bad, all-zero error, preflight rules, begin_streaming_qc mask wired) |

Estimated change size: ~75 net lines (≈ +40 logic, +35 tests).

## Verification Plan

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
2. Tutorial validator reports 14 canonical OK.
3. `test_streaming_interior_mean_matches_post_save` numerically
   asserts streaming and post-save `interior_mean` agree
   (`atol=1e-5`). Manual spot-check: temporarily revert C2's
   subsampling line in `finalize` → this test must fail with a
   numerical mismatch.
4. `test_preflight_streaming_image_path_rules` covers all three
   branches (reject / accept with mask / accept with image).
5. `test_begin_streaming_qc_skips_when_mask_already_wired` returns
   `None` and proves C4b's no-mismatch property.
6. The existing
   `test_run_affinity_qc_streaming_short_circuit_when_mask_already_wired`
   continues to pass.

Coordinator-side static check:

- `git diff connectomics/inference/` must not introduce any
  `from connectomics.decoding` import.
- `AffinityQCAccumulator.finalize` signature unchanged (still accepts
  `img=` kwarg).
- `AffinityQCReport` dataclass field set unchanged.

## Risks and Questions

Direct answers to plan_v1_review questions:

1. **Q.** Should a pre-wired `decoding.affinity_mask_path` disable
   streaming QC setup entirely?
   **A.** Yes. C4b implements this: `begin_streaming_qc` returns `None`
   when `affinity_mask_path` is non-empty, so no accumulator is built
   and `finish_streaming_qc` is never reached.

2. **Q.** Should streaming reports expose raw per-Z curves or match
   post-save sampled rows?
   **A.** Raw per-Z. The full curve is free for streaming and useful
   for diagnosing slice-level outliers. The `interior_mean` and cutoff
   math still agrees with post-save because the *internal* derivation
   subsamples. The report-array divergence is informational only.

Residual risks:

- **R.** `test_streaming_interior_mean_matches_post_save` uses a
  random seed (`default_rng(0)`). If a NumPy version shift changes the
  sequence, the test continues to pass as long as both code paths see
  the same array — both implementations operate on the same `pred`.
  Robust.

- **R.** `test_preflight_streaming_image_path_rules` mutates a loaded
  config dataclass three times in one test. If `validate_runtime_coherence`
  ever caches state, this could leak. Today it does not.

- **R.** plan_v2 is terminal (`p2-c2`). No further plan review; the
  workflow proceeds to `code_v0` immediately after `plan_v2.done`. The
  coder must read plan_v2 in full (no need for a separate plan
  review).

## Changes Since Previous Plan Version

- **C7 reshaped again (response to plan_v1_review major):** Replaced
  the non-deterministic `z_stride=5` head-fill variant with
  `test_streaming_interior_mean_matches_post_save`, a direct
  `np.testing.assert_allclose(interior_mean, …)` check. This is the
  only deterministic regression guard for C2's subsampling fix.
  Reverted the `z_stride=1` head-fill test stays.
- **C4b added (response to plan_v1_review major):** `begin_streaming_qc`
  now short-circuits to `None` when `decoding.affinity_mask_path` is
  pre-set. This eliminates the preflight/runtime mismatch where
  preflight would accept a config and chunked inference would still
  build an accumulator that later fails on the C3 `image_path` check.
- **C2 report-array contract clarified (response to plan_v1_review
  minor):** Documented that streaming `AffinityQCReport` exposes raw
  per-Z `z_idx`/`means`/`stds` (length Z), not sampled rows. Pinned
  this in the dataclass docstring. The `interior_mean` math agrees
  with post-save; the array shapes intentionally differ.
- **Test count grew from 3 → 5** to cover the new C4b path and the
  three-branch preflight rule.
- **Reviewer questions answered explicitly in `## Risks and Questions`.**
