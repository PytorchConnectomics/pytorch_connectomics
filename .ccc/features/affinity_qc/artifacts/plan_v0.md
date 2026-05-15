# Plan v0

## Summary

Audit of the affinity-QC subsystem found three correctness-class issues and
several minor cleanups. The largest is a distributed-runs bug where
`finish_streaming_qc` crashes on non-rank-0 workers; the second is a quiet
algorithmic divergence between the post-save Z-cut refinement
(`_refine_z_cuts`, windowed at the head/tail) and the streaming
`AffinityQCAccumulator.finalize` (full-curve linear scan); the third is a
dead `else` branch in `finish_streaming_qc` that claims to build a Z-only
mask but always raises. The plan applies surgical fixes to those three,
tightens preflight (streaming + `image_path` required), removes one dead
parameter (`finalize(img=...)`), and adds three targeted unit tests. The
existing public API (config schema field names, function signatures
exported from `decoding.qc`) does not change.

## Scope

In scope:

- `connectomics/decoding/qc/affinity.py` — finalize algorithm alignment,
  dead-branch cleanup, dead-parameter removal.
- `connectomics/training/lightning/test_pipeline.py` — gate
  `finish_streaming_qc` on rank-0.
- `connectomics/runtime/preflight.py` — require `image_path` when
  `affinity_qc.mode='streaming'`.
- `tests/unit/test_decoding_qc_affinity.py` — three new tests covering
  interior-outlier alignment, short-Z corner case, and preflight rejection.

Out of scope (deliberate — not raised by review or beyond surgical):

- Numerical-stability rewrite (`E[X²] − E[X]²` → Welford).
- Refactor of `_xy_border_rows` to structured output.
- Splitting `run_affinity_qc` into smaller functions.
- Removing the legacy CLI scripts under `dev/nisb/`.
- Streaming-mode integration test requiring real chunked inference.

## Proposed Changes

### C1 [major] Distributed rank-0 guard for `finish_streaming_qc`

**Bug.** In `connectomics/training/lightning/test_pipeline.py` (around
line 728–751 of the existing change), `begin_streaming_qc` is called on
every rank before `run_chunked_prediction_inference`, and
`finish_streaming_qc(module.cfg, qc_acc)` is called immediately after,
also on every rank. But the chunked stitcher (`_stitch_chunk_prediction_files`)
runs only on rank 0 in the distributed path (see
`connectomics/inference/chunked.py:_run_chunked_prediction_per_rank`,
`if rank == 0:` block around line 368–404), so non-rank-0 accumulators
receive zero `.update()` calls. Then `AffinityQCAccumulator.finalize()`
raises `ValueError("... N Z slices were never updated")`.

**Fix.** Two coordinated changes:

1. In `test_pipeline.py`, only call `finish_streaming_qc(...)` when
   `rank == 0` (use the same `torch.distributed.get_rank()` probe already
   present below the call).
2. As a belt-and-suspenders, change `AffinityQCAccumulator.finalize` to
   raise a distinct, clearer error when the accumulator is *entirely*
   unpopulated (`(self._n == 0).all()`), pointing the user at the
   rank-guard contract. This catches misuse from any future caller.

### C2 [major] Align streaming Z-cut algorithm with post-save

**Bug.** `scan_prediction` derives `low_z`/`high_z` via
`_refine_z_cuts`, which intentionally scans only the first
`refine_window` and last `refine_window` slices, leaving interior z
intact even if they drift. `AffinityQCAccumulator.finalize` walks the
*entire* per-Z mean curve linearly looking for the first/last slice with
`mean >= cutoff`. Result: on a volume with bad slices in the middle,
post-save reports `low_z=0, high_z=Z` while streaming reports a far
narrower range. The unit test
(`test_streaming_accumulator_matches_one_shot_scan`) currently passes
only because all bad slices in the synthetic case live within the
head/tail windows.

**Fix.** Replace the full-curve scan in `finalize` with a windowed scan
that mirrors `_refine_z_cuts`:

```text
head_end = min(refine_window, Z)
low_z = head_end                       # default: no good slice in head → cut entire head
for z in range(head_end):
    if (means[z] >= cutoff).all():
        low_z = z; break

tail_start = max(0, Z - refine_window)
high_z = tail_start                    # default: no good slice in tail → cut entire tail
last_ok = -1
for z in range(tail_start, Z):
    if (means[z] >= cutoff).all():
        last_ok = z
high_z = last_ok + 1 if last_ok >= 0 else tail_start
```

This makes the streaming finalize numerically equivalent to the
post-save `_refine_z_cuts` operating on the same `means` curve. The
existing alignment test stays meaningful; a new interior-outlier test
verifies the *windowed* (and now equivalent) behavior.

### C3 [major] Remove dead-branch in `finish_streaming_qc`

**Bug.** Lines 604–619 of `affinity.py` form an `else: ...` branch that
constructs `Z`, validates the range, makes a directory, then
unconditionally raises `ValueError("streaming affinity_qc requires
affinity_qc.image_path …")`. The h5py import and directory creation are
dead code; the error message is reached through a path that pretends to
have a fallback. The intended contract is "streaming requires
image_path".

**Fix.** Replace the dead branch with an early check at the top of
`finish_streaming_qc`:

```text
image_path, mask_path, report_path = _resolve_qc_paths(cfg, qc_cfg)
if not image_path:
    raise ValueError(
        "streaming affinity_qc requires decoding.affinity_qc.image_path; "
        "no in-memory prediction is retained to size the (X, Y, Z) mask."
    )
```

This collapses the function: do the I/O only when we know we can finish.

### C4 [minor] Preflight rejects streaming without `image_path`

To catch the C3 error before inference starts, extend
`validate_runtime_coherence` in `connectomics/runtime/preflight.py`:
when `decoding.affinity_qc.enabled` and `mode == 'streaming'`, require
`decoding.affinity_qc.image_path` to be non-empty. Otherwise raise with
"decoding.affinity_qc.mode='streaming' requires
decoding.affinity_qc.image_path to be set (post-stitch mask build reads
the source image)."

Note: post-save mode keeps the existing fallback that constructs a
Z-only mask from the in-memory prediction's `(X, Y, Z)` extent, so this
new rule applies only to streaming.

### C5 [minor] Remove dead `img=` parameter from `AffinityQCAccumulator.finalize`

The `img` parameter is declared but never read; the streaming path
hardcodes the "spatial pred not retained" message. Removing it:

- Simplifies the streaming finalize contract (no IO inputs).
- Means callers (just `finish_streaming_qc`) drop the unused arg.

### C6 [minor] Tighten `AffinityQCAccumulator.update` docstring

Default `z_axis=-1` is correct for the post-save axis convention but
opposite of the primary in-tree caller (chunked stitcher uses
`z_axis=1`). Tighten the docstring to spell out the two layouts
explicitly. No signature change; this only prevents future confusion.

### C7 [minor] Three new unit tests

Add to `tests/unit/test_decoding_qc_affinity.py`:

1. `test_streaming_finalize_matches_post_save_with_interior_outliers` —
   Build a `(C=3, X=8, Y=8, Z=60)` synthetic with bad z at indices
   `(0, 1, 30, 31, 58, 59)` and verify streaming and post-save produce
   the same `low_z`/`high_z`. This case proves C2 is fixed: both should
   ignore the interior 30/31 outliers because they lie outside both
   head and tail windows.
2. `test_streaming_accumulator_zero_updates_raises` — Build an
   accumulator, call `finalize` with no `update` calls, expect a clear
   error mentioning rank-0 / "no slabs were folded in" so a future
   caller failure is debuggable.
3. `test_preflight_streaming_requires_image_path` — load the config in
   memory, set `enabled=True, mode='streaming', strategy='chunked'`,
   leave `image_path=""`, call `validate_runtime_coherence`, expect a
   `ValueError` matching "requires …image_path".

## Files and Areas

| File | Section / Range | Change |
|---|---|---|
| `connectomics/decoding/qc/affinity.py` | `AffinityQCAccumulator.finalize` (≈ lines 109–164) | C2 windowed scan; C5 drop `img=` arg; clearer "zero updates" error per C1 belt-and-suspenders |
| `connectomics/decoding/qc/affinity.py` | `finish_streaming_qc` (≈ lines 562–622) | C3 early `image_path` check, remove dead `else` branch, drop `img=...` kwarg from `accumulator.finalize()` |
| `connectomics/decoding/qc/affinity.py` | `AffinityQCAccumulator.update` (≈ lines 79–107) | C6 docstring only |
| `connectomics/training/lightning/test_pipeline.py` | around the `finish_streaming_qc` call (≈ line 750) | C1 wrap in `if rank == 0:` using the same `torch.distributed` probe already used below |
| `connectomics/runtime/preflight.py` | `validate_runtime_coherence` (existing `affinity_qc` block) | C4 new check |
| `tests/unit/test_decoding_qc_affinity.py` | append | C7 three new tests |

Estimated change size: ~70 net lines (≈ +40 logic, +30 tests).

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
3. The new `test_streaming_finalize_matches_post_save_with_interior_outliers`
   passes both before-fix-on-synth (because edges) and after-fix-on-interior
   (because windowing); spot-check by temporarily reverting C2 and
   confirming the new test fails. (Manual.)
4. `test_preflight_streaming_requires_image_path` fails cleanly when C4
   is reverted.
5. The existing
   `test_run_affinity_qc_streaming_short_circuit_when_mask_already_wired`
   continues to pass — C3's tightening must not break it (it does not
   exercise `finish_streaming_qc`).

Manual sanity check (no code change required, but reviewer should
confirm):

- Re-read `connectomics/inference/chunked.py` for any static
  `from connectomics.decoding ...` imports. `qc_streaming_callback`
  must remain typed `Any`.
- Re-read the diff and confirm no new `getattr(..., default)` on
  declared dataclass fields (v3 strict-config rule).

## Risks and Questions

1. **Q.** Is the windowed Z-cut behavior actually what we want for
   streaming, or should both paths converge on the *full-curve* scan
   instead? **Recommendation:** windowed. The original `check_aff.py`
   was intentionally edge-only because real data has slow gradients
   inside the dataset that are not failure modes. Aligning streaming to
   match preserves the user-tuned `drift_thresh` semantics. If the
   reviewer prefers full-curve, swap both `_refine_z_cuts` and the
   accumulator finalize together (out of scope for this round).

2. **R.** `finish_streaming_qc` raising in C3 will surface as a
   user-visible failure for a config that currently silently fails at
   the same place — net is "fails earlier, with a clearer message". No
   regression in user-runnable configs.

3. **R.** C1's belt-and-suspenders error in `finalize` slightly changes
   the existing
   `test_streaming_accumulator_zero_updates_raises` *if* the existing
   test asserts a specific substring. The existing test asserts
   "N Z slices were never updated" — keep that phrasing, just refine
   the message when *all* slices are missing.

4. **Q.** Should we also add a `mode_choices` `Literal` typing to
   `AffinityQCConfig.mode`? The dataclass is read by `OmegaConf.structured`,
   which supports `Literal`. **Recommendation:** defer — current
   preflight rejection of unknown modes is sufficient, and `Literal`
   would tighten the schema in a way that's slightly orthogonal to this
   audit. Worth a separate ticket.

## Changes Since Previous Plan Version

Initial plan.
