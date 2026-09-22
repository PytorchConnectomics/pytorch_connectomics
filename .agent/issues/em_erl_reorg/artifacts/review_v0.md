# Review v0

## Summary

The reorganization is a clean, behavior-preserving relocation and is approved. Reusable
J0126 primitives now live in `em_erl` (`io`: URL normalizer / CloudVolume opener / skeleton
loader; `sampling`: `sample_cloudvolume_lut`; `eval`: LUT save/load/validate, LUT scoring,
generalized `evaluate_skeletons_cloudvolume`), `scripts/` → `examples/` via `git mv`, and the
j0126 example is a thin CLI keeping only the dataset default URL. Independently re-verified:
48 tests pass, all four examples run with `em_erl` uninstalled, and all ten new names are
exported. Raw notes: `state/review_v0.review.raw.md`.

## Diff Baseline

run_start_ref: 3549cad07165e9bcd3949501eb611228c66b628b

Review surface: working tree of the nested `lib/em_erl` repo vs run_start_ref. HEAD unchanged
(no driver commit).

## Findings

- **[resolved] `sys.path` bootstrap in all examples** — confirmed; all four run uninstalled.
- **[resolved] thin `run_j0126_eval` wrapper + package-side `evaluate_skeletons_cloudvolume`
  (no j0126 default URL)** — confirmed.
- **[confirmed] Monkeypatch point** — `evaluate_skeletons_cloudvolume` calls
  `open_seg_cloudvolume`/`sample_cloudvolume_lut` via the `em_erl.eval` module namespace; the
  reuse test (patching `em_erl.eval.open_seg_cloudvolume`) passes.
- **[confirmed] Behavior preserved** — `examples/volume_eval.py` end-to-end still prints the
  expected ERL; full suite green; no leftover `scripts/` references.
- No new findings.

## Tests to Add

None. The suite already covers the relocated API (sampler, URL normalizer, LUT round-trip
asserting no CloudVolume access on reuse, mismatch guard, mip/cache plumbing) now via package
imports.

## Questions

None.

## Verdict

VERDICT: APPROVE
