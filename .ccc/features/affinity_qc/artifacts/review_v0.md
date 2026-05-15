# Review v0

## Summary

Attested summary of `state/review_v0.review.raw.md` (in-session reviewer
= planner = claude). Codex implemented plan_v2 with high fidelity. C1
(rank-0 guard + zero-update error wording), C2 (subsampling + windowed
Z-cut + raw report-array contract), C3 (early `image_path` check +
dead-branch removal in `finish_streaming_qc`), C4 (preflight
`image_path` OR `affinity_mask_path`), C4b (`begin_streaming_qc`
short-circuits on pre-wired mask), and C7 (all five new tests)
all landed faithfully. C5 withdrawal honored (`AffinityQCAccumulator.finalize`
still accepts `img=`). Two [minor] observations: C6 docstring was
shortened rather than enriched as plan_v2 specified, and the
working-tree shows unrelated drift in `scripts/decode_large.py` /
`scripts/demo.py` that codex did not claim authorship of in its Files
Changed table.

The independent re-run of the verification suite shows 41 passed, 2
skipped custom workflows in the tutorial validator — matching codex's
own report. The v3 boundary
(`test_inference_static_imports_do_not_reference_decoding`) still
passes; `qc_streaming_callback` remains duck-typed. The mutation guard
(pre/post `git diff` and `git diff --cached` around the review reads)
was byte-clean.

Verdict: APPROVE with minor comments.

## Diff Baseline

run_start_ref: fa8fce67202394ab8b44d0f37857daabd8ba0a68

The CCC run started with a heavily dirty working tree (141 KB).
Reviewer compared the captured `state/run_start.diff` against the
current `git diff` to isolate codex's incremental edits. In-scope
files (`connectomics/decoding/qc/affinity.py`,
`connectomics/runtime/preflight.py`,
`connectomics/training/lightning/test_pipeline.py`,
`tests/unit/test_decoding_qc_affinity.py`) changed exactly per plan_v2.
Two out-of-scope deltas (`scripts/decode_large.py`,
`scripts/demo.py`) appear in the cumulative diff but are not claimed
by `code_v0.md`'s Files Changed table; most likely external editing
during the CCC run.

## Findings

- **[minor] C6 docstring less detailed than plan_v2 asked.**
  `AffinityQCAccumulator.update` is now documented as
  `"Fold one chunk slab into the running per-Z streaming statistics."` —
  a single line. Plan_v2 C6 explicitly asked for a docstring spelling
  out that `z_axis=-1` is the post-save `(C, X, Y, Z)` convention while
  streaming callers (chunked stitcher) pass `z_axis=1` because slabs
  there are `(C, Zslab, Y, X)`. The implementation removed that note
  entirely. Trivial follow-up: extend the docstring to include the two
  layout examples. Does not affect runtime behavior.

- **[minor] Working-tree drift in scripts/decode_large.py and scripts/demo.py
  unrelated to affinity_qc.**
  Worker-mode refactoring (SLURM array, `--worker-id`, `--wait` race
  handling) appears in the cumulative diff between `run_start.diff`
  and current `git diff`. Codex's `Files Changed` table correctly does
  NOT claim these files, and the `Risks` section explicitly
  acknowledges pre-existing dirty state. Treat as diff-hygiene
  observation, not a code_v0 finding. Recommend quiescing the working
  tree before future CCC runs.

- **[minor] V3 boundary preserved.** No new `from connectomics.decoding`
  static import in `connectomics/inference/`. `qc_streaming_callback`
  remains `Any` per the duck-typing contract.

## Tests to Add

None. The five new unit tests in `tests/unit/test_decoding_qc_affinity.py`
cover the algorithmic-equality property (C2), the C1 zero-update guard,
the preflight rules (C4), and the `begin_streaming_qc` pre-wired-mask
short-circuit (C4b). Plus the existing head-fill test (C2 windowed
loop) and the pre-existing parity test. Total 14 QC tests, all
passing.

## Questions

None.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
