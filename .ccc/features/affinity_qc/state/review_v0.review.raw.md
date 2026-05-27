# review_v0 raw reviewer log (claude, in-session)

Per CCC protocol, in-session reviewer raw log. Reviewer = planner (claude),
same agent that wrote plan_v0/v1/v2.

## Inputs consulted

- `.ccc/features/affinity_qc/artifacts/code_v0.md` (codex coder output).
- `connectomics/decoding/qc/affinity.py` working-tree state.
- `connectomics/runtime/preflight.py` working-tree state.
- `connectomics/training/lightning/test_pipeline.py` working-tree state.
- `tests/unit/test_decoding_qc_affinity.py` working-tree state (with the
  later linter touchup noted in a system-reminder).
- Diff of `state/run_start.diff` vs current `git diff` (codex-incremental
  changes since CCC run-start baseline).
- Independent re-run of the four-target pytest suite: 41 passed.

## Mutation guard

Captured `git diff` and `git diff --cached` before and after the review
read operations. Both are byte-identical pre/post. No tracked or staged
content was mutated by the review.

## Plan_v2 item-by-item check

- **C1a (rank-0 guard in test_pipeline.py).** Implemented at lines 748–754.
  Uses `torch.distributed.is_available()/is_initialized()/get_world_size()/get_rank()`.
  Matches plan_v2 pseudocode. ✓
- **C1b (all-zero error wording in finalize).** Implemented at
  `affinity.py:116-120` with the rank-0 hint. Matches plan_v2 wording. ✓
- **C2 (subsampling + windowed scan + raw report arrays).** Implemented at
  `affinity.py:126-149` (raw `means`/`stds`/`z_idx`, sampling via
  `means_raw[::z_stride]` for `interior`, then windowed head/tail scan
  over raw `means_raw`). Returned `AffinityQCReport` exposes raw arrays
  (`z_idx_raw`, `means_raw`, `stds_raw`) per the new contract. ✓
- **C3 (finish_streaming_qc early image_path check + dead branch removal).**
  Implemented at `affinity.py:584-626`. Early raise on missing
  `image_path`, unconditional `build_affinity_mask` call afterwards, no
  `else:` fallback. ✓
- **C4 (preflight image_path OR affinity_mask_path).** Implemented at
  `preflight.py:286-317`. ✓
- **C4b (begin_streaming_qc short-circuit on pre-wired mask).** Implemented
  at `affinity.py:579-580`. ✓
- **C5 WITHDRAWN.** `AffinityQCAccumulator.finalize` at `affinity.py:114`
  still accepts `img=None` keyword. ✓
- **C6 (update docstring).** ⚠️ Implemented imprecisely. The new docstring
  is a single line "Fold one chunk slab into the running per-Z streaming
  statistics." which is SHORTER than the original plan_v0 docstring and
  does NOT spell out the `z_axis=-1` (post-save) vs `z_axis=1` (streaming)
  convention that plan_v2 C6 explicitly asked for. This is a [minor]
  regression from plan intent.
- **C7 (five new tests).** All five tests present and passing:
  - `test_streaming_interior_mean_matches_post_save` (random non-trivial
    volume, asserts `np.testing.assert_allclose(interior_mean)` and
    matching `low_z`/`high_z`).
  - `test_streaming_finalize_matches_post_save_when_head_window_fully_bad`
    (head-fill, `low_z=10, high_z=60`).
  - `test_streaming_accumulator_finalize_all_zero_message` (regex match
    of the rank-0 hint).
  - `test_preflight_streaming_image_path_rules` (covers all three
    branches: reject, accept-with-mask, accept-with-image).
  - `test_begin_streaming_qc_skips_when_mask_already_wired` (returns
    `None` when `affinity_mask_path` is non-empty).
  ✓

## V3 boundary check

`git diff connectomics/inference/` shows no addition of any
`from connectomics.decoding ...` import. The chunked stitcher still uses
the duck-typed `qc_streaming_callback: Any` parameter from earlier work.
`tests/unit/test_v3_guardrails.py::test_inference_static_imports_do_not_reference_decoding`
passes. ✓

## Diff baseline observations

- The CCC run started with a heavily dirty working tree (`run_start.diff`
  = 141 KB). Codex's stage modified the four affinity_qc files in scope
  PLUS the unrelated `scripts/decode_large.py` and `scripts/demo.py`.
  Inspecting the `decode_large.py` delta, the changes are worker-mode
  refactoring (SLURM array, `--worker-id`, `--wait` race handling)
  unrelated to affinity_qc. This is not in plan_v2's scope.
- Codex's `code_v0.md::Files Changed` table lists ONLY the four
  affinity_qc files plus the artifact — it does NOT claim
  `decode_large.py` or `demo.py`. Codex's `Risks` section explicitly
  acknowledges "Existing repository state was already dirty before this
  implementation; this stage did not attempt to separate or revert
  unrelated changes."
- Most likely explanation: the `decode_large.py` and `demo.py` deltas
  came from a parallel editing source (user keystrokes during a long
  CCC run, an external sync tool, or an IDE auto-formatter), not from
  codex.
- This is a [minor] diff-hygiene observation — outside code_v0's
  contract, but worth noting so future runs know to quiesce the working
  tree before starting CCC.

## Verification results

41 passed in 21.56s on independent re-run. Same as codex's run.
Validator: 14 canonical tutorials OK, 2 skipped custom workflows.

## Findings summary

- [minor] C6 docstring is less detailed than plan_v2 specified
  (no `z_axis` convention note).
- [minor] Working-tree drift in `scripts/decode_large.py`/`demo.py`
  appears unrelated to affinity_qc; the artifact's Files Changed table
  correctly omits them; flag as diff-hygiene only.

No major findings. C1, C2, C3, C4, C4b, C7 implemented faithfully. C5
withdrawal honored. Tests pass independently.

READY: yes (with minor comments).
