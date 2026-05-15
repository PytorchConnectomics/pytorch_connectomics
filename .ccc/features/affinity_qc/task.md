# Task: Affinity QC – code review and update the implementation

## Goal

Conduct a code review of the recently added affinity-QC subsystem and update the
implementation in response to any review findings, while preserving the public
API and the v3 package boundary (`inference` must not statically import
`decoding`).

## Scope of the existing implementation under review

The relevant changes already exist in the working tree (uncommitted). Touch
points to review:

- `connectomics/config/schema/decoding.py` — new `AffinityQCConfig` dataclass
  on `DecodingConfig.affinity_qc` (fields: `enabled`, `mode`, `image_path`,
  `mask_path`, `report_path`, `z_stride`, `k_edge`, `refine_window`,
  `drift_thresh`, `border_width`, `bg_thresh`, `n_z_border`).
- `connectomics/decoding/qc/affinity.py` — new module owning
  `AffinityQCParams`, `AffinityQCReport`, `AffinityQCAccumulator`,
  `scan_prediction`, `render_markdown_report`, `build_affinity_mask`,
  `run_affinity_qc`, `begin_streaming_qc`, `finish_streaming_qc`.
- `connectomics/decoding/qc/__init__.py` — public exports.
- `connectomics/decoding/stage.py` — wires `run_affinity_qc(cfg, predictions)`
  before `_maybe_apply_affinity_mask` inside `run_decoding_stage`.
- `connectomics/inference/chunked.py` — added `qc_streaming_callback`
  parameter on `run_chunked_prediction_inference`,
  `_run_chunked_prediction_per_rank`, and `_stitch_chunk_prediction_files`.
  Calls `qc_streaming_callback.update(slab, z_offset, z_axis=1)` inline during
  chunked write. The parameter is typed `Any` — no static decoding import.
- `connectomics/training/lightning/test_pipeline.py` — at the
  `run_chunked_prediction_inference` call site, builds the accumulator via
  `begin_streaming_qc(cfg, channel_count, z_extent)` (computing `z_extent`
  via `resolve_global_prediction_crop`), threads it through
  `qc_streaming_callback`, and calls `finish_streaming_qc(cfg, acc)` after
  inference returns.
- `connectomics/runtime/preflight.py` — preflight rejects
  `affinity_qc.mode='streaming'` when `inference.strategy != 'chunked'`, and
  rejects unknown mode values.
- `dev/nisb/check_aff.py`, `dev/nisb/build_aff_mask.py` — refactored to thin
  CLI wrappers around the canonical module functions.
- `tests/unit/test_decoding_qc_affinity.py` — 9 tests covering post-save,
  streaming, mode validation, short-circuit.

## What the review must answer

1. **Correctness**
   - Does `AffinityQCAccumulator.finalize()` derive identical `low_z`/`high_z`
     to `scan_prediction(...)` on equivalent inputs? Test covers a synthetic
     case; are there pathological inputs (e.g. constant-zero volumes, fewer
     than `2*k_edge+1` z slices, single-channel) where they diverge?
   - Does the streaming path correctly handle the `z_axis=1` chunked layout
     (CZYX) vs `z_axis=-1` post-save layout (CXYZ)?
   - Are the per-channel mean/std formulas in `_per_z_scan` and the
     accumulator numerically equivalent (Welford vs sum/sumsq), or does
     float64 accumulation drift under realistic chunk sizes?

2. **Boundary preservation**
   - `inference/chunked.py` must remain free of static `connectomics.decoding`
     imports. The duck-typed callback parameter is deliberate. Re-verify that
     `test_inference_static_imports_do_not_reference_decoding` still passes.

3. **API and config surface**
   - `AffinityQCConfig` field choices: keep `mode` typed as `str` or
     introduce an enum/literal? Are all NISB-specific fields (`bg_thresh`,
     `border_width`) appropriately decoding-side rather than dataset-side?
   - Resolution order for `mask_path`/`report_path`: when both are empty,
     they default to `<decoding.save_path or inference.save_path>/...`.
     Is the precedence correct? What if neither is set?

4. **Failure modes**
   - `finish_streaming_qc` requires `image_path` when streaming because the
     mask builder needs the source image. Should it instead fall back to a
     Z-only mask using `accumulator.shape`, like the post-save path does?
   - Streaming mode failures during inference (e.g. OOM, NaN flood): does the
     accumulator state survive partial chunks gracefully, or do we leave a
     partial mask behind?

5. **Test coverage gaps**
   - Single-rank chunked path is wired via `test_pipeline.py:728` but has no
     integration test. Should we add a fake-cfg smoke test that exercises
     just the callback plumbing without full inference?
   - The `_xy_border_rows` helper has only one ad-hoc invocation; should it
     have direct tests?

6. **Code quality**
   - `_xy_border_rows` returns prose strings instead of structured data; the
     markdown renderer then re-formats. Should the data flow be refactored to
     a structured intermediate?
   - `run_affinity_qc` has a 50-line linear flow with mixed concerns
     (resolve paths → open image → scan → write report → write mask → wire
     path). Worth factoring further?

## Non-goals

- Streaming integration test that requires actual GPU inference.
- Replacing the dev CLI scripts; they are useful as ad-hoc QC tools.
- Touching unrelated v2/v3 cleanup (`evaluation` extraction etc.).

## Verification

After implementation changes:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
python -m pytest tests/unit/test_decoding_qc_affinity.py \
                 tests/unit/test_v3_guardrails.py \
                 tests/unit/test_v2_boundaries.py \
                 tests/unit/test_decoding_pipeline.py -q
python scripts/validate_tutorial_configs.py
```

All four should pass; tutorial validator should report 14 canonical configs OK.

## Constraints

- Do not commit during the CCC run (driver stage policy).
- Match existing pytc style (no emojis, no multi-paragraph docstrings,
  config accessors via `_cfg_get`, no facade re-exports per v3 contract).
- Keep changes surgical; per CLAUDE.md "every changed line should trace
  directly to the user's request".
