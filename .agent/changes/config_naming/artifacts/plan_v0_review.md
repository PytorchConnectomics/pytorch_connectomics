# Plan v0 Review

## Summary

The plan covers the right broad surfaces and identifies most of the code paths that will need migration. It is not ready for implementation yet because the proposed naming rule is internally inconsistent and the output filename/path contract has contradictory examples. These are design-level issues; resolving them in the plan will reduce churn and prevent a partially migrated config surface.

## Findings

1. The naming convention is not actually applied uniformly to all four stage sections.

   The plan states that a root leaf is always `<verb>` or `<verb>_<noun>` and that a key without an underscore indicates a nested block. That does not match the proposed result: `evaluation` is declared a no-op even though it keeps root scalar leaves such as `metrics`, `prediction_threshold`, `instance_iou_threshold`, and `nerl_*`; `tune` also keeps root scalars such as `n_trials`, `timeout`, `trial_timeout`, `study_name`, `storage`, `load_if_exists`, `sampler`, `pruner`, and `optimization`. This conflicts with the task's requirement to apply one naming rule across `inference`, `decoding`, `evaluation`, and `tune`. The next plan should either narrow the rule explicitly to storage/load leaves only, or specify a complete rule for all root scalar leaves and explain why existing non-storage leaves are exempt.

2. The output filename token policy contradicts the examples and verification plan.

   The filename policy says to keep `_x{n}` because TTA pass count varies within one volume, but the proposed per-volume examples and verification expectation use `decoded_affinity_cc_numba-0-0.75.h5` / `final_prediction_output_tag(...) == decoded_affinity_cc_numba-0-0.75` with no `x1` token. The implementation needs one canonical answer for decoded outputs, raw outputs, and intermediate decoder outputs. The plan should define whether decoded filenames include the TTA/pass tag, and update all examples and tests to match.

3. The volume subdirectory rule is under-specified for decode-only and cache-hit paths.

   The plan describes resolving a volume stem from batch metadata or image paths, but several runtime flows do not have the original image batch when writing or reusing artifacts: explicit `decoding.input_prediction_path`, `inference.tta_result_path` cache loading, cache-only decode/postprocess/save, and tuning prediction cache reuse. Those paths currently derive identity from prediction filenames or configured directories. The next plan should define how `resolve_volume_save_dir` behaves for these flows so decode-only does not accidentally nest under the wrong directory or lose the input prediction identity.

4. The proposed `load_path` names are too generic for strict, stage-specific config.

   `inference.load_path` and `decoding.load_path` are symmetric, but they load different artifact types and can be confused with data loading. The current task explicitly calls out overloaded `output_path` semantics; replacing explicit `input_prediction_path` / `tta_result_path` with generic `load_path` risks recreating that problem on the input side. The plan should either choose artifact-specific names such as `load_prediction_path` / `load_tta_path`, or justify why generic `load_path` remains unambiguous in each section.

5. The tune schema migration needs a clearer dataclass/API decision.

   The plan removes `tune.output` and hoists its fields, while also listing the dataclass hierarchy skeleton as out of scope and saying no new dataclasses. That may be the right direction, but it is a public schema shape change and affects `TuneOutputConfig` exports, strict-key rejection, OmegaConf merge behavior for the `visualizations` and `report` dicts, and tests that instantiate `cfg.tune.output`. The next plan should state explicitly whether `TuneOutputConfig` is deleted, retained only internally, or replaced by flat fields on `TuneConfig`, and list the exact public API/test fallout to update.

## Questions

- Is the intended naming rule storage/load-only (`save_*` / `load_*`) or truly every root scalar leaf under the four stage sections?
- Should decoded filenames include the TTA/pass tag, or is pass count considered fully encoded by the raw prediction cache only?
- For decode-only mode from a single prediction file, should the volume directory be derived from the prediction filename, the parent directory, or an explicit config field?
- Would `decoding.load_prediction_path` and `inference.load_prediction_path` be clearer than `load_path`, even if slightly longer?

## Verdict

VERDICT: NEEDS_CHANGES
