# Plan v1 Review

## Summary

Plan v1 addresses the major v0 review findings: the naming rule is now scoped to storage/load leaves, input load names use artifact-specific nouns, decoded filename tokens are defined, tune output flattening has an explicit public-API decision, and cache behavior is specified. Two remaining inconsistencies should be fixed before implementation because they affect concrete file names and the exact schema edit target.

## Findings

1. The raw prediction filename policy still has a contradiction between the canonical table and tune layout.

   Section E says raw prediction artifacts are named `raw_x{n}{head}{ch}.h5` and chunked raw artifacts are also `raw_*`. Section F's tune layout then puts cached intermediate predictions under `predictions/<volume_stem>/tta_x1_ch0-1-2.h5`. Those appear to be the same artifact type: cached raw/intermediate model predictions. The code stage needs one canonical prefix for `intermediate_prediction_cache_suffix`, `tta_cache_suffix`, cache hit detection, tuning prediction caches, and raw test outputs. The next plan should choose `raw_` or `tta_` for cached prediction files and update Sections E, F, H, and I consistently.

2. The data schema change names classes that do not exist in the current schema.

   Section C proposes adding `name` to `TestDataConfig` and `ValDataConfig`, and Section H repeats that. The current schema has a shared `DataInputConfig` used by `DataConfig.train`, `DataConfig.val`, and `DataConfig.test`; there are no separate `TestDataConfig` or `ValDataConfig` classes. The next plan should state the exact implementation target: add `name` to `DataInputConfig` (and document that `train.name` becomes structurally available but is only used for val/test/tune outputs), or introduce separate split-specific dataclasses if that is truly intended. The latter is much larger and should not be implied accidentally.

3. The strict-key/deletion plan for `inference.chunks` and `inference.write_mode` needs one sentence on checkpoint/load fallout.

   The plan says these fields are dead and should be deleted, which looks reasonable from current code search. Because configs are serialized into checkpoints and hparams, deleting schema fields can affect checkpoint-load-safe globals or config reconstruction paths. The next plan should add a brief implementation note to check checkpoint/config deserialization tests or explicitly state that strict YAML rejection is the only targeted behavior. This is small, but it prevents surprise during code review.

## Questions

- Should all cached model predictions use `raw_` regardless of whether TTA was enabled, with `_x{n}` carrying the TTA count?
- If `name` is added to shared `DataInputConfig`, should the validator reject or ignore `data.train.name` to avoid implying train outputs use per-volume names?

## Verdict

VERDICT: NEEDS_CHANGES
