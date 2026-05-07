# Inference Config Refactor PR Summary

## Overview

This change refactors inference configuration ownership so model-output policy, execution/windowing, chunking, and storage are separate canonical sections. The main new YAML shape is `inference.model`, `inference.execution`, `inference.window`, `inference.chunking`, and `inference.save_inference`; the old mixed `output_array` section is removed from tutorials and schema.

## Implemented Behavior

- Added `InferenceModelConfig` for inference-time model-output decisions: `head`, `select_channel`, `output_dtype`, `activation_profile`, `channel_activations`, and `crop_pad`.
- Added `SaveInferenceConfig` for prediction artifact storage: `enabled`, `backend`, `output_path`, `cache_suffix`, `save_all_heads`, `dtype`, `compression`, `chunks`, and `write_mode`.
- Removed canonical `OutputArrayConfig`, `OutputArrayPartitionConfig`, and `OutputArrayStoreConfig` from the inference schema exports.
- Moved activation-profile expansion from `inference.test_time_augmentation.activation_profile` to `inference.model.activation_profile`.
- Moved TTA channel activation specs from `inference.test_time_augmentation.channel_activations` to `inference.model.channel_activations`.
- Moved output selection/crop semantics from flat inference fields and old `output_array` into `inference.model`.
- Kept runtime alias materialization in `sync_inference_runtime_aliases` so existing runtime code can still consume `sliding_window`, `save_prediction`, and narrow runtime fields after config load.
- Rejected runtime alias fields in user-facing config inputs (`inference.sliding_window`, `inference.save_prediction`, flat `inference.strategy`, etc.) and made config serialization/printing emit canonical sections only.
- Changed lazy sliding-window and patch-first TTA accumulation so value accumulators follow `inference.model.output_dtype`, while blending/normalization weight accumulators stay `float32`.
- Added a lazy-mode guard in `build_sliding_inferer`: when `lazy_load=true`, the eager MONAI inferer is not built, so lazy-only options such as `distance_transform` blending do not fail during initialization.
- Updated output writing/storage conversion to prefer `inference.save_inference.dtype` and `inference.save_inference.backend`.
- Updated model-output helper accessors and consumers for filename tags, artifact metadata, preflight validation, chunk crop/affinity offset resolution, TTA preprocessing, and prediction crop handling.
- Updated pipeline profiles so activation profiles and selected channels are emitted under `inference.model`.
- Updated `tutorials/neuron_nisb/*.yaml` to the new inference layout and removed old `output_array`, `accumulator_dtype`, `save_prediction`, `storage_dtype`, `output_formats`, and `sliding_window` YAML keys.
- Updated the remaining tutorials with the same old activation/save/window keys so repository tutorial validation uses only canonical inference YAML.
- Updated tutorial validation guidance for the new activation path.

## Testing

- `conda run -n pytc python -m py_compile ...` passed for the touched config, inference, runtime, training, and validation Python files.
- `conda run -n pytc isort --check-only ...` passed for the touched Python/test files.
- `conda run -n pytc black --check ...` passed for the touched Python/test files, with the existing Python 3.11 versus Python 3.12 target-version warning.
- `conda run -n pytc python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_nisb/*.yaml'` passed: 31 canonical tutorial configs validated, 2 custom workflow YAMLs skipped.
- `conda run -n pytc python scripts/validate_tutorial_configs.py --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'` passed: 38 canonical tutorial configs validated, 2 custom workflow YAMLs skipped.
- `conda run -n pytc pytest tests/unit/test_hydra_config.py tests/unit/test_lazy_inference.py tests/unit/test_inference_tta_masking.py tests/unit/test_inference_stage.py tests/unit/test_chunked_inference.py -q` passed: 93 tests passed.
- `conda run -n pytc pytest tests/unit/test_lit_utils.py tests/unit/test_connectomics_module.py tests/unit/test_test_pipeline_multi_volume_eval.py -q` passed: 62 tests passed.
- `conda run -n pytc pytest tests/unit/test_prediction_transform.py tests/unit/test_inference_stage.py tests/unit/test_hydra_config.py -q` passed: 54 tests passed after the `save_inference` output-write adjustment.

## Untested Areas

- Full repository test suite was not run.
- Actual GPU inference jobs were not run with the new config layout.
- Distributed/chunked HDF5 write behavior was not exercised end-to-end in this summary pass.
- The worktree contains additional dirty changes beyond this inference config refactor, including affinity/loss-mask, decoding/kernel, visualization, and augmentation changes; review should separate those concerns if this becomes a PR.

## Review Focus

- Verify the canonical schema split is clean: `inference.model` for model outputs, `inference.window` for sliding windows, `inference.chunking` for partitioning, and `inference.save_inference` for artifacts.
- Check that runtime aliases are still necessary only as internal materialization and are not reintroduced as canonical YAML paths.
- Inspect dtype behavior in lazy and patch-first TTA paths, especially value-accumulator `float16` precision and final division stability with fp32 weights.
- Confirm `save_inference.backend`/`dtype` correctly maps to existing artifact writers and cache naming without reviving `save_prediction` as a user-facing config path.
- Check tutorial migrations for semantic equivalence, especially NISB BANIS configs, `base_banis_chunk.yaml`, and configs with multi-head `head` plus channel activation specs.

## Changed Files

- `connectomics/config/schema/inference.py`: new inference model/storage schema, removed `output_array`, moved runtime alias sync.
- `connectomics/config/pipeline/profile_engine.py`: activation profile selectors now target `inference.model`.
- `connectomics/config/profiles/pipeline_profiles.yaml`: pipeline profiles now set activation/channel selection under `inference.model`.
- `connectomics/config/pipeline/config_io.py`: validation reads `inference.model.head` and reports new path names.
- `connectomics/utils/model_outputs.py`: added helpers for reading `inference.model` values.
- `connectomics/inference/lazy.py`, `connectomics/inference/tta.py`, `connectomics/inference/sliding.py`: model output dtype controls value accumulators/output casting; sliding-window weights stay fp32; lazy-only options skip eager inferer construction.
- `connectomics/inference/output.py`: storage dtype/backend now prefer `inference.save_inference`.
- `connectomics/inference/chunk_grid.py`, `connectomics/inference/artifact.py`, `connectomics/runtime/output_naming.py`, `connectomics/runtime/preflight.py`, `connectomics/training/lightning/prediction_crops.py`, `connectomics/training/lightning/model.py`: consumers updated to read inference model-output config.
- `tutorials/neuron_nisb/*.yaml`: migrated to the new inference section layout.
- `tutorials/*.yaml`, `tutorials/mitoEM/*.yaml`, `tutorials/neuron_snemi/*.yaml`: migrated old activation/save/window paths where needed for strict schema compatibility.
- `tests/unit/test_hydra_config.py`, `tests/unit/test_inference_tta_masking.py`, `tests/unit/test_lazy_inference.py`, `tests/unit/test_prediction_transform.py`, and related inference/Lightning tests: updated to assert the new paths and dtype behavior.
