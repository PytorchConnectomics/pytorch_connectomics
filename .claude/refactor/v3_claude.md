# V3 Refactor Plan (post-Codex v2 pass)

Audit run on commit `ba0f482`. Codex v2 round delivered the package layout
(`config/{schema,pipeline,hardware}`, `data/{io,augmentation,processing,datasets}`,
`decoding/{decoders,postprocessing,tuning}`, `evaluation/`, `runtime/`,
`training/{lightning,losses,optimization}`). What remains is *strictness*,
*stage separation*, *boundary integrity*, and *dead-code removal*. No
backward compatibility is preserved.

## Overview

| Theme | Goal | Net LOC | PRs |
|---|---|---|---|
| A. Dead code | delete unreachable / orphan modules | −5 KLOC | 1 |
| B. Boundary fixes | break decoding↔training and inference↔decoding loops | −1 KLOC, +moves | 4 |
| C. Strict config | drop ghost-field probes, raise on unknown keys | −0.5 KLOC | 1 |
| D. Stage separation | move PostprocessingConfig + EvaluationConfig out of `inference.py`; build out `connectomics.evaluation`; raw artifacts first | mixed | 3 |
| E. File splits | every file < 600 lines | mixed | 5 |
| F. Runtime extraction | move dispatch logic out of `scripts/main.py` and `lightning/utils.py` | +moves | 1 |
| G. Public API trim | minimize `__init__` exports | −small | 1 |
| H. Schema cleanup | rename architectures, drop legacy fields | tutorial migration | 1 |
| I. Docs refresh | rewrite `docs/source/*.rst` for v2 paths | docs only | 1 |

Implementation order: **A → C → G → D + B(eval) → B(tuner) + F → B(chunked) → B(decode-log) + B(config-data) → D(schema split) → E → H → I**.

---

## Theme A — Delete dead code

All verified by grep against `connectomics/`, `tests/`, `scripts/`,
`tutorials/`. Items marked **broken** would crash if called.

### Files to delete outright

| Path | Lines | Reason |
|---|---|---|
| `connectomics/decoding/tuning/auto_tuning.py` | 535 | broken imports `from .segmentation import decode_affinity_cc` at lines 219, 338, 457 (target file does not exist); only re-exported by `tuning/__init__.py`; zero callers |
| `connectomics/data/processing/blend.py` | 62 | zero importers (verified) |
| `connectomics/config/hardware/slurm_utils.py` | 389 | only `tests/test_banis_features.py` references; no production caller |
| `tests/unit/train_val_split.py` | 264 | misnamed: no `test_` prefix, no test functions, only `example_*` functions; depends on dead split helpers |
| `tutorials/mito_betaseg.yaml~` | — | editor backup, untracked |

### Symbols to delete

| Location | Symbols | Reason |
|---|---|---|
| `connectomics/config/hardware/auto_config.py` | `auto_plan_config`, `AutoConfigPlanner`, `AutoPlanResult` (~200 lines) | only `tests/integration/test_auto_config.py` |
| `connectomics/config/schema/monitor.py` | `WandbConfig`, `MonitorConfig.wandb` field (~25 lines) | no wandb consumer anywhere; no `WandbLogger` instantiation |
| `connectomics/models/losses/build.py` | `create_combined_loss`, `create_loss_from_config`, `CombinedLoss` (~120 lines) | no production callers |
| `connectomics/models/losses/losses.py` | `GANLoss` (~30 lines) | metadata `call_kind="unsupported"`; no use |
| `connectomics/training/lightning/model.py` | `create_lightning_module` (factory wrapper, no callers); `_compute_test_metrics` (no callers) | duplicates of `ConnectomicsModule(cfg, model)` |
| `connectomics/data/io/transforms.py` | `SaveVolumed`, `TileLoaderd` (~80 lines) | zero callers |
| `connectomics/data/datasets/split.py` | `create_split_masks`, `pad_volume_to_size`, `split_and_pad_volume`, `save_split_masks_h5`, `apply_volumetric_split` (~270 lines) | only the misnamed `tests/unit/train_val_split.py` |
| `connectomics/data/datasets/sampling.py` | `calculate_inference_grid` (~70 lines) | zero callers |
| `connectomics/data/processing/bbox.py` | `adjust_bbox`, `index2bbox`, `rand_window` (~50 lines) | zero callers |
| `connectomics/data/augmentation/transforms.py` | `RandMixupd` (~50 lines) | docstring concedes no-op for ndim<4; not in any tutorial |
| `connectomics/data/processing/transforms.py` | single-task wrappers `ComputeBinaryRatioWeightd`, `ComputeUNet3DWeightd`, `SegToFlowFieldd`, `SegToSynapticPolarityd`, `SegToSmallObjectd`, `SegSelectiond`, `SegDilationd`, `EnergyQuantized`, `DecodeQuantized` | duplicate `MultiTaskLabelTransformd._TASK_REGISTRY` entries |
| `connectomics/config/schema/__init__.py` | `DataTransformProfileConfig`, `EdgeModeConfig` exports | dead exports (Edge config still used internally; just drop from `__all__`) |
| `connectomics/config/schema/stages.py` | `TestConfig.output_path`, `TestConfig.cache_suffix` | outside the v2 stage allowlist; never read |
| `connectomics/training/lightning/utils.py` | `expand_file_paths` re-export (line 754) | duplicate path forbidden by v2 rule 1; canonical is `path_utils` |

### Net effect

≈ 5,000 LOC removed across 4 files + 13 symbol deletions. No behavior change
because none of these names are reachable from `scripts/main.py` or any v2
tutorial.

---

## Theme B — Boundary violations

### B1. `decoding/tuning/optuna_tuner.py` (1,911 lines) imports `connectomics.training.lightning`

Specifically: lazy imports `tta_cache_suffix`, `tuning_best_params_filename`,
`tuning_study_db_filename`, `tuning_best_params_filename_candidates` (lines
57–88); calls `trainer.test(model, datamodule=...)` at line ≈ 1687; reads
`cfg.inference.save_prediction.output_path` (lines 1604, 1648, 1840); embeds
final ARE/precision/recall reporting (`_print_results`, `_save_results`).

This is the worst offender in the package.

**Fix.** Split into:

| New module | Responsibility | Approx LOC |
|---|---|---|
| `connectomics/decoding/tuning/optuna_tuner.py` | `OptunaDecodingTuner` operating on saved arrays only | ~700 |
| `connectomics/decoding/tuning/search_space.py` | `_suggest_param`, `_sample_parameters`, `_reconstruct_decoding_params`, `_reconstruct_postproc_params` | ~250 |
| `connectomics/decoding/tuning/trial_runner.py` | timeout multiprocessing trial payload | ~250 |
| `connectomics/runtime/tune_runner.py` | `run_tuning`, `load_and_apply_best_params`, `_temporary_tuning_inference_overrides` (Lightning trainer lives here) | ~400 |

Pure tuner consumes a metric callable from `connectomics.evaluation`. No
imports of `connectomics.training` from anywhere under `connectomics/decoding/`.

### B2. `inference/chunked.py::run_chunked_affinity_cc_inference` imports decoding

Imports `decoding.pipeline.normalize_decode_modes` and
`decoding.decoders.segmentation.decode_affinity_cc`; runs decoding + CC
stitching inline.

**Fix.** Move `run_chunked_affinity_cc_inference` (and `_resolve_decode_affinity_cc_kwargs`)
to `connectomics/decoding/streamed_chunked.py` (or `runtime/chunked_decode.py`).
`inference/chunked.py` keeps only `run_chunked_prediction_inference` (raw-only).

Also remove `_validate_chunked_output_contract` postprocessing checks (lines
175–182): postprocessing should not be reachable from raw-prediction inference.

### B3. `training/lightning/test_pipeline.py` (2,005 lines) owns evaluation

Functions to move to `connectomics.evaluation`:

| Function | Lines |
|---|---|
| `_compute_instance_metrics` | 565–647 |
| `_compute_binary_metrics` | 650–693 |
| `_import_em_erl` | 743–757 |
| `_reorder_coordinate_axes` | 760 |
| `_networkx_skeleton_to_erl_graph` | 777 |
| `_load_nerl_graph` | 870 |
| `_nerl_node_positions` | 893 |
| `_prepare_nerl_segmentation` | 905 |
| `_extract_nerl_score_outputs` | 920 |
| `_compute_nerl_metrics` | 969 |
| `compute_test_metrics` | 1048–1113 |
| `log_test_epoch_metrics` | 1116–1284 |

Also move from `training/lightning/model.py`:

| Function | Lines |
|---|---|
| `_save_metrics_to_file` | 709–857 |
| `_create_metrics`, `_setup_test_metrics`, `_resolve_validation_target_slice` for the test-side metric instantiation only | 372–504 |

**New evaluation layout:**

```
connectomics/evaluation/
├── __init__.py            # public API (EvaluationConfig, EvaluationResult, run_evaluation_stage,
│                          # evaluate_segmentation_artifact)
├── stage.py               # rewritten — no Lightning callback, real metric resolver
├── metrics.py             # binary + instance metric resolution from torchmetrics names
├── nerl.py                # skeleton/ERL graph pipeline
├── report.py              # text/JSON/CSV writers
└── curvilinear.py         # already exists
```

Drops ~900 lines from training; removes 25 `module._*` private-method couplings.
TestContext (currently 56–110) grows to a complete dataclass; `module.log()`
becomes the only Lightning-module reference.

### B4. `training/lightning/model.py::_log_decode_experiment` imports decoding

Lazy-imports `decoding.pipeline.normalize_decode_modes` and
`resolve_decode_modes_from_cfg` (lines 859–974, 115 lines).

**Fix.** Move to `connectomics/decoding/experiment_log.py`. Hook into
`run_decoding_stage` post-execution.

### B5. `config/pipeline/config_io.py` imports `data.processing.build`

`data.processing.build` imports MONAI transform builders. Config import →
data execution machinery. Per v2 rule 5, config must not reach execution code.

**Fix.** Either (a) split `count_stacked_label_transform_channels` so the
counter lives without MONAI import, or (b) move cross-section channel
validation from `validate_config` into `runtime/preflight.py`. Option (b)
is cleaner: validation that needs domain knowledge runs outside config.

---

## Theme C — Strict config

Replace pervasive `getattr(cfg.x, "y", default)` with direct attribute
access. Schema dataclasses always populate fields; defensive defaults only
mask drift.

### Counts by file

| File | Patterns |
|---|---|
| `models/architectures/monai_models.py` | 27 |
| `config/pipeline/config_io.py` | 25 |
| `training/lightning/model.py` | 17 |
| `training/lightning/data_factory.py` | 17 |
| `models/architectures/mednext_models.py` | 17 |
| `models/architectures/rsunet.py` | 15 |
| `inference/sliding.py` | 12 |
| `inference/chunked.py` | 10 |
| `inference/output.py` | 9 |
| Total over connectomics/ | **184** |

### Specific ghost reads to delete

| Reference | File:line | Schema field |
|---|---|---|
| `cfg.inference.test_time_augmentation.act` | `inference/tta.py:537` | does not exist; `channel_activations` is the canonical path |
| `cfg.inference.output_act` | `inference/tta.py:551` | does not exist |
| `cfg.test.output_path / cache_suffix` | `config/schema/stages.py:50–51` | declared but never read |

Both `act` paths in `tta.py` (537–551) duplicate the strict
`channel_activations` path (488–531); deleting them removes a potential
double-activation when channel activations is empty but model output is
already sigmoid'd.

### Strictness policy fix

`config/pipeline/config_io.py:_warn_unconsumed_keys` (lines 98–115) calls
`warnings.warn` for unknown top-level keys. v2 rule 3 says
"removed fields raise during config load". Switch to `raise ValueError` with
the same typo-helpful message.

### Approach

One mechanical PR. For each `getattr(cfg.<section>, "<field>", <default>)`:
- if `<default>` matches the schema default → direct access
- if no schema field → either add to schema or delete the read entirely
- if genuinely optional (e.g., `cfg.test`, `cfg.tune` may be `None`) →
  keep `getattr(cfg, "tune", None)` only at the section root level

Cross-cutting tests (already partial in `tests/unit/test_v2_boundaries.py`)
should be expanded with an import-time grep check.

---

## Theme D — Stage separation

### D1. Top-level `decoding` config object

Currently `Config.decoding: Optional[List[DecodeModeConfig]]`. v2 wants
decoding to own its own config tree.

**New schema** (`connectomics/config/schema/decoding.py`):

```python
@dataclass
class DecodingConfig:
    steps: List[DecodeModeConfig] = field(default_factory=list)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    output_path: str = ""
    input_prediction_path: str = ""           # decode-only entrypoint
    tuning: Optional[DecodingTuningConfig] = None
```

Move from `config/schema/inference.py`:
- `PostprocessingConfig`, `BinaryPostprocessingConfig`, `ConnectedComponentsConfig`
- `DecodeModeConfig`, `DecodeBinaryContourDistanceWatershedConfig`

Delete from `InferenceConfig`:
- `postprocessing` field
- `decoding_path`
- `saved_prediction_path` (moves to `decoding.input_prediction_path`)

Update `decoding/stage.py::apply_decoding_postprocessing` to read from
`cfg.decoding.postprocessing` (currently reads `cfg.inference.postprocessing`).

### D2. Top-level `evaluation` config

Move `EvaluationConfig` from `config/schema/inference.py` into
`config/schema/evaluation.py`. Re-export from `connectomics.evaluation`.

### D3. Real evaluation stage

Currently `connectomics/evaluation/stage.py` takes a
`compute_metrics_fn: Callable` from the Lightning module — evaluation
cannot run without it. v2 plan: must run standalone.

**Rewrite** (per Theme B3):
```python
def run_evaluation_stage(
    decoded_path: str,        # or in-memory array (overload)
    gt_path: str,
    cfg: EvaluationConfig,
) -> EvaluationResult: ...

def evaluate_segmentation_artifact(seg, gt, cfg) -> EvaluationResult: ...
```

Internal metric resolver translates `EvaluationConfig.metrics` names into
torchmetrics callables; no callback, no Lightning dependency.

### D4. Inference produces raw `(C, Z, Y, X)` artifacts

- `inference/stage.py::run_prediction_inference` is currently a 30-line
  passthrough with no caller. Expand to be the canonical non-chunked
  entry: load model + run sliding window + write `write_prediction_artifact`
  with full metadata.
- `inference/chunked.py::run_chunked_prediction_inference` already writes
  raw — switch from manual h5 dataset creation to `write_prediction_artifact`
  so paths converge.
- `inference/output.py::_restore_prediction_to_input_space` (170–221) is a
  second postprocessing pass on inverse transpose/resample/crop. Move to
  `data` inverse pipeline or `decode` consumer reading the artifact.
- `PredictionArtifactMetadata` adds: `transpose: tuple[int,...] | None`,
  `model_architecture: str | None`, `model_output_identity: str | None`,
  `decode_after_inference: bool | None`. `chunked.py` actually populates
  `checkpoint_path` (currently None at lines 423–447, 617–642).

### D5. `decoding/postprocessing/postprocessing.py` doubly-nested

Rename to `connectomics/decoding/postprocess.py` (drop the subpackage).
Update `decoding/__init__.py` and consumers.

---

## Theme E — File-size and locality

Every file > 800 lines. Split candidates:

### `training/lightning/test_pipeline.py` (2,005)
- After Theme B3 (~−900 lines, 1,100 remaining):
  - `cached_pipeline.py` (cache hit → decode/eval branch)
  - `chunked_pipeline.py` (chunked raw + chunked affinity branches)
  - `prediction_crops.py` (`_apply_predecode_prediction_crops`,
    `_apply_affinity_inference_crop_if_needed`, `_resolve_inference_crop_pad`)
  - `test_pipeline.py` (TestContext + `run_test_step` orchestrator only)
- Target: each < 500 lines.

### `decoding/tuning/optuna_tuner.py` (1,911)
- Per Theme B1 — already split into 4 modules.

### `inference/tta.py` (1,547)
- `tta/combinations.py` (~270): augmentation enumeration helpers (lines 49–276)
- `tta/predictor.py` (~600): `TTAPredictor` + ensemble
- `tta/patch_first_local.py` (~280): patch-first-local TTA path (1000–1283)
- `tta/distributed.py` (~150): reduction helpers (shared with `lazy.py`)
- `tta/mask.py` (~150): mask handling (619–728, 1449–1498)

### `training/lightning/model.py` (1,366)
- After Theme B3 + B4 (~−400 lines):
  - core `LightningModule` (init/forward/training_step/validation_step/configure_optimizers)
  - `lightning/metrics_setup.py` (validation metric instantiation only)
- Target: < 700.

### `data/augmentation/transforms.py` (1,346)
- `transforms_geometric.py`: `RandAxisPermuted`, `RandRotate90Alld`,
  `RandElasticd`, `ResizeByFactord` (~250)
- `transforms_slicewise.py`: `RandSliceDropd`, `RandSliceShiftd`,
  `RandMisAlignmentd`, `RandMissingSectiond`, `RandMissingPartsd`,
  `RandMotionBlurd` (~500, EM-defect family)
- `transforms_intensity.py`: `RandMulAddIntensityd`, `RandCutNoised`,
  `RandCutBlurd`, `RandStriped`, `NormalizeLabelsd`,
  `SmartNormalizeIntensityd` (~350)
- `transforms_advanced.py`: `RandCopyPasted` (~200)
- `_helpers.py`: `_to_numpy`, `_from_numpy`, `_infer_*`, `_sample_*` (~140)

### `inference/lazy.py` (1,295)
- Extract `LazyVolumeAccessor` (377–787, ~410 lines) → `data/io/lazy_accessor.py`.
  Imports become local once moved.
- Collapse `lazy_predict_region` (855–1058) and `lazy_predict_volume`
  (1061–1295) onto a shared `_run_lazy_sliding(...)` core (~250 lines saved).
- `lazy.py` shrinks to ~600.

### `training/lightning/data_factory.py` (1,168)
- `data_factory/auto_download.py` (dataset download + interactive prompt)
- `data_factory/random_data.py` (random data generation)
- `data_factory/sdt_precompute.py` (label_aux precompute, includes the two
  `print()` calls at 117/131 → `logger.info`)
- `data_factory/datamodule_builders.py` (mode-specific datamodule construction)

### `training/lightning/callbacks.py` (1,001)
- `callbacks/visualization.py` (`VisualizationCallback`)
- `callbacks/ema.py` (`EMAWeightsCallback`)
- `callbacks/nan_detection.py` (`NaNDetectionCallback`)
- `callbacks/validation_reseed.py` (`ValidationReseedingCallback`)

Also fold `lightning/visualizer.py` (489 lines, single consumer) into
`callbacks/visualization.py`.

### `config/pipeline/config_io.py` (981)
- `pipeline/loader.py`: `load_config`, `merge_configs`, `update_from_cli`,
  `from_dict`, `_warn_unconsumed_keys`
- `pipeline/validation.py`: `validate_config`,
  `_validate_cross_section_coherence` (large, the main split target)
- `pipeline/paths.py`: `resolve_data_paths`
- `pipeline/naming.py`: `get_config_hash`, `create_experiment_name`,
  `print_config`, `to_dict`, `save_config`

### `data/processing/transforms.py` (979)
- `transforms_basic.py`: `SegToBinaryMaskd`, `SegToAffinityMapd`,
  `SegToInstanceBoundaryMaskd`
- `transforms_distance.py`: `SegToInstanceEDTd`, `SegToSkeletonAwareEDTd`,
  `SegToSemanticEDTd`
- `transforms_morph.py`: `SegErosiond`, `SegErosionInstanced`,
  `RelabelConnectedComponentsd`, `LeadingSpatialCropd`
- `multitask.py`: `MultiTaskLabelTransformd` (the complex 330-line core)

### `decoding/decoders/segmentation.py` (815)
- Extract `_compute_edt` (257–337, ~80 lines) →
  `connectomics/data/processing/distance.py` (where `edt_semantic`
  already lives).
- Extract `_connected_components_affinity_3d_numba` (554–693, ~140 lines)
  → `connectomics/decoding/decoders/_affinity_cc.py`.
- Remove `**kwargs` swallow in `decode_distance_watershed` (line 355) and
  `decode_waterz` (line 139). v2: no compatibility shims.
- `segmentation.py` shrinks to ~600.

---

## Theme F — Runtime/CLI extraction

`scripts/main.py` (1,107 lines) holds dispatch logic that should not live
in the entrypoint. `training/lightning/utils.py` (771 lines) holds
filename/cache helpers that are not Lightning-specific.

### New `connectomics/runtime/` modules

| Module | Source | Responsibility |
|---|---|---|
| `runtime/cli.py` | `lightning/utils.py:37–326` | `parse_args`, `setup_config` |
| `runtime/output_naming.py` | `lightning/utils.py:406–744` | `format_select_channel_tag`, `format_output_head_tag`, `format_decode_tag`, `format_checkpoint_name_tag`, `final_prediction_output_tag`, `tta_cache_suffix*`, `tuning_*_filename`, `is_tta_cache_suffix`, `resolve_prediction_cache_suffix` |
| `runtime/checkpoint_dispatch.py` | `scripts/main.py:135–200` + `_configure_checkpoint_output_paths`, `_setup_runtime_directories` | derive output base from checkpoint, extract step |
| `runtime/cache_resolver.py` | `scripts/main.py:203–384, 637–846` | `_resolve_cached_prediction_files`, `_is_valid_hdf5_prediction_file`, `_has_cached_predictions_in_output_dir`, `preflight_test_cache_hit`, `try_cache_only_test_execution`, `_handle_test_cache_hit` |
| `runtime/sharding.py` | `scripts/main.py:414–634` | `maybe_enable_independent_test_sharding`, `has_assigned_test_shard`, `shard_test_datamodule`, `maybe_limit_test_devices`, `resolve_test_rank_shard_from_env`, `resolve_test_image_paths`, `_estimate_tta_total_passes` |
| `runtime/tune_runner.py` | per Theme B1 | `run_tuning`, `load_and_apply_best_params` |
| `runtime/torch_safe_globals.py` | `config/schema/root.py:93–131` + `scripts/main.py:60–69` + `lightning/trainer.py:53–60` | one canonical implementation; called once from CLI entry |
| `runtime/preflight.py` | already exists | expand with cross-section validation moved out of `config_io.py` per Theme B5 |

### Result

- `scripts/main.py` shrinks to ~150 lines: parse → dispatch by mode →
  call `runtime.{train,test,tune}`.
- `lightning/utils.py` reduces to Lightning-specific helpers
  (`extract_best_score_from_checkpoint`, `compute_tta_passes`,
  `setup_seed_everything`).
- Three duplicate `torch.serialization.add_safe_globals` registrations
  collapse to one.
- `connectomics.decoding.tuning` no longer needs to lazy-import
  `connectomics.training.lightning.utils` — it pulls names from
  `connectomics.runtime.output_naming`.

---

## Theme G — Public API trim

### `connectomics/config/__init__.py`

Drop from `__all__`: `merge_configs`, `update_from_cli`, `to_dict`,
`from_dict`, `print_config`, `get_config_hash`, `create_experiment_name`,
`to_plain` (none on v2 list, most only test-referenced).

Add to `__all__`: `DefaultConfig`, `TrainConfig`, `TestConfig`, `TuneConfig`,
`SystemConfig`, `ModelConfig`, `DataConfig`, `OptimizationConfig`,
`MonitorConfig`, `InferenceConfig`, `DecodingConfig` (after Theme D1),
`EvaluationConfig` (after Theme D2). These are listed in the v2 plan but
currently buried under `config.schema`.

### `connectomics/training/lightning/__init__.py`

Drop from `__all__`: `compute_tta_passes`, `extract_best_score_from_checkpoint`,
`final_prediction_output_tag`, `is_tta_cache_suffix`, `parse_args`,
`resolve_prediction_cache_suffix`, `setup_config`, `setup_seed_everything`,
`tta_cache_suffix`, `tta_cache_suffix_candidates`, `expand_file_paths`.
After Theme F, these live in `runtime/`.

Keep: `ConnectomicsModule`, `ConnectomicsDataModule`, `SimpleDataModule`,
`create_trainer`, `create_datamodule`, `setup_run_directory`,
`cleanup_run_directory`, `modify_checkpoint_state`, the four callbacks.

### `connectomics/inference/__init__.py`

Drop from `__all__`: `resolve_inferer_overlap`, `resolve_inferer_roi_size`,
`apply_storage_dtype_transform`, `resolve_output_filenames`,
`write_prediction_artifact_attrs`.

Keep v2 list: `build_sliding_inferer`, `InferenceManager`,
`run_prediction_inference`, `run_chunked_prediction_inference`,
`write_prediction_artifact`, `read_prediction_artifact`,
`is_2d_inference_mode` + real consumer surface (`TTAPredictor`,
`apply_prediction_transform`, `write_outputs`,
`is_chunked_inference_enabled`).

### `connectomics/decoding/__init__.py`

Make `register_builtin_decoders()` lazy — call inside
`get_decoder`/`apply_decode_pipeline` instead of at module top-level.
Avoids importing waterz/abiss/numba on every `import connectomics.decoding`.

After Theme A, drop `auto_tuning` re-exports from `tuning/__init__.py`.

### `connectomics/models/__init__.py`

After Theme A, drop `create_combined_loss`, `create_loss_from_config`.

---

## Theme H — Schema and architecture cleanup

### Architecture rename

`connectomics/models/architectures/nnunet_models.py` registers
`nnunet_pretrained`. The `_pretrained` suffix violates v2 ("no `*_pretrained`
aliases"). The 2D/3D differentiator already lives in
`cfg.model.nnunet.spatial_dims`, and `pretrained` is a boolean.

**Action.** Rename to `nnunet`. Tutorials with `architecture: nnunet_pretrained`
update to `architecture: nnunet` and `cfg.model.nnunet.pretrained: true`.

### Drop `_MONAI_AVAILABLE` flag

`connectomics/models/architectures/__init__.py:42–66` has
`_MONAI_AVAILABLE` and `_MEDNEXT_AVAILABLE` and `_NNUNET_AVAILABLE`. MONAI
is a hard dependency (in `setup.py` core deps). The MONAI try/except is
dead.

**Action.** Remove MONAI guard. Keep MedNeXt and nnUNet guards (genuinely
optional).

### Augmentation aliases

`RandSliceShiftZd` (extends `RandMisAlignmentd`) and `RandSliceDropZd`
(extends `RandMissingSectiond`) — verified they are real subclass aliases
referenced in `build.py`. Two options:

- (A) Keep them as named transforms, document as "z-only variant of
  {parent}", remove from `__all__` mismatch (Theme G).
- (B) Fold parent + alias into single class with a `z_only: bool` flag.

(A) is less disruptive; (B) is cleaner. Pick A, defer B.

### Tutorial migration

YAML keys to rename in 40 tutorial files:

| Old | New |
|---|---|
| `inference.postprocessing.*` | `decoding.postprocessing.*` |
| `inference.saved_prediction_path` | `decoding.input_prediction_path` |
| `inference.decoding_path` | `decoding.output_path` |
| `model.architecture: nnunet_pretrained` | `model.architecture: nnunet` (+ `model.nnunet.pretrained: true`) |
| `monitor.wandb.use_wandb` | (deleted; section dropped) |

### Schema split target

After all D + H changes, `config/schema/inference.py` shrinks from 311 lines
to ~150 (just `InferenceConfig`, `SlidingWindowConfig`,
`TestTimeAugmentationConfig`, `PredictionTransformConfig`,
`SavePredictionConfig`, `ChunkingConfig`, `InferenceMemoryCleanupConfig`).
New files: `decoding.py` (~120 lines), `evaluation.py` (~30 lines).

---

## Theme I — Documentation refresh

`docs/source/*.rst` references v1 paths everywhere:

| File | Stale references |
|---|---|
| `tutorials/synapse.rst:33,109` | `connectomics.data.dataset.VolumeDataset` |
| `tutorials/neuron.rst:17` | `connectomics.data.dataset.VolumeDataset` |
| `tutorials/mito.rst:165` | `connectomics.data.dataset.TileDataset` |
| `notes/migration.rst:534` | "✅ YACS configs (still work, but deprecated)" — false; YACS is fully removed per CLAUDE.md |
| `notes/dataloading.rst:12,36,121–135` | `from connectomics.data.augment import *`, `TileDataset` doc section |
| `notes/faq.rst:23–28` | `VolumeDataset` vs `TileDataset` Q&A — both classes gone |
| `modules/model.rst` and `modules/models.rst` | duplicates referencing old paths `connectomics.models.arch`, `connectomics.models.loss`, `connectomics.models.utils` |

**Action.** Single PR rewriting each `.rst` file to reference v2 paths
(`connectomics.data.datasets.*`, `connectomics.data.augmentation.*`,
`connectomics.models.architectures`, `connectomics.models.losses`),
delete one of the model.rst/models.rst duplicates, drop the YACS reference.

---

## Implementation order (10 PRs)

| # | Theme | What | Risk | Tutorials touched |
|---|---|---|---|---|
| 1 | A | delete dead code | low | none |
| 2 | C | strict config (mechanical sweep) | low | none |
| 3 | G | public API trim + lazy decoder registration | low | none |
| 4 | D2 + B3 | build out `connectomics.evaluation`, move metric logic out of training | med | none |
| 5 | F + B1 | extract `connectomics.runtime`, split `optuna_tuner.py` | med | none |
| 6 | B2 | move chunked-affinity-CC out of inference | low | none |
| 7 | B4 + B5 | move decode-experiment log; move config↔data validation | low | none |
| 8 | D1 + D3 + D4 + D5 | schema split (`decoding.py`, `evaluation.py`); raw artifact first; postprocess module rename | high | all 40 |
| 9 | E | file splits | low | none |
| 10 | H + I | architecture rename + tutorial migration + docs refresh | med | all 40 |

PRs 1–7 are mergeable in any order. PR 8 carries the tutorial breakage and
must coordinate with users. PR 9 follows because file splits are easier to
review after the bigger structural moves land. PR 10 closes the loop with
docs and the final architecture rename.

## Estimated impact

- ≈ 7,000 LOC removed (5 KLOC dead code + 1 KLOC ghost-field probes + 1 KLOC duplicated paths)
- ≈ 5,000 LOC moved (training → evaluation, scripts → runtime, tuner orchestrator out of decoding)
- 0 files > 600 lines (currently 12)
- 0 lazy cross-domain imports (currently 5)
- 0 ghost-field reads of fields not in schema (currently 3 verified, more pending check)
- Decode-only and evaluate-only modes runnable without instantiating Lightning
- One canonical owner per concept; v2 architecture contract fully enforced

## Tradeoffs

- **Tutorial breakage**: Themes D and H rename YAML keys in 40 tutorial
  files. v2 contract accepts this.
- **Tune mode**: `run_tuning` moves from decoding to runtime. Cleaner but
  changes the `tune-test` CLI mode's call graph; `scripts/main.py` updates
  imports.
- **Test pipeline**: pulling evaluation out of `test_pipeline.py` removes
  25 `module._*` private-method couplings, but `TestContext` becomes the
  single Lightning-module reference; tests inspecting specific module
  methods will need updates.
- **Schema reorganization**: 79 dataclasses; moving
  `EvaluationConfig`/`PostprocessingConfig` may trigger torch
  safe-globals registration ordering. Already handled by reflective
  discovery in `_register_torch_safe_globals`; should be safe after
  Theme F consolidates the registration.
