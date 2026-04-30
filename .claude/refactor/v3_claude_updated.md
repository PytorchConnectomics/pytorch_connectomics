# V3 Refactor Plan (revised after Codex feedback)

Audit run on commit `ba0f482`. Codex v2 round delivered the package layout
(`config/{schema,pipeline,hardware}`, `data/{io,augmentation,processing,datasets}`,
`decoding/{decoders,postprocessing,tuning}`, `evaluation/`, `runtime/`,
`training/{lightning,losses,optimization}`). What remains is *boundary
integrity*, *strict config*, *stage separation*, and *dead-code removal*.
No backward compatibility is preserved.

This revision incorporates Codex feedback on the original v3 plan
(`v3_claude_feedback.md`). Three things changed:

1. **Dead-code claims were too broad.** Several "zero caller" items in the
   original plan are public API exports or test-covered features. Theme A
   is now split into three risk buckets, and items that change accepted
   surface or product behavior are routed through later PRs.
2. **PR ordering was unsafe.** "PRs 1–7 mergeable in any order" was wrong:
   public API trim depends on new homes existing, schema field deletion
   depends on tutorial migration, tuner split depends on runtime naming
   extraction. Order is now strict and dependency-driven.
3. **Performance was understated.** "Clean and high-performance" needs
   concrete targets, not just architecture cleanup. New Theme P captures
   them and lands as part of the boundary moves rather than a separate
   sweep.

## Themes

| Theme | Goal | Risk |
|---|---|---|
| 0. Guardrails | boundary tests, strict-config tests, public-API snapshot tests, smoke benchmarks | none — adds tests |
| A1. Safe dead artifacts | delete only files that are unreachable AND unexported | low |
| A2. Public API removals | delete exported names with explicit `__init__` and test updates | medium |
| A3. Product decisions | features that "look dead" but require an explicit decision | requires sign-off |
| B. Boundary fixes | decoding↛training, inference↛decoding, config↛data execution | medium |
| C. Strict config | raise on unknown keys; delete verified ghost reads (not all 184) | low |
| D. Stage separation | top-level `decoding`/`evaluation` schemas; raw artifacts first | high (tutorials) |
| E. File splits | secondary smell-check, after ownership is fixed | low |
| F. Runtime extraction | move dispatch out of `scripts/main.py` and `lightning/utils.py` | low |
| G. Public API trim | minimize `__init__` exports after homes exist | low |
| H. Architecture rename + tutorial migration | `nnunet_pretrained → nnunet`; YAML key migration | atomic with D |
| I. Docs refresh | v1 paths and dead Sphinx pages | docs only |
| P. Performance | dtype round-trips, HDF5 chunking, TTA memory, streamed write paths | benchmark-gated |

Implementation order is the strict dependency chain, not the theme order.
See "PR sequence" below.

---

## Honest reassessment of the original Theme A

The original Theme A treated 13 items as "dead". Verified against grep over
`connectomics/`, `tests/`, `scripts/`, `tutorials/`, **only some are
truly orphan**. Reclassification:

### A1. Truly safe to delete (unreachable AND unexported)

| Path / symbol | Lines | Evidence |
|---|---|---|
| `connectomics/decoding/tuning/auto_tuning.py` | 535 | `from .segmentation import decode_affinity_cc` at 219/338/457 — target file does not exist; only re-exported via `tuning/__init__.py`; would `ImportError` on first call |
| `connectomics/data/processing/blend.py` | 62 | zero importers anywhere |
| `connectomics/data/datasets/sampling.py::calculate_inference_grid` | ~70 | zero callers |
| `connectomics/data/processing/bbox.py::adjust_bbox / index2bbox / rand_window` | ~50 | zero callers |
| `tutorials/mito_betaseg.yaml~` | — | editor backup, untracked |

Plus PR-1 cleanup of `tuning/__init__.py` re-exports of the deleted
`auto_tuning` symbols (`SkeletonMetrics`, `grid_search_threshold`,
`optimize_parameters`, `optimize_threshold`).

### A2. Public API removals (require export + test updates)

These are exported from `__init__` files. Removing them changes the
package surface; must land with explicit export updates and snapshot tests.

| Symbol | Exported from | Notes |
|---|---|---|
| `SaveVolumed`, `TileLoaderd` | `connectomics.data.io` | no production caller, but exported |
| `create_combined_loss`, `create_loss_from_config`, `CombinedLoss` | `connectomics.models.losses` | no production caller, but exported |
| `DataTransformProfileConfig`, `EdgeModeConfig` | `connectomics.config.schema` | exported, no consumer for `DataTransformProfileConfig`; `EdgeModeConfig` is still used internally — drop only the `__all__` entry |
| `to_plain` | `connectomics.config` | exported, no consumer |
| `merge_configs`, `update_from_cli`, `to_dict`, `from_dict`, `print_config`, `get_config_hash`, `create_experiment_name` | `connectomics.config` | exported, only test references; v2 `config_v2.md` lists none of these |
| `expand_file_paths` re-export at `lightning/utils.py:754` | `connectomics.training.lightning.utils` | duplicate path; canonical is `path_utils` |
| `unregister_architecture` | `connectomics.models.architectures.registry` | tests-only; keep documented as testing-only or drop |

These belong in PR 9 (Public API Trim), after new homes exist.

### A3. Product / feature decisions (require explicit sign-off)

Codex flagged these correctly: deletion is a behavior change, not a
cleanup. They need a decision before the change, ideally with the project
maintainers.

| Item | Why it isn't dead | Required decision |
|---|---|---|
| `RandMixupd` | imported by `data/augmentation/__init__.py`, registered in `build.py:965`, exercised in `tests/unit/test_em_augmentations.py:599-609` | keep, fix to work for ndim<4, or formally deprecate |
| `auto_plan_config` / `AutoConfigPlanner` / `AutoPlanResult` | exercised in `tests/integration/test_auto_config.py`; called from no script today, but is a planned CLI feature | keep behind `--auto-plan` flag, or remove with the test |
| `slurm_utils` | exercised in `tests/test_banis_features.py`; documents intended SLURM helpers | keep, or move to `scripts/` and remove the schema package surface |
| `WandbConfig` | no `WandbLogger` consumer wired today, but the schema field accepts user input. Deletion is a config-surface break | route through PR 8 (schema-break) with tutorial validation, or restore the wandb consumer |
| `GANLoss` | exported from `connectomics.models.losses`, registered with `call_kind="unsupported"`. Either it's a placeholder for future training loops or it's truly orphan | confirm with maintainers; if truly orphan, route through PR 9 |
| `single-task wrappers` (`ComputeBinaryRatioWeightd`, `ComputeUNet3DWeightd`, `SegToFlowFieldd`, `SegToSynapticPolarityd`, `SegToSmallObjectd`, `SegSelectiond`, `SegDilationd`, `EnergyQuantized`, `DecodeQuantized`) | `SegToFlowFieldd` and `ComputeBinaryRatioWeightd` are exercised in `tests/unit/test_monai_transforms.py`; they overlap with `MultiTaskLabelTransformd._TASK_REGISTRY` but are not strictly dead | decide: route through registry and delete wrappers in PR 8/9, or keep wrappers as the public single-use API |
| `TestConfig.output_path / cache_suffix` | set in `tests/unit/test_nnunet_preprocessing.py:87`; may have stage fallback in `inference/output.py` | confirm fallback before deletion; route through PR 8 |
| `create_lightning_module` (model.py factory) | wraps `ConnectomicsModule(cfg, model)`; no caller today | keep as documented factory, or delete in PR 9 |
| `lightning/visualizer.py` (489 lines) | single consumer (`callbacks.py`), but it's the working visualization implementation | fold into callbacks during PR 10 file splits, not as a dead-code delete |
| `data/datasets/split.py` orphan helpers (`create_split_masks`, `pad_volume_to_size`, `split_and_pad_volume`, `save_split_masks_h5`, `apply_volumetric_split`) | only consumer is `tests/unit/train_val_split.py` (which is itself misnamed and contains no `test_*` functions, only `example_*`) | decide whether `train_val_split.py` is an example to keep (move to `examples/` or `notebooks/`) or to delete; helper deletion follows |
| `_compute_test_metrics` (model.py) | no caller today; was probably called by an earlier path | delete in PR 4 (Evaluation Extraction) along with the rest of the test-metric machinery |

The original plan's "≈ 5,000 LOC removed" estimate was inflated. Realistic
PR-1 (A1) deletions sum to ≈ 700 LOC plus the broken `auto_tuning.py`
(535) — about 1.2 KLOC of truly orphan code. The remaining cleanup is
real but lives in PRs 4, 8, 9, 10 where it can be paired with tests and
schema migration.

---

## Theme B — Boundary fixes (the structural work)

Same structure as the original plan; risk and ordering revised below.

### B1. `decoding/tuning/optuna_tuner.py` (1,911 lines) imports `connectomics.training`

Lazy imports of `tta_cache_suffix`, `tuning_best_params_filename`,
`tuning_best_params_filename_candidates`, `tuning_study_db_filename`
(lines 57–88). Calls `trainer.test(...)` near line 1687. Reads
`cfg.inference.save_prediction.output_path` (1604, 1648, 1840). Embeds
final ARE/precision/recall reporting.

**Fix.** Two-step, not one:

- *Step 1 (PR 2, runtime naming extraction):* move naming helpers
  (`tta_cache_suffix`, `tuning_best_params_filename*`, `tuning_study_db_filename`,
  `format_*_tag`, `final_prediction_output_tag`,
  `resolve_prediction_cache_suffix`, `is_tta_cache_suffix`) from
  `training/lightning/utils.py` to `connectomics/runtime/output_naming.py`.
  Update `optuna_tuner.py`, `scripts/main.py`, `model.py`,
  `test_pipeline.py` imports. **Tuning behavior unchanged in this step.**
- *Step 2 (PR 5, tuning runtime split):* move `run_tuning`,
  `load_and_apply_best_params`, `_temporary_tuning_inference_overrides`
  to `connectomics/runtime/tune_runner.py`. The pure tuner stays in
  `decoding/tuning/optuna_tuner.py` and operates on saved arrays only.
  File splits within `optuna_tuner.py` (search-space, trial-runner) are
  optional and live in PR 10.

This split protects against the ordering trap the feedback flagged: you
cannot extract `run_tuning` cleanly until naming helpers no longer cross
the decoding↔training boundary.

### B2. `inference/chunked.py::run_chunked_affinity_cc_inference` imports decoding

Imports `decoding.pipeline.normalize_decode_modes` and
`decoding.decoders.segmentation.decode_affinity_cc`; runs decoding +
CC stitching inline.

**Fix (PR 6).** Move `run_chunked_affinity_cc_inference` and
`_resolve_decode_affinity_cc_kwargs` to
`connectomics/decoding/streamed_chunked.py` (preferred, since the function
is decoder-aware) or `connectomics/runtime/chunked_decode.py` (alternative
if it grows orchestration glue). `inference/chunked.py` keeps
`run_chunked_prediction_inference` (raw-only).

Also remove `_validate_chunked_output_contract` postprocessing checks
(175–182): postprocessing is not reachable from raw-prediction inference
once B2 lands.

### B3. `training/lightning/test_pipeline.py` (2,005 lines) owns evaluation

Functions to move into `connectomics.evaluation`:

| Function | Lines | Target |
|---|---|---|
| `_compute_instance_metrics` | 565–647 | `evaluation/metrics.py` |
| `_compute_binary_metrics` | 650–693 | `evaluation/metrics.py` |
| `_import_em_erl` | 743–757 | `evaluation/nerl.py` |
| `_reorder_coordinate_axes`, `_networkx_skeleton_to_erl_graph`, `_load_nerl_graph`, `_nerl_node_positions`, `_prepare_nerl_segmentation`, `_extract_nerl_score_outputs`, `_compute_nerl_metrics` | 760–1046 | `evaluation/nerl.py` |
| `compute_test_metrics` | 1048–1113 | `evaluation/stage.py` (new entry) |
| `log_test_epoch_metrics` | 1116–1284 | `evaluation/report.py` |

Plus from `model.py`:

| Function | Lines | Target |
|---|---|---|
| `_save_metrics_to_file` | 709–857 | `evaluation/report.py` |
| validation/test metric instantiation in `_create_metrics`, `_setup_test_metrics` | 372–504 | partial — keep validation in `model.py`; move test instantiation to `evaluation/metrics.py` |

**Rewrite of `evaluation/stage.py`.** Current implementation requires a
`compute_metrics_fn` callback from the Lightning module. Replace with a
real metric resolver that translates `EvaluationConfig.metrics` names into
torchmetrics callables; no callback parameter.

```python
def run_evaluation_stage(decoded_path, gt_path, cfg) -> EvaluationResult: ...
def evaluate_segmentation_artifact(seg, gt, cfg) -> EvaluationResult: ...
```

**Result.** Drops ≈ 700–900 lines from training. Removes 25 `module._*`
private-method couplings (currently: `_get_runtime_inference_config`,
`_get_test_evaluation_config`, `_is_test_evaluation_enabled`, `_cfg_float`,
`_cfg_value`, `_resolve_test_output_config`, `_save_metrics_to_file`,
`_get_prediction_checkpoint_path`, `_load_cached_predictions`,
`_summarize_tta_plan`). Decode-only and evaluate-only modes become
runnable without instantiating `ConnectomicsModule`.

### B4. `model.py::_log_decode_experiment` lazy-imports decoding

Lines 859–974 (115 lines). Lazy-imports
`decoding.pipeline.normalize_decode_modes` and `resolve_decode_modes_from_cfg`.

**Fix (PR 5 or PR 7).** Move to `connectomics/decoding/experiment_log.py`,
hooked into `run_decoding_stage` post-execution.

### B5. `config/pipeline/config_io.py` imports `data.processing.build`

`data.processing.build` imports MONAI transform builders. Config import
chain → MONAI execution machinery.

**Fix (PR 7).** Move cross-section channel validation from
`validate_config` into `runtime/preflight.py`. Validation that needs
domain knowledge runs outside config; config import stays free of MONAI.

(Alternative considered: split `count_stacked_label_transform_channels` so
the counter doesn't pull MONAI. Rejected because the counter logic
naturally belongs near other preflight checks anyway.)

---

## Theme C — Strict config (scope-corrected)

Original plan called for replacing all 184 `getattr(cfg.x, "y", default)`
patterns. Per feedback, this is too aggressive in one PR. Revised scope:

### C1. Verified ghost reads of fields not in schema (PR 3)

Three confirmed:

| Reference | Location | Schema state |
|---|---|---|
| `cfg.inference.test_time_augmentation.act` | `inference/tta.py:537` | not in schema; canonical is `channel_activations` |
| `cfg.inference.output_act` | `inference/tta.py:551` | not in schema |
| `cfg.test.output_path / cache_suffix` | `config/schema/stages.py:50–51` | declared but Codex flagged stage-fallback behavior may exist; verify before delete |

Both `act` paths in `tta.py` (537–551) duplicate the strict
`channel_activations` path (488–531). Removing them eliminates a potential
double-activation when `channel_activations` is empty but the model output
is already sigmoid'd. Land in PR 3.

### C2. Strictness policy fix (PR 3)

`config/pipeline/config_io.py:_warn_unconsumed_keys` (98–115) calls
`warnings.warn` for unknown top-level keys. v2 architecture rule 3 says
"removed fields raise during config load". Switch to `raise ValueError`.
Add a strict-config test in PR 0 that asserts unknown keys raise.

### C3. Bulk getattr cleanup (deferred to per-domain PRs)

The remaining ~180 `getattr(cfg.…, "…", default)` patterns are real but
mostly tolerable. Sweeping all of them in one PR conflicts with the schema
work in PR 8 and risks breaking tutorial loads silently.

Revised approach: address bulk getattr cleanup *inside each domain PR*
that touches that domain. So:

- PR 4 (evaluation extraction): clean the 17 in `model.py` and the
  evaluation portion of `test_pipeline.py`
- PR 5 (tune runner extraction): clean the 8 in `optuna_tuner.py`
- PR 6 (chunked decode boundary): clean the 10 in `inference/chunked.py`
  and the 12 in `inference/sliding.py`
- PR 8 (schema split): clean the 25 in `config_io.py`
- PR 9 (public API trim): clean architecture-builder probes
  (27 in `monai_models.py`, 17 in `mednext_models.py`, 15 in `rsunet.py`,
  6 in `nnunet_models.py`)

This trades a single mechanical PR for incremental cleanup, but each
cleanup lands with the structural change that motivates it.

---

## Theme D — Stage separation (atomic with H per feedback)

### D1 + H. Schema split + tutorial migration (PR 8)

Per feedback: D and H must be one PR. Removing schema fields and renaming
YAML keys in the same atomic change avoids the window where tutorials
fail to load.

**New schema** (`connectomics/config/schema/decoding.py`):

```python
@dataclass
class DecodingConfig:
    steps: List[DecodeModeConfig] = field(default_factory=list)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    output_path: str = ""
    input_prediction_path: str = ""
    tuning: Optional[DecodingTuningConfig] = None
```

**New schema** (`connectomics/config/schema/evaluation.py`): move
`EvaluationConfig` here from `inference.py`.

**Move from `config/schema/inference.py` to `decoding.py`:**
- `PostprocessingConfig`, `BinaryPostprocessingConfig`, `ConnectedComponentsConfig`
- `DecodeModeConfig`, `DecodeBinaryContourDistanceWatershedConfig`

**Delete from `InferenceConfig`:**
- `postprocessing` (moves to `decoding.postprocessing`)
- `decoding_path` (moves to `decoding.output_path`)
- `saved_prediction_path` (moves to `decoding.input_prediction_path`)

**Rename architecture (H):** `nnunet_pretrained` → `nnunet`. The 2D/3D
differentiator already lives in `cfg.model.nnunet.spatial_dims`;
`pretrained: true` is the only distinguishing flag.

**Tutorial migration (40 YAML files):**

| Old key | New key |
|---|---|
| `inference.postprocessing.*` | `decoding.postprocessing.*` |
| `inference.saved_prediction_path` | `decoding.input_prediction_path` |
| `inference.decoding_path` | `decoding.output_path` |
| `model.architecture: nnunet_pretrained` | `model.architecture: nnunet` |
| `monitor.wandb.use_wandb` | (deleted; section dropped) |

Add a tutorial-load test that imports every YAML in `tutorials/` and
materializes the structured config without errors. Run in CI to prevent
silent regressions during the migration.

### D2 + B3. Real evaluation stage (PR 4)

Already covered in B3. Lands before PR 8 because evaluation extraction
removes coupling that the schema split would otherwise need to navigate.

### D3. Inference produces raw artifacts (PR 6 + PR 8)

Split across two PRs:

- *PR 6:* convert `inference/chunked.py::run_chunked_prediction_inference`
  to use `write_prediction_artifact` (currently constructs h5 dataset
  manually). Populate `checkpoint_path` (currently None at 423–447,
  617–642). Remove duplicated raw R/W loops between standard and chunked
  paths.
- *PR 8:* expand `inference/stage.py::run_prediction_inference` (currently
  a 30-line passthrough with no caller) into the canonical non-chunked
  entry: load model + run sliding window + write artifact with full
  metadata. Wire up so that `chunking.output_mode == "raw_prediction"`
  routes through both paths consistently.

Add fields to `PredictionArtifactMetadata`: `transpose: tuple[int,...] | None`,
`model_architecture: str | None`, `model_output_identity: str | None`,
`decode_after_inference: bool | None`.

`output.py::_restore_prediction_to_input_space` (170–221, the second
postprocessing pass) moves to `data` inverse pipeline. PR 8.

### D4. Postprocess module rename (PR 8)

`connectomics/decoding/postprocessing/postprocessing.py` → drop the
doubly-nested package and rename to `connectomics/decoding/postprocess.py`.

---

## Theme E — File splits (demoted to secondary, last in order)

Per feedback: file size is a smell check, not a design rule. Splits land
after ownership boundaries are fixed (PR 10).

After PRs 1–9 land, the high-leverage splits are:

| File | Current LOC | After prior PRs | Split target |
|---|---|---|---|
| `training/lightning/test_pipeline.py` | 2,005 | ≈1,100 (after B3) | `cached_pipeline.py` + `chunked_pipeline.py` + `prediction_crops.py` |
| `inference/tta.py` | 1,547 | 1,547 | `tta/combinations.py` + `tta/predictor.py` + `tta/patch_first_local.py` + `tta/distributed.py` + `tta/mask.py` |
| `inference/lazy.py` | 1,295 | ≈900 (after `LazyVolumeAccessor` moved to `data/io/`) | shared `_run_lazy_sliding` core; collapse `lazy_predict_region` + `lazy_predict_volume` |
| `data/augmentation/transforms.py` | 1,346 | 1,346 | `transforms_geometric.py` + `transforms_slicewise.py` + `transforms_intensity.py` + `transforms_advanced.py` + `_helpers.py` |
| `training/lightning/data_factory.py` | 1,168 | ≈1,000 (after `print()` → logger) | `data_factory/auto_download.py` + `random_data.py` + `sdt_precompute.py` + `datamodule_builders.py` |
| `training/lightning/callbacks.py` | 1,001 | 1,001 | per-callback files; fold `lightning/visualizer.py` (489) into `callbacks/visualization.py` |
| `config/pipeline/config_io.py` | 981 | ≈700 (after PR 7 moves cross-section validation) | `pipeline/loader.py` + `pipeline/validation.py` + `pipeline/paths.py` + `pipeline/naming.py` |
| `data/processing/transforms.py` | 979 | 979 | `transforms_basic.py` + `transforms_distance.py` + `transforms_morph.py` + `multitask.py` |
| `decoding/decoders/segmentation.py` | 815 | 815 | extract `_compute_edt` (~80 lines) → `data/processing/distance.py`; extract numba CC kernel (~140 lines) → `decoding/decoders/_affinity_cc.py`; remove `**kwargs` swallows in `decode_distance_watershed` and `decode_waterz` |

Skip splits if the file already has clean ownership and would just
generate more imports without locality benefit.

---

## Theme F — Runtime/CLI extraction (PR 7)

`scripts/main.py` (1,107 lines) and `training/lightning/utils.py` (771)
hold dispatch logic that doesn't belong in either entrypoint or Lightning
glue.

Build out `connectomics/runtime/`:

| Module | Source | Responsibility |
|---|---|---|
| `runtime/cli.py` | `lightning/utils.py:37–326` | `parse_args`, `setup_config` |
| `runtime/output_naming.py` | `lightning/utils.py:406–744` | naming helpers (created in PR 2 already) |
| `runtime/checkpoint_dispatch.py` | `scripts/main.py:135–200` + `_configure_checkpoint_output_paths`, `_setup_runtime_directories` | derive output base from checkpoint, extract step |
| `runtime/cache_resolver.py` | `scripts/main.py:203–384, 637–846` | `_resolve_cached_prediction_files`, `_is_valid_hdf5_prediction_file`, `_has_cached_predictions_in_output_dir`, `preflight_test_cache_hit`, `try_cache_only_test_execution`, `_handle_test_cache_hit` |
| `runtime/sharding.py` | `scripts/main.py:414–634` | `maybe_enable_independent_test_sharding`, `has_assigned_test_shard`, `shard_test_datamodule`, `maybe_limit_test_devices`, `resolve_test_rank_shard_from_env`, `resolve_test_image_paths`, `_estimate_tta_total_passes` |
| `runtime/tune_runner.py` | per B1 (PR 5) | `run_tuning`, `load_and_apply_best_params`, `_temporary_tuning_inference_overrides` |
| `runtime/torch_safe_globals.py` | `config/schema/root.py:93–131` + `scripts/main.py:60–69` + `lightning/trainer.py:53–60` | one canonical implementation |
| `runtime/preflight.py` | already exists | expand with cross-section validation moved out of `config_io.py` (B5) |

Target `scripts/main.py` ≈ 150 lines: parse → dispatch by mode → call
`runtime.{train,test,tune}`.

Three duplicate `torch.serialization.add_safe_globals` registrations
collapse to one in `runtime/torch_safe_globals.py`.

---

## Theme P — Performance (new, per feedback)

Added because original plan claimed "high-performance" without concrete
targets. Each item has a smoke benchmark in PR 0.

| Concern | Target | Where |
|---|---|---|
| Avoid round-trips like `tensor.cpu().float().numpy()` when downstream stays in torch or preserves dtype | grep audit; convert hot paths in `test_pipeline.py`, `tta.py`, `output.py`, `lazy.py` | PR 4 (evaluation), PR 6 (chunked decode) |
| Chunked inference write throughput | smoke benchmark on a representative volume; verify HDF5 chunk shape and compression on output artifact | PR 0 (baseline), PR 6 (no regression) |
| HDF5 chunk shape mismatch with read patterns | profile read patterns of saved artifacts (training cache reuse, decoding load, evaluation read); align chunk shapes with the most common read pattern | PR 6 |
| Duplicated raw prediction read/write loops between standard and chunked inference | identify shared code; extract a single I/O helper | PR 6 |
| TTA memory and data movement (patch-first-local, distributed reductions) | profile memory peak; reduce intermediate tensor allocations; verify CPU↔GPU transfer points | PR 4 or dedicated mini-PR |
| Don't force every streamed path through the artifact abstraction | benchmark the chunked path before/after PR 6; if `write_prediction_artifact` adds overhead vs direct h5 write, retain the direct path under a clearly named function | PR 6 |
| Lazy decoder registration | move `register_builtin_decoders()` out of `decoding/__init__.py` top-level; lazy-call from `get_decoder`/`apply_decode_pipeline` | PR 1 (with import-time test) |

Set the bar at "no regression". Anything that improves throughput is a
bonus; anything that regresses without justification gets reverted.

---

## Theme I — Docs refresh (PR 11)

`docs/source/*.rst` references v1 paths in 15 places:

| File | Stale references |
|---|---|
| `tutorials/synapse.rst:33,109` | `connectomics.data.dataset.VolumeDataset` |
| `tutorials/neuron.rst:17` | `connectomics.data.dataset.VolumeDataset` |
| `tutorials/mito.rst:165` | `connectomics.data.dataset.TileDataset` |
| `notes/migration.rst:534` | "✅ YACS configs (still work, but deprecated)" — false |
| `notes/dataloading.rst:12,36,121–135` | `from connectomics.data.augment import *`, `TileDataset` doc section |
| `notes/faq.rst:23–28` | `VolumeDataset` vs `TileDataset` Q&A — both classes gone |
| `modules/model.rst` and `modules/models.rst` | duplicates referencing old paths `connectomics.models.arch`, `connectomics.models.loss`, `connectomics.models.utils` |

Action: rewrite each `.rst` for v2 paths. Confirm Sphinx toctree
references before deleting `model.rst` or `models.rst` duplicate.

---

## PR sequence (revised, dependency-driven)

| # | Theme | Description | Depends on |
|---|---|---|---|
| 0 | Guardrails | import-boundary tests, strict-config tests, public-API snapshot tests, smoke benchmarks for chunked inference / TTA / lazy prediction | — |
| 1 | A1 | delete truly orphan files (`auto_tuning.py`, `blend.py`, `bbox.py` orphan helpers, `calculate_inference_grid`, `mito_betaseg.yaml~`); make `decoding/__init__.py` lazy-register builtins | 0 |
| 2 | F (partial) | extract naming helpers from `lightning/utils.py` to `runtime/output_naming.py`; update imports in `optuna_tuner.py`, `scripts/main.py`, `model.py`, `test_pipeline.py` | 0 |
| 3 | C1 + C2 | strict config: `_warn_unconsumed_keys` raises; delete `cfg.inference.test_time_augmentation.act`, `cfg.inference.output_act`; verify `TestConfig.output_path/cache_suffix` fallback then act | 0 |
| 4 | B3 + D2 | move evaluation logic out of training: build out `connectomics.evaluation` (`stage.py`, `metrics.py`, `nerl.py`, `report.py`); delete callback-injection from `evaluation/stage.py`; delete `_compute_test_metrics` | 2, 3 |
| 5 | B1 (step 2) + B4 | move tuning orchestrator to `runtime/tune_runner.py`; move `_log_decode_experiment` to `decoding/experiment_log.py`; pure tuner stays in `decoding/tuning/` | 2, 4 |
| 6 | B2 + D3 (step 1) + P (chunked) | move `run_chunked_affinity_cc_inference` to `decoding/streamed_chunked.py`; convert `run_chunked_prediction_inference` to use `write_prediction_artifact`; populate `checkpoint_path`; benchmark chunked write throughput vs baseline | 0, 4 |
| 7 | F (rest) + B5 | extract `runtime/cli.py`, `checkpoint_dispatch.py`, `cache_resolver.py`, `sharding.py`, `torch_safe_globals.py`; move cross-section channel validation from `config_io.py` to `runtime/preflight.py` | 2, 5 |
| 8 | D1 + H + D4 + D3 (step 2) | atomic schema split + tutorial migration: new `decoding.py` and `evaluation.py` schema modules; rename `nnunet_pretrained` → `nnunet`; rename `decoding/postprocessing/postprocessing.py` → `decoding/postprocess.py`; expand `inference/stage.py::run_prediction_inference`; migrate 40 tutorials in same commit | 4, 5, 6, 7 |
| 9 | A2 + G | public API removals: `SaveVolumed`, `TileLoaderd`, `create_combined_loss`, `create_loss_from_config`, `CombinedLoss`, `to_plain`, `merge_configs`, `update_from_cli`, `to_dict`, `from_dict`, `print_config`, `get_config_hash`, `create_experiment_name`, `expand_file_paths` re-export, dead schema exports; trim `__init__.py` files; A3 items per maintainer decisions | 8 |
| 10 | E | file splits, in priority order: `test_pipeline.py`, `tta.py`, `lazy.py`, `transforms.py` × 2, `data_factory.py`, `callbacks.py`, `config_io.py`, `decoders/segmentation.py` | 8 |
| 11 | I | docs refresh; remove duplicate `model.rst`/`models.rst` after Sphinx toctree check | 8 |

Note that PRs 1–7 are no longer "mergeable in any order". Real
dependencies:

- PR 2 must precede PR 5 (tuner needs naming helpers in their new home).
- PR 4 must precede PR 5 (the metric callable consumed by the pure tuner
  lives in evaluation).
- PR 6 must precede PR 8 (raw artifact writing is a contract the schema
  split assumes).
- PR 8 is the only PR that breaks tutorials; everything before it must
  keep YAML loading green.
- PR 9 must follow PR 8 (public API trim makes sense after schema split
  produces the final shape).
- PR 10 follows PR 8 (file splits are easier to review after structural
  moves land).

## PR 0 verification guardrails (concrete tests to add)

Per feedback: add tests *before* moving code. PR 0 contents:

```python
# tests/unit/test_v2_boundaries.py — extend
def test_decoding_does_not_import_training():
    """connectomics.decoding must not import connectomics.training (any path)."""
    # walk decoding/* modules; importlib.util.find_spec each; check sys.modules
    # after import for any 'connectomics.training.*' entry.

def test_inference_does_not_import_decoding():
    ...

def test_config_does_not_import_monai():
    ...

def test_config_load_raises_on_unknown_top_level_key():
    cfg_yaml = "experiment_name: x\nunknown_section: {}"
    with pytest.raises(ValueError, match="Unknown key"):
        load_config_from_string(cfg_yaml)

# tests/unit/test_public_api_snapshot.py — new
def test_connectomics_config_public_api():
    """Snapshot of intentional public API. Update when v2 contract changes."""
    expected = {"Config", "load_config", "save_config", "validate_config",
                "resolve_default_profiles", "as_plain_dict", "cfg_get",
                "DefaultConfig", "TrainConfig", "TestConfig", "TuneConfig",
                "SystemConfig", "ModelConfig", "DataConfig",
                "OptimizationConfig", "MonitorConfig", "InferenceConfig",
                "DecodingConfig", "EvaluationConfig"}
    assert set(connectomics.config.__all__) == expected

def test_connectomics_inference_public_api():
    ...

# tests/benchmarks/test_chunked_inference_throughput.py — new
def test_chunked_inference_smoke():
    """Baseline throughput on a small representative volume."""
    # write a small h5, run run_chunked_prediction_inference with a fake
    # forward fn, time the write, assert duration < threshold (calibrated).
```

These are smoke-level, not exhaustive. The point is to catch regressions
introduced by later PRs, not to produce reproducible benchmark numbers.

## Estimated impact (revised, honest)

- ≈ 1,200 LOC removed in PR 1 (truly orphan files)
- ≈ 800 LOC removed in PR 4 (eval extraction net of new `evaluation/` files)
- ≈ 400 LOC removed in PR 9 (public API removals net of test additions)
- ≈ 1,500 LOC moved (training → evaluation, scripts → runtime, decoding → runtime)
- ≈ 0.5 KLOC removed in PR 3 (ghost reads + verified-dead schema fields)
- After PR 10: 0 files >700 lines; most files <400 lines

The "≈ 5 KLOC removed in dead-code deletion" estimate from the original
plan was inflated. Real total deletion across all PRs is closer to
2.5 KLOC, with another 1.5 KLOC of moves.

## Tradeoffs

Same set as the original plan, with adjusted weights:

- **Tutorial breakage (PR 8 only):** atomic schema split + key rename hits
  40 tutorials. v2 contract accepts this; PR 0's tutorial-load test in CI
  catches mistakes early.
- **Tune mode call graph (PR 5):** `run_tuning` moves from decoding to
  runtime. `scripts/main.py` import path changes.
- **Test pipeline coupling (PR 4):** removing 25 `module._*` accesses
  forces TestContext to grow into a complete dataclass; some tests
  inspecting specific Lightning-module methods need updates.
- **Schema reorganization (PR 8):** torch safe-globals registration
  ordering is consolidated in PR 7's `runtime/torch_safe_globals.py`,
  which already uses reflective discovery; safe.
- **Performance regressions:** PR 6's artifact-abstraction refactor is
  the main risk. PR 0's smoke benchmark gates it.

## What stays out of v3

These came up during audit but are not in scope:

- Bulk getattr cleanup beyond the per-domain PRs above. The remaining
  cases are tolerable until they're touched for another reason.
- Augmentation alias unification (`RandSliceShiftZd`/`RandSliceDropZd` as
  subclasses vs flag). Cosmetic; defer.
- Splitting `data/augmentation/build.py` and `data/processing/build.py`
  beyond the getattr cleanup. Files are dense but cohesive.
- `connectomics.config.pipeline.__init__.py` facade — leave as-is unless
  a consumer surfaces. Per v2 architecture rule 2 it should go, but it's
  inert.
- Optional-dependency guards (`_MEDNEXT_AVAILABLE`, `_NNUNET_AVAILABLE`).
  Genuinely optional; keep.

These are tracked but not blocking v3. They become candidates for v4 if
the codebase evolves enough to surface real friction.
