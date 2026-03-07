# Config System: Current State and Remaining Work

**Last updated:** 2026-03-07 (architect review + implementation pass + Option A grouping)

---

## 1. Architecture Overview

The configuration system is a **stage-aware, profile-driven** architecture built on Hydra/OmegaConf dataclasses. Total: **~4,500 lines** across 21 Python files organized in 3 subpackages.

### 1.1 Directory Structure

```
config/
├── __init__.py              # Package public API (re-exports from subpackages)
├── pipeline/                # Core load/merge/resolve
│   ├── __init__.py
│   ├── config_io.py         # Load, save, merge, validate, path resolution
│   ├── profile_engine.py    # YAML profile applier classes + declarative engine
│   ├── stage_resolver.py    # default/train/test/tune → runtime merge
│   └── dict_utils.py        # as_plain_dict, cfg_get utilities
├── hardware/                # Optional GPU/SLURM planning
│   ├── __init__.py
│   ├── auto_config.py       # GPU-aware auto-planning (nnUNet-inspired)
│   ├── gpu_utils.py         # GPU info, memory estimation, batch size suggestion
│   └── slurm_utils.py       # SLURM cluster utilities
└── schema/                  # Pure dataclass definitions (no logic)
    ├── __init__.py           # Re-export hub for all schema dataclasses
    ├── root.py              # Config root + torch safe globals
    ├── data.py              # 27 data/augmentation config dataclasses
    ├── inference.py         # 12 inference config dataclasses
    ├── stages.py            # Stage configs (Default, Train, Test, Tune)
    ├── monitor.py           # Checkpoint, logging, W&B
    ├── optimization.py      # Optimizer, scheduler, EMA
    ├── model.py             # Model, loss, arch config
    ├── model_monai.py       # MONAI UNet/Transformer config
    ├── model_mednext.py     # MedNeXt config
    ├── model_rsunet.py      # RSUNet config
    ├── model_nnunet.py      # nnUNet config
    └── system.py            # System config
```

### 1.2 Data Flow

```
YAML file
  → _load_config_with_bases()        [_base_ inheritance, cycle detection]
  → _YAML_PROFILE_ENGINE.apply()     [profile resolution on raw DictConfig]
  → _warn_unconsumed_keys()          [typo detection]
  → OmegaConf.merge(defaults, yaml)  [schema validation]
  → OmegaConf.to_object()            [convert to dataclass tree]
  → resolve_default_profiles(mode)   [default/stage merge into runtime sections]
  → validate_config()                [cross-section coherence checks]
  → resolve_data_paths()             [glob expansion, base path joining]
```

### 1.3 Stage Precedence Model

```
schema defaults  <  default.*  <  train/test/tune.*
(lowest)                              (highest)
```

`default` is the shared-across-modes layer. `train`/`test`/`tune` carry mode-specific overrides. All stage sections use typed dataclasses.

### 1.4 Profile System

Profiles are named YAML snippets in `tutorials/bases/*.yaml`, resolved pre-conversion by the profile engine. Supported profile families:

| Family | Selector path | Target |
|--------|--------------|--------|
| `pipeline_profiles` | `default.pipeline_profile` | root (multi-section) |
| `arch_profiles` | `{stage}.model.arch.profile` | `{stage}.model` |
| `system_profiles` | `{stage}.system.profile` | `{stage}.system` |
| `augmentation_profiles` | `{stage}.data.augmentation.profile` | `{stage}.data.augmentation` |
| `dataloader_profiles` | `{stage}.data.dataloader.profile` | `{stage}.data.dataloader` |
| `optimizer_profiles` | `{stage}.optimization.profile` | `{stage}.optimization` |
| `loss_profiles` | `{stage}.model.loss.profile` | `{stage}.model.loss.losses` |
| `label_profiles` | `{stage}.data.label_transform.profile` | `{stage}.data.label_transform` |
| `decoding_profiles` | `{stage}.inference.decoding_profile` | `{stage}.inference.decoding` |
| `activation_profiles` | `{stage}.inference.test_time_augmentation.activation_profile` | `{stage}.inference.test_time_augmentation.channel_activations` |

Selectors are only accepted at canonical paths; non-canonical paths raise `ValueError`.

---

## 2. Completed Work

All recommendations from the original review plus all cleanup items have been implemented:

| Item | Status |
|------|--------|
| Declarative profile tables (replaced ~50 boilerplate appliers) | Done |
| Split `hydra_utils.py` → `config_io.py`, `profile_engine.py`, `stage_resolver.py` | Done |
| Unconsumed-key warnings | Done |
| Cross-section validation (out_channels vs loss/label/decoding/activation, input_size vs patch_size, deep_supervision vs arch) | Done |
| `--debug-config` flag | Done |
| `tutorials/minimal.yaml` | Done |
| `shared` → `default` (old key hard-rejected) | Done |
| Canonical-only selector paths enforced | Done |
| Single-phase profile resolution enforced | Done |
| Typed stage configs (no more `Dict[str, Any]`) | Done |
| `MergeContext` dataclass (replaced `setattr` hack) | Done |
| Auto-enable removed (profiles set `enabled: true` explicitly) | Done |
| **Delete facade files** (`hydra_config.py`, `hydra_utils.py`) | Done |
| **Clean `MonaiConfig`** (removed duplicate `act`, `upsample` fields) | Done |
| **Remove optimization aliases** (kept `n_steps_per_epoch`, removed `iter_num_per_epoch`, `iter_num`) | Done |
| **Consolidate `SchedulerConfig`** (moved scheduler-specific fields to `params` dict) | Done |
| **Remove `WandbConfig` prefix** (`wandb_project` → `project`, etc.) | Done |
| **Delete commented-out code** in `DataInputConfig` | Done |
| **Fix unnecessary lambda** in `DataConfig.augmentation` | Done |
| **Table-drive `_collect_stage_overrides()`** (replaced 3 repetitive branches with `_MODE_SECTIONS` lookup) | Done |
| **Update all imports** (`stage_resolver.py`, `config_io.py` import from `.schema` not `.hydra_config`) | Done |
| **Redesign augmentation profiles** (5 systematic profiles replacing 4 ad-hoc + 5 standalone presets) | Done |
| **Option A grouping** (reorganized config/ into `pipeline/`, `hardware/`, `schema/` subpackages) | Done |

### Verification

- No references to `hydra_config` or `hydra_utils` remain in the codebase
- No references to old `wandb_project`/`wandb_entity`/`wandb_tags`/`wandb_name` fields remain
- No references to old `iter_num_per_epoch` or config-level `iter_num` remain (dataset-internal `iter_num` property is unrelated)
- No references to old `act`/`upsample` MonaiConfig fields remain in YAML or Python
- Scheduler builder (`training/optim/build.py`) already uses `_scheduler_specific_param()` which reads from `params` dict first — fully compatible with new schema
- WandB config has no downstream consumers in training code (field names only exist in schema)

---

## 3. Current Assessment

### 3.1 Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Conceptual design** | 9/10 | Clean six-section + three-stage model; typed throughout |
| **Profile system** | 9/10 | Declarative tables, canonical-only paths, coherence validation |
| **Type safety** | 9/10 | All stage configs typed; schema validation on merge |
| **Implementation** | 9/10 | Dead code removed; all arch specs registered; no private API usage |
| **User experience** | 8/10 | Profiles, minimal.yaml, debug-config, clear errors |
| **Maintainability** | 9/10 | Adding a profile = 1 table entry; adding a schema = 1 dataclass |

**Overall: 9.0/10 — Production ready. Clean, well-organized, no dead code.**

### 3.2 Augmentation Profile Library

The augmentation profiles (`tutorials/bases/augmentation_profiles.yaml`) provide a systematic library of 5 profiles organized by use case:

| Profile | Augmentations | Use case |
|---------|--------------|----------|
| `aug_light` | flip, rotate, mild intensity | Quick experiments, clean data, fine-tuning |
| `aug_standard` | light + misalignment, missing_section | Default for most 3D EM tasks (RECOMMENDED) |
| `aug_strong` | standard + affine, elastic, motion_blur, missing_parts, cut_noise, strong intensity | Small datasets, overfitting prevention |
| `aug_instance` | standard + copy_paste, mixup | Instance segmentation (neuron, mito, synapse) |
| `aug_superres` | light + cut_blur, motion_blur | Super-resolution / multi-scale learning |

Design principles:
- **Self-contained**: each profile is complete — pick one, optionally override fields
- **Strength-ordered**: light → standard → strong follows a clear intensity progression
- **Task-specific**: instance and superres address specific segmentation needs
- **Derived from real usage**: parameters match patterns from 14 tutorial configs
- **No standalone preset files**: removed `tutorials/presets/` (was 5 standalone experiment configs that duplicated model/optimizer/data alongside augmentation)

### 3.3 What Works Well

- **No re-export facades**: `__init__.py` imports directly from real modules. `schema/__init__.py` is the single source of truth for schema exports.
- **No field aliases**: Each concept has exactly one field name. `n_steps_per_epoch` is the only name for steps-per-epoch. `activation` is the only name for activation function. `upsample_mode` is the only name for upsample mode.
- **Scheduler extensibility**: `SchedulerConfig` has `name` + `params: Dict[str, Any]` for scheduler-specific args, plus shared fields (`warmup_epochs`, `min_lr`, `interval`, `frequency`) that apply to all schedulers. `build_lr_scheduler()` uses `_scheduler_specific_param()` to read from `params` first.
- **Table-driven stage overrides**: `_MODE_SECTIONS` dict maps each mode to its allowed sections. `_collect_stage_overrides()` is a single loop with a conditional for test/tune data hoisting.
- **Strict validation pipeline**: Non-canonical selectors, unresolved post-conversion selectors, unconsumed keys, removed stage keys, and cross-section coherence are all validated with clear error messages.
- **`dict_utils.py` properly integrated**: `as_plain_dict` used by `stage_resolver.py`; `cfg_get` used by `training/lightning/model.py` and `training/loss/plan.py`. Small (41 lines), well-scoped.

---

## 4. Resolved Items (2026-03-07 architect review)

All items from the architect review have been implemented:

| # | Issue | Resolution |
|---|-------|-----------|
| B1 | `_has_explicit_path` was O(n²) | Pre-computed prefix set for O(1) lookup |
| B2 | `SystemConfig` None defaults fragility | `SystemConfig` now has sensible non-None defaults (`num_gpus=1`, `num_workers=8`, `seed=42`); `Config.system` uses plain `SystemConfig()` factory |
| M1 | Private OmegaConf `_get_node` API | Replaced with `_safe_get_child()` using `OmegaConf.is_interpolation()` + public API access with exception handling |
| M2 | `MonaiBasicUNet3DSpec` throwaway instances | Deleted `model_arch.py` entirely (see A1) |
| M3 | Missing arch specs for UNETR variants | Deleted `model_arch.py` entirely (see A1) |
| A1 | Arch profiles used custom translation layer (`model_arch.py`, `ArchProfileApplier`) while all other profiles used simple `ValueProfileApplier` | Deleted `model_arch.py` (300 lines), `ArchProfileApplier` class, arch-specific build functions. Updated `arch_profiles.yaml` to use ModelConfig structure directly (`arch.type` instead of `type`, `mednext.size` instead of `variant`). Arch profiles now use the same `ValueProfileApplier` as every other profile family. |
| D1 | `MappingProfileApplier` unused | Deleted (65 lines) |
| D2 | `PathMoveApplier` unused | Deleted (23 lines) |
| D3 | `schema/helpers.py` unused | Deleted file + removed from `schema/__init__.py` exports |
| D4 | `slurm_utils.py` no production use | Kept (useful for SLURM users); removed `__main__` block |
| D5 | `__main__` blocks in 3 files | Removed from `auto_config.py`, `gpu_utils.py`, `slurm_utils.py` (~40 lines) |
| S1 | Emojis in print output | Replaced with plain text in `auto_config.py` |
| S2 | Mixed type annotation syntax | Noted; `from __future__ import annotations` makes both syntaxes work identically at runtime |
| S3 | Mixed `.format()` / f-strings | Converted to f-strings in `config_io.py` |
| S4 | Misleading "Binary search" comment | Fixed to "Linear scan" in `gpu_utils.py` |

---

## 5. Remaining Observations (no action required)

These are architectural notes for future reference, not cleanup items.

### 5.1 `iter_num` in dataset classes vs `n_steps_per_epoch` in config

The dataset classes (`dataset_base.py`, `dataset_volume.py`, etc.) use `self.iter_num` as an internal property to control dataset length. This is distinct from `cfg.optimization.n_steps_per_epoch` which is the config-level setting. The mapping happens in `data_factory.py` (line 513): `iter_num_cfg = cfg.optimization.n_steps_per_epoch`. No aliasing issue.

### 5.2 Stage section allowlists

In `_collect_stage_overrides()`, the sections extracted per mode are:
- **train**: model, data, optimization, monitor (no inference)
- **test**: model, data, inference (no optimization, no monitor)
- **tune**: model, data, inference (no optimization, no monitor)

If a user writes `test.optimization.lr: 0.001`, OmegaConf will accept it (since `TestConfig` doesn't have an `optimization` field, it goes through typed merge and will error). This is correct behavior — the schema prevents invalid stage sections.

Note: `DefaultConfig` includes all 6 sections (system, model, data, optimization, monitor, inference). `TrainConfig` has 5 (no inference). `TestConfig` has 4 (system, model, data, inference). `TuneConfig` has 4 (system, model, data, inference). The `_MODE_SECTIONS` table matches what each stage config actually supports (minus system, which is handled separately by `_merge_system()`).

### 5.3 `_extract_mode_data_overrides` batch_size hoisting

Test/tune data sections support `batch_size` at the top level (e.g., `test.data.batch_size`), which gets hoisted into `dataloader.batch_size`. Train data does not have this hoisting. This asymmetry is intentional — test/tune data is simpler and benefits from a flat shorthand.

### 5.4 `SchedulerConfig` shared fields

`SchedulerConfig` still has some fields (`monitor`, `mode`, `factor`, `patience`, `threshold`, `cooldown`, `eps`) that are only used by `ReduceLROnPlateau`. These could technically move to `params` too, but they're also generic enough to be useful as documentation of common scheduler options. Current state is a reasonable compromise.

### 5.5 `schema/__init__.py` size

At 158 lines, this file is a large `__all__` list. This is acceptable — it's the canonical export point for ~50 schema dataclasses. The alternative (wildcard re-export) would lose explicit control over the public API.

### 5.6 `_extract_max_referenced_channel` arithmetic

The formula `start - end` (when `end < 0`) in `config_io.py:381-383` looks surprising but is mathematically correct: it gives the 0-based max channel index, and the caller adds 1 to get required channel count. A clarifying comment would help future readers.

---

## 6. Line Count History

| Milestone | Total lines | Files |
|-----------|------------|-------|
| Pre-refactor (hydra_utils.py monolith) | ~1,950 | 1 file |
| Post-split (config_io + profile_engine + stage_resolver) | ~5,217 | 22 files |
| Post-cleanup (facades deleted, aliases removed, table-driven) | **4,940** | **20 files** |
| Post-dict_utils addition | ~5,000 | 21 files |
| Post-architect review cleanup | ~4,800 | 20 files |
| Post-arch unification (model_arch.py deleted) | **~4,500** | **19 files** |
| Post-Option A grouping (pipeline/ + hardware/ subdirs) | **~4,510** | **21 files** |

Net reduction from post-split: **~717 lines** removed. Key changes: deleted `model_arch.py` (arch profiles now use `ValueProfileApplier` like all other profiles), deleted `schema/helpers.py`, removed unused appliers, removed `__main__` blocks, fixed bugs. Option A grouping added 2 `__init__.py` files (~10 lines) but improved organization by separating pipeline logic from hardware utilities.
