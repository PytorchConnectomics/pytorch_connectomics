# Refactor: All `utils` Boundaries — Holistic Plan

**Status:** Planning
**Scope:** Every `utils.py`, `utils/` directory, and utility-like module across the entire `connectomics/` package.

## Current Inventory

### 1. `connectomics/utils/` (top-level shared utils)

| File | Lines | Consumers | Packages |
|------|-------|-----------|----------|
| `channel_slices.py` | 62 | `inference/tta.py`, `training/loss/orchestrator.py`, `data/process/affinity.py` | inference, training, data |
| `label_overlap.py` | 24 | `metrics/segmentation_numpy.py`, `decoding/postprocess/postprocess.py` | metrics, decoding |
| `errors.py` | 130 | `scripts/main.py` | scripts only |
| `__init__.py` | 16 | re-exports channel_slices + label_overlap | — |

**Verdict:** `channel_slices.py` and `label_overlap.py` are genuinely cross-cutting (3 and 2 packages respectively). `errors.py` has a single consumer.

### 2. `connectomics/decoding/utils.py` (128 lines)

| Function | Consumers |
|----------|-----------|
| `cast2dtype` | `decoders/segmentation.py`, `decoders/synapse.py`, `decoders/abiss.py` |
| `remove_small_instances` | `decoders/segmentation.py`, `tuning/optuna_tuner.py` |
| `remove_large_instances` | `decoders/segmentation.py` |
| `merge_small_objects` | internal (called by `remove_small_instances`) |

**Verdict:** All consumers are within `decoding/`. This is a well-scoped package-internal utility. Keep as-is.

### 3. `connectomics/training/lightning/utils.py` (337 lines)

| Function | Consumers |
|----------|-----------|
| `parse_args` | `scripts/main.py` |
| `setup_config` | `scripts/main.py` |
| `extract_best_score_from_checkpoint` | `training/lightning/runtime.py` |
| `setup_seed_everything` | `scripts/main.py` |

**Verdict:** This is really CLI/entrypoint plumbing for `scripts/main.py`. Only `extract_best_score_from_checkpoint` is used internally (by `runtime.py`). The file is correctly scoped within `training/lightning/`.

### 4. `connectomics/training/lightning/path_utils.py` (31 lines)

| Function | Consumers |
|----------|-----------|
| `expand_file_paths` | `training/lightning/utils.py`, `training/lightning/data_factory.py` |

**Verdict:** Both consumers are within `training/lightning/`. Well-scoped. Could be inlined into `data_factory.py` (its primary consumer), but the current split is fine.

### 5. `connectomics/data/io/utils.py` (28 lines)

| Function | Consumers |
|----------|-----------|
| `rgb_to_seg` | `data/io/io.py`, `data/io/tiles.py` |

**Verdict:** Both consumers are within `data/io/`. Well-scoped package-internal utility.

### 6. `connectomics/data/utils/` (re-export shim)

| File | Lines | Canonical Location | Active Consumers |
|------|-------|-------------------|-----------------|
| `__init__.py` | 23 | re-exports from `dataset/split.py` + `dataset/sampling.py` | — |
| (no own files) | — | all code lives in `dataset/split.py` and `dataset/sampling.py` | — |

Active consumers of the underlying files:
- `dataset/split.py` → `data/augment/build.py` (ApplyVolumetricSplitd), `training/lightning/data_factory.py` (split_volume_train_val)
- `dataset/sampling.py` → `training/lightning/data_factory.py` (compute_total_samples)
- `data/utils/__init__.py` → `tests/unit/train_val_split.py` (the only consumer of the re-export shim)

**Verdict:** `data/utils/` is a backward-compat re-export shim. Only one test imports through it; all production code imports directly from `dataset/`. **Delete the shim.**

### 7. `connectomics/config/pipeline/dict_utils.py` (41 lines)

| Function | Consumers |
|----------|-----------|
| `as_plain_dict` | `config/pipeline/stage_resolver.py` |
| `cfg_get` | `training/loss/plan.py`, `training/lightning/model.py` |

**Verdict:** `cfg_get` crosses the config→training boundary (2 training consumers). `as_plain_dict` is config-internal. The file is correctly placed in `config/pipeline/` since `cfg_get` is a config-accessor pattern. Cross-package usage is justified.

### 8. `connectomics/training/lightning/visualizer.py` (491 lines)

| Export | Consumers |
|--------|-----------|
| `Visualizer` | `training/lightning/callbacks.py` |
| `get_visualization_mask` | `training/lightning/callbacks.py` |

**Verdict:** Single consumer within same package. Well-scoped.

### 9. `connectomics/data/download.py` (181 lines)

| Export | Consumers |
|--------|-----------|
| `download_dataset` | `scripts/download_data.py` |
| `list_datasets` | `scripts/download_data.py` |

**Verdict:** Single consumer (script). Well-scoped in `data/` since it's about data acquisition.

### 10. `connectomics/training/debugging.py` (441 lines)

| Export | Consumers |
|--------|-----------|
| `DebugManager` | `training/lightning/model.py` |
| `NaNDetectionHookManager` | internal (via DebugManager) |

**Verdict:** Single consumer within same package. Well-scoped.

### 11. `connectomics/config/hardware/gpu_utils.py` and `slurm_utils.py`

Both consumed by `config/hardware/auto_config.py` and tests. Package-internal. Well-scoped.

### 12. `connectomics/data/process/misc.py`

| Function | Consumers |
|----------|-----------|
| `get_seg_type` | `decoding/utils.py` (cross-package) |
| `bbox_ND`, `crop_ND` | `decoding/utils.py`, `decoding/postprocess/postprocess.py` (cross-package) |

**Verdict:** These are cross-package utilities consumed by `decoding/`. They live in `data/process/` because that's the data-processing domain, and decoding depends on data processing. The dependency direction (decoding → data) is correct.

---

## Holistic Assessment

### What's Already Right

Most utility boundaries are **well-defined**:
- **Top-level `utils/`** contains only genuinely cross-cutting code (3+ package consumers)
- **Package-internal utils** (`decoding/utils.py`, `data/io/utils.py`, `training/lightning/utils.py`) serve only their own package
- **No circular dependencies** — all utility imports flow downward (scripts → training → data/config, decoding → data)
- **No utility dumping ground** — the old anti-pattern of stuffing everything into `utils/` has been resolved

### What Needs Fixing

Only **one issue**: the `data/utils/` re-export shim.

---

## Changes

### 1. Delete `connectomics/data/utils/` (re-export shim)

**Rationale:** This directory contains zero original code. It re-exports from `data/dataset/split.py` and `data/dataset/sampling.py`. All production code already imports directly from `data/dataset/`. Only one test file uses the shim path.

| Action | File |
|--------|------|
| Delete | `connectomics/data/utils/__init__.py` |
| Delete | `connectomics/data/utils/` (directory) |
| Update | `tests/unit/train_val_split.py` — change `from connectomics.data.utils.split` → `from connectomics.data.dataset.split` |
| Update | `connectomics/data/__init__.py` — remove any re-export of `data.utils` if present |

**Impact:** 23 lines removed, 1 test import updated. Zero production code changes.

### 2. Move `errors.py` to `scripts/` (optional, low priority)

**Rationale:** `errors.py` has exactly one consumer: `scripts/main.py`. It's a pre-flight validation module for the CLI entrypoint. Moving it to `scripts/preflight.py` would follow the dissolve-to-consumer principle.

However, this is **low priority** because:
- `errors.py` is small (130 lines)
- It's clearly documented as "for scripts"
- Moving it to `scripts/` might feel wrong since `scripts/` isn't a proper Python package

**Recommendation:** Keep as-is unless `scripts/` becomes a proper package. Document the single-consumer relationship.

---

## Non-Changes (Explicitly Kept)

| File | Why Keep |
|------|----------|
| `utils/channel_slices.py` | 3 packages depend on it (inference, training, data) |
| `utils/label_overlap.py` | 2 packages depend on it (metrics, decoding) |
| `utils/errors.py` | Single consumer, but `scripts/` isn't a package |
| `decoding/utils.py` | All consumers within decoding/ |
| `training/lightning/utils.py` | CLI utilities, correctly scoped |
| `training/lightning/path_utils.py` | Both consumers within training/lightning/ |
| `data/io/utils.py` | Both consumers within data/io/ |
| `config/pipeline/dict_utils.py` | Config-accessor pattern, cross-package use justified |
| `training/lightning/visualizer.py` | Single consumer, correctly scoped |
| `training/debugging.py` | Single consumer, correctly scoped |
| `data/download.py` | Script utility, correctly in data/ |

---

## Dependency Flow

```
scripts/main.py
  └─ training/lightning/utils.py (CLI plumbing)
       └─ config/ (load, validate, resolve)
       └─ training/lightning/path_utils.py

training/lightning/model.py
  └─ config/pipeline/dict_utils.py (cfg_get)
  └─ training/debugging.py (NaN hooks)

training/loss/orchestrator.py
  └─ utils/channel_slices.py (cross-cutting)

inference/tta.py
  └─ utils/channel_slices.py (cross-cutting)

data/process/affinity.py
  └─ utils/channel_slices.py (cross-cutting)

metrics/segmentation_numpy.py
  └─ utils/label_overlap.py (cross-cutting)

decoding/postprocess/postprocess.py
  └─ utils/label_overlap.py (cross-cutting)
  └─ data/process/ (bbox_ND, crop_ND)

decoding/utils.py
  └─ data/process/ (get_seg_type, bbox_ND, crop_ND)

training/lightning/callbacks.py
  └─ training/lightning/visualizer.py
```

All arrows point downward or sideways within the same package. No cycles.

---

## Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Boundary clarity** | 9/10 | Every utils file has a clear scope; only `data/utils/` shim is wrong |
| **Cross-cutting justification** | 10/10 | Only 2 files in top-level `utils/`, both used by 2-3 packages |
| **No dumping ground** | 10/10 | Package-internal utils stay in their package |
| **Dependency direction** | 10/10 | No cycles, clean downward flow |
| **Remaining work** | 1 item | Delete `data/utils/` re-export shim |

**Overall: 9.5/10 — Utility boundaries are already well-defined. One shim to delete.**
