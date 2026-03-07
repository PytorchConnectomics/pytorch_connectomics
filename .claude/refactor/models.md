# Refactor Plan: connectomics/models/

## Status: ✅ Complete

All identified issues have been addressed. 191 unit tests pass.

**Files analyzed (14 total):**
- `__init__.py` - Package exports
- `build.py` - Model factory
- `arch/__init__.py` - Architecture module init
- `arch/registry.py` - Registry system
- `arch/base.py` - Base model ABC
- `arch/monai_models.py` - MONAI wrappers
- `arch/mednext_models.py` - MedNeXt wrappers
- `arch/nnunet_models.py` - nnUNet wrappers
- `arch/rsunet.py` - RSUNet implementation
- `loss/__init__.py` - Loss exports
- `loss/build.py` - Loss factory
- `loss/losses.py` - Custom losses
- `loss/metadata.py` - Loss metadata
- `loss/regularization.py` - Regularization losses

---

## Issues Found and Resolved

### 1. ✅ [CORRECTNESS BUG] RSUNet deep supervision keys incompatible with LossOrchestrator

- **Impact**: RSUNet deep supervision outputs were silently dropped/misaligned
- **Root cause**: RSUNet produced 0-based keys (`ds_0..ds_3`) but LossOrchestrator consumes 1-based (`ds_1..ds_4`), matching MedNeXt and the base class contract
- **Fix**: Changed `rsunet.py` line 443 from `f"ds_{i}"` to `f"ds_{i + 1}"` — now produces `ds_1..ds_4`

### 2. ✅ [CORRECTNESS] RSUNet mutable default argument

- **Impact**: Latent Python footgun — mutable list default `width: List[int] = [16, 32, 64, 128, 256]`
- **Fix**: Changed to `width: Optional[List[int]] = None` with `if width is None: width = [...]` inside the function body

### 3. ✅ [CORRECTNESS] `print()` used instead of `logging` throughout models/

- **Impact**: Print statements not capturable by logging frameworks, not suppressible, pollute stdout
- **Fix**: Replaced all `print()` with `logging.getLogger(__name__)` in:
  - `build.py` — model info logging
  - `arch/__init__.py` — `print_available_architectures()` now uses `logger.info()`
  - `arch/nnunet_models.py` — all loading progress, plus `warnings.warn()` for dimension mismatches

### 4. ✅ [DEAD CODE] Convenience loss factories removed

- **Impact**: `create_binary_segmentation_loss`, `create_multiclass_segmentation_loss`, `create_focal_loss` were exported but never called anywhere
- **Fix**: Removed from `loss/build.py`, `loss/__init__.py`, and `models/__init__.py`

### 5. ✅ [DEAD CODE] `ConnectomicsModel.summary()` removed

- **Impact**: Zero callers across the codebase, untested
- **Fix**: Removed from `arch/base.py`. `__repr__` kept (follows Python conventions). Updated test that referenced it.

### 6. ✅ [DEAD CODE] `_RSUNET_AVAILABLE` flag removed

- **Impact**: Hardcoded `True`, meaningless (RSUNet is pure PyTorch, always available)
- **Fix**: Removed from `arch/__init__.py`. `get_available_architectures()` now filters rsunet archs unconditionally.

### 7. ⏭️ [KEPT] `nnunet_2d_pretrained` and `nnunet_3d_pretrained` wrappers

- **Original proposal**: Remove as trivial delegations
- **Decision**: Kept because `nnunet_2d_pretrained` is referenced in `tutorials/misc/mito_2dsem_seg.yaml`. Removing would break existing configs.
- **Improvement**: Converted `print()` warnings to `warnings.warn()` and all other `print()` to `logging`.

### 8. ✅ [EFFICIENCY] Duplicated spatial_dims inference consolidated

- **Impact**: Same 4-line pattern duplicated in `build_basic_unet` and `build_monai_unet`
- **Fix**: Extracted `_infer_spatial_dims(cfg) -> int` helper in `monai_models.py`

### 9. ✅ [EFFICIENCY] Duplicated GroupNorm handling consolidated

- **Impact**: Same 5-line pattern duplicated in `build_basic_unet` and `build_monai_unet`
- **Fix**: Extracted `_resolve_norm(cfg)` helper in `monai_models.py`

### 10. ⏭️ [KEPT] `CombinedLoss.forward()` does not log individual loss components

- **Decision**: Acceptable as-is. The orchestrator is the primary path and has its own loss tracking.

### 11. ✅ [DESIGN] Dual metadata registry eliminated

- **Impact**: `_CLASSNAME_TO_METADATA_NAME` was a 20-entry dict that had to be kept in sync with `_LOSS_METADATA_BY_NAME`
- **Fix**: Removed `_CLASSNAME_TO_METADATA_NAME`. Added `CrossEntropyLossWrapper` directly to `_LOSS_METADATA_BY_NAME`. `get_loss_metadata_for_module()` now does a single dict lookup.

### 12. ✅ [DESIGN] `LossCallKind` and `TargetKind` now use `Literal` types

- **Impact**: Were bare `str` aliases providing zero type safety
- **Fix**: Changed to `Literal["pred_target", "pred_only", "pred_pred", "unsupported"]` and `Literal["dense", "class_index", "none"]`

### 13. ✅ [DESIGN] Redundant try/except in `build_model` removed

- **Impact**: `build.py` caught `ValueError` from `get_architecture_builder` just to print and re-raise, duplicating the error message
- **Fix**: Removed try/except. The error from `get_architecture_builder` is already informative.

### 14. ⏭️ [KEPT] `GANLoss.forward()` incompatible signature

- **Decision**: Acceptable as-is. Already marked `call_kind="unsupported"` in metadata, preventing standard pipeline usage.

---

## What Was Already Good (No Changes Needed)

- `arch/registry.py` — Clean, extensible registry pattern
- `arch/mednext_models.py` — Correct list-to-dict conversion for Lightning compatibility
- `loss/losses.py` — Well-structured custom losses with proper `_reduce_weighted_tensor`
- `loss/regularization.py` — Well-documented with clear mathematical motivation
- `MONAIModelWrapper` — Correct 2D/3D dimension handling with `was_5d` tracking

---

## Summary of Changes

| Category | Items Fixed | Items Kept As-Is |
|----------|-----------|-----------------|
| Correctness bugs | 2 (RSUNet ds keys, mutable default) | 0 |
| Dead code | 3 (summary, _RSUNET_AVAILABLE, convenience factories) | 1 (nnunet 2d/3d — used in tutorial) |
| DRY violations | 2 (spatial_dims, norm helpers) | 0 |
| Design improvements | 4 (dual registry, Literal types, try/except, print→logging) | 2 (CombinedLoss logging, GANLoss signature) |
| **Total** | **11 fixed** | **3 kept** |

---

## Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| **Correctness** | 9/10 | RSUNet deep supervision key bug fixed; mutable default fixed |
| **Dead code removal** | 9/10 | Convenience factories, `summary()`, `_RSUNET_AVAILABLE` all removed |
| **Code quality** | 9/10 | All `print()` replaced with `logging`; `Literal` types for metadata |
| **DRY** | 9/10 | spatial_dims and norm helpers consolidated; dual metadata registry eliminated |
| **Architecture** | 9/10 | Clean registry pattern; proper base class contract; MONAI/MedNeXt/RSUNet well-separated |
| **Maintainability** | 9/10 | Adding new arch = decorator + builder function; loss metadata is single-source |

**Overall: 9.0/10 — Clean, extensible model system. Registry pattern is solid.**

---

### Net effect
- ~80 lines of dead code removed
- ~20 lines of duplicated code consolidated
- All `print()` replaced with `logging` or `warnings.warn()`
- Type safety improved with `Literal` types
- RSUNet deep supervision now correctly matches the base class contract
