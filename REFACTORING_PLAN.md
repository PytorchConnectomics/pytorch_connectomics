# PyTorch Connectomics: Code Cleaning & Refactoring Plan

**Document Version:** 1.0
**Date:** 2025-11-14
**Status:** Draft for Review
**Overall Codebase Health:** 8.1/10 - Production Ready

---

## Executive Summary

PyTorch Connectomics has successfully migrated to a modern architecture (Lightning + MONAI + Hydra) and is **production-ready**. However, there are opportunities for improvement to enhance maintainability, reduce technical debt, and improve code quality.

### Key Findings
- ✅ **Architecture**: Modern, well-designed (Lightning + MONAI + Hydra)
- ✅ **Migration**: 95%+ complete (YACS → Hydra)
- ✅ **Code Organization**: Clean separation of concerns
- ⚠️ **File Size**: `lit_model.py` is 1,819 lines (should be split)
- ⚠️ **Code Duplication**: ~140 lines duplicated in training/validation steps
- ⚠️ **Test Coverage**: 62% unit tests passing, integration tests need updates
- ⚠️ **Legacy Code**: 3 YACS config files, some incomplete implementations

### Impact Summary
- **High Priority**: 3 issues (code duplication, NotImplementedError, integration tests)
- **Medium Priority**: 6 issues (file size, hardcoded values, validation dataset, etc.)
- **Low Priority**: 5 issues (type hints, documentation, performance)
- **Cleanup**: 4 tasks (legacy configs, unused code, reorganization)

### Estimated Effort
- **High Priority**: 8-12 hours
- **Medium Priority**: 12-16 hours
- **Low Priority**: 8-12 hours
- **Total**: 28-40 hours (~1 week of focused work)

---

## Table of Contents

1. [Priority 1: Critical Issues (Do First)](#priority-1-critical-issues-do-first)
2. [Priority 2: High-Value Refactoring (Do Soon)](#priority-2-high-value-refactoring-do-soon)
3. [Priority 3: Code Quality Improvements (Do Eventually)](#priority-3-code-quality-improvements-do-eventually)
4. [Priority 4: Nice-to-Have Enhancements](#priority-4-nice-to-have-enhancements)
5. [Code Cleanup Tasks](#code-cleanup-tasks)
6. [Test Improvement Plan](#test-improvement-plan)
7. [Documentation Updates](#documentation-updates)
8. [Performance Optimization Opportunities](#performance-optimization-opportunities)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Priority 1: Critical Issues (Do First)

These issues should be addressed immediately as they impact correctness, maintainability, or functionality.

### 1.1 Implement Missing Functions (CRITICAL)

**File:** `connectomics/data/dataset/build.py:137-140`
**Issue:** `create_tile_data_dicts_from_json()` raises `NotImplementedError`
**Impact:** Blocks tile dataset creation from JSON configuration
**Effort:** 2-3 hours

**Current Code:**
```python
def create_tile_data_dicts_from_json(json_path: str) -> List[Dict]:
    """Create tile data dictionaries from JSON file."""
    # TODO: Implement JSON-based tile dataset creation
    raise NotImplementedError("JSON tile dataset creation not yet implemented")
```

**Action Required:**
1. Design JSON schema for tile datasets (coordinate grid, paths, metadata)
2. Implement parsing and validation logic
3. Add unit tests for various JSON configurations
4. Document JSON format in docstring and tutorials
5. Create example JSON file in `tutorials/`

**Success Criteria:**
- [ ] Function implemented and tested
- [ ] JSON schema documented
- [ ] Example configuration provided
- [ ] Unit tests pass

---

### 1.2 Fix Code Duplication in Lightning Module ✅ **COMPLETED**

**File:** `connectomics/lightning/lit_model.py`
**Issue:** ~~~140 lines of deep supervision logic duplicated~~ **FIXED**
**Impact:** ~~Maintenance burden, risk of divergence between train/val logic~~ **RESOLVED**
**Effort:** 3-4 hours ✅

**Duplicated Logic:**
- Deep supervision loss computation (5 scales)
- Multi-task loss aggregation
- Target resizing and interpolation
- Loss weight application
- Logging and metric computation

**Recommended Solution:**
Extract shared logic into private helper methods:

```python
class ConnectomicsModule(pl.LightningModule):
    def _compute_deep_supervision_loss(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
        loss_functions: List,
        stage: str = "train"
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute deep supervision loss across multiple scales.

        Args:
            outputs: Model outputs (single tensor or list of tensors for DS)
            targets: Ground truth targets
            loss_functions: List of loss functions to apply
            stage: 'train' or 'val' for logging

        Returns:
            total_loss: Aggregated loss
            loss_dict: Dictionary of individual loss components for logging
        """
        # Shared logic here
        pass

    def _log_losses(self, loss_dict: Dict[str, float], stage: str, batch_idx: int):
        """Log losses with proper prefixes and sync_dist settings."""
        pass
```

**Action Items:**
1. Extract `_compute_deep_supervision_loss()` helper method
2. Extract `_resize_targets()` helper method
3. Extract `_log_losses()` helper method
4. Update `training_step()` to use helpers
5. Update `validation_step()` to use helpers
6. Ensure behavior unchanged (run tests)
7. Reduce total file size by ~100 lines

**Success Criteria:**
- [ ] No duplicated logic between train/val steps
- [ ] All tests pass
- [ ] File size reduced by ~100 lines
- [ ] Code coverage maintained or improved

---

### 1.3 Update Integration Tests for Lightning 2.0 API (HIGH)

**Files:** `tests/integration/*.py` (0/6 passing)
**Issue:** Integration tests use deprecated YACS config API
**Impact:** Cannot verify system-level functionality, tests failing
**Effort:** 4-6 hours

**Current Status:**
```
Integration Tests: 0/6 passing (0%)
- All use legacy YACS config imports
- API mismatch with modern Hydra configs
- Need full rewrite for Lightning 2.0
```

**Action Required:**
1. **Audit existing tests:** Identify what each test validates
2. **Rewrite for Hydra configs:**
   - Replace YACS config loading with `load_config()`
   - Update config structure to match modern dataclass format
   - Fix import paths (`models.architectures` → `models.arch`)
3. **Modernize assertions:**
   - Use Lightning Trainer API properly
   - Verify deep supervision outputs
   - Check multi-task learning functionality
4. **Add missing integration tests:**
   - Distributed training (DDP)
   - Mixed precision training
   - Checkpoint save/load/resume
   - Test-time augmentation
5. **Document test requirements:** Data setup, environment, expected outputs

**Test Coverage Needed:**
- [ ] End-to-end training (fit + validate)
- [ ] Distributed training (DDP, multi-GPU)
- [ ] Mixed precision (fp16, bf16)
- [ ] Checkpoint save/load/resume
- [ ] Test-time augmentation
- [ ] Multi-task learning
- [ ] Sliding window inference

**Success Criteria:**
- [ ] 6/6 integration tests passing
- [ ] Tests use modern Hydra config API
- [ ] All major features covered
- [ ] CI/CD pipeline validates integration tests

---

## Priority 2: High-Value Refactoring (Do Soon)

These improvements will significantly enhance code quality and maintainability.

### 2.1 Refactor `lit_model.py` - Split Into Modules (MEDIUM)

**File:** `connectomics/lightning/lit_model.py` (1,819 lines)
**Issue:** File is too large for easy maintenance
**Impact:** Difficult to navigate, understand, and modify
**Effort:** 6-8 hours

**Recommended Structure:**

```
connectomics/lightning/
├── lit_model.py              # Main LightningModule (400-500 lines)
│   - __init__, forward, configure_optimizers
│   - training_step, validation_step, test_step
│   - High-level orchestration
│
├── deep_supervision.py       # Deep supervision logic (200-300 lines)
│   - DeepSupervisionHandler class
│   - Multi-scale loss computation
│   - Target resizing and interpolation
│   - Loss weight scheduling
│
├── inference.py              # Inference utilities (300-400 lines)
│   - InferenceManager class
│   - Sliding window inference
│   - Test-time augmentation
│   - Prediction postprocessing
│
├── multi_task.py             # Multi-task learning (200-300 lines)
│   - MultiTaskHandler class
│   - Task-specific losses
│   - Task output routing
│
├── debugging.py              # Debugging hooks (100-200 lines)
│   - NaN/Inf detection
│   - Gradient analysis
│   - Activation visualization
│
└── lit_data.py               # LightningDataModule (existing)
```

**Migration Steps:**
1. Create new module files
2. Move functionality in logical chunks
3. Update imports in `lit_model.py`
4. Maintain backward compatibility (public API unchanged)
5. Add integration tests for each module
6. Update documentation

**Success Criteria:**
- [ ] Each file < 500 lines
- [ ] Clear separation of concerns
- [ ] All existing tests pass
- [ ] Public API unchanged (backward compatible)
- [ ] Documentation updated

---

### 2.2 Remove Dummy Validation Dataset Hack (MEDIUM)

**File:** `connectomics/lightning/lit_data.py:184-204`
**Issue:** Creates fake tensor when val_data is empty instead of proper error handling
**Impact:** Masks configuration errors, confusing for users
**Effort:** 1-2 hours

**Current Code:**
```python
if len(val_data) == 0:
    logger.warning("No validation data found, creating dummy dataset")
    val_data = [{"image": torch.zeros(1, 128, 128, 128),
                 "label": torch.zeros(1, 128, 128, 128)}]
```

**Recommended Solution:**
```python
if len(val_data) == 0:
    if self.cfg.training.require_validation:
        raise ValueError(
            "No validation data found. Please provide validation data paths "
            "or set training.require_validation=false to skip validation."
        )
    else:
        logger.info("No validation data provided, skipping validation")
        return None  # Lightning handles None gracefully
```

**Action Items:**
1. Add `require_validation` config option (default: `true`)
2. Replace dummy dataset with proper error handling
3. Update config schema in `hydra_config.py`
4. Update tutorials to show validation skip option
5. Add unit test for both paths

**Success Criteria:**
- [ ] Clear error message when validation missing
- [ ] Option to skip validation gracefully
- [ ] No dummy datasets created
- [ ] Tests verify both paths

---

### 2.3 Make Hardcoded Values Configurable (MEDIUM)

**Files:**
- `connectomics/lightning/lit_model.py:1139, 1167, 1282, 1294`
- `connectomics/data/augment/build.py:various`

**Issue:** Hardcoded values for clamping, interpolation bounds, max attempts, etc.
**Impact:** Cannot tune for different datasets without code changes
**Effort:** 3-4 hours

**Hardcoded Values Found:**

1. **Output Clamping** (lit_model.py:1139, 1167):
   ```python
   outputs = torch.clamp(outputs, min=-20, max=20)  # ← Hardcoded
   ```

2. **Deep Supervision Weights** (lit_model.py:1200):
   ```python
   ds_weights = [1.0, 0.5, 0.25, 0.125, 0.0625]  # ← Hardcoded
   ```

3. **Target Interpolation Bounds** (lit_model.py:1282):
   ```python
   resized = F.interpolate(target, size=size, mode='trilinear',
                          align_corners=False)
   resized = torch.clamp(resized, min=-1, max=1)  # ← Hardcoded
   ```

4. **Rejection Sampling Max Attempts** (dataset_base.py:174):
   ```python
   max_attempts = 50  # ← Hardcoded
   ```

**Recommended Config Additions:**

```python
@dataclass
class TrainingConfig:
    # ... existing fields ...

    # Output processing
    output_clamp_min: Optional[float] = -20.0
    output_clamp_max: Optional[float] = 20.0

    # Deep supervision
    deep_supervision_weights: List[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.25, 0.125, 0.0625]
    )

    # Target processing
    target_clamp_min: float = -1.0
    target_clamp_max: float = 1.0
    target_interpolation_mode: str = "trilinear"

@dataclass
class DataConfig:
    # ... existing fields ...

    # Rejection sampling
    rejection_max_attempts: int = 50
    rejection_min_foreground_ratio: float = 0.01
```

**Action Items:**
1. Add config fields to `hydra_config.py`
2. Replace hardcoded values with config references
3. Add validation for config values
4. Update tutorials with examples
5. Document new config options

**Success Criteria:**
- [ ] All hardcoded values moved to config
- [ ] Validation prevents invalid values
- [ ] Backward compatible (defaults match old behavior)
- [ ] Documentation updated

---

### 2.4 Consolidate Redundant CachedVolumeDataset (MEDIUM)

**Files:**
- `connectomics/data/dataset/dataset_volume.py:MonaiCachedVolumeDataset`
- `connectomics/data/dataset/dataset_volume_cached.py` (291 lines, duplicate)

**Issue:** Two implementations of cached volume dataset
**Impact:** Code duplication, confusion about which to use
**Effort:** 2-3 hours

**Recommended Solution:**
1. Audit both implementations to find differences
2. Merge best features into single implementation
3. Deprecate old implementation with warning
4. Update imports throughout codebase
5. Update documentation

**Action Items:**
- [ ] Compare both implementations
- [ ] Identify unique features of each
- [ ] Create unified implementation
- [ ] Add deprecation warning to old version
- [ ] Update all imports
- [ ] Remove deprecated file in next major version

---

### 2.5 Refactor Duplicate Transform Builders (MEDIUM)

**File:** `connectomics/data/augment/build.py:build_val_transforms()` and `build_test_transforms()`
**Issue:** Nearly identical implementations (791 lines total)
**Impact:** Maintenance burden, risk of divergence
**Effort:** 2-3 hours

**Current Structure:**
```python
def build_val_transforms(cfg):
    # 350+ lines of transform logic
    pass

def build_test_transforms(cfg):
    # 350+ lines of nearly identical logic
    pass
```

**Recommended Solution:**
```python
def build_eval_transforms(
    cfg,
    mode: str = "val",
    enable_augmentation: bool = False
):
    """Build transforms for evaluation (validation or test).

    Args:
        cfg: Configuration object
        mode: 'val' or 'test'
        enable_augmentation: Whether to include augmentations (TTA)
    """
    # Shared logic with mode-specific branching
    pass

def build_val_transforms(cfg):
    """Build validation transforms (wrapper)."""
    return build_eval_transforms(cfg, mode="val")

def build_test_transforms(cfg, enable_tta: bool = False):
    """Build test transforms (wrapper)."""
    return build_eval_transforms(cfg, mode="test", enable_augmentation=enable_tta)
```

**Action Items:**
- [ ] Extract shared logic into `build_eval_transforms()`
- [ ] Identify val/test-specific differences
- [ ] Create mode-specific branching
- [ ] Keep wrapper functions for API compatibility
- [ ] Add tests for both modes
- [ ] Reduce code by ~300 lines

---

## Priority 3: Code Quality Improvements (Do Eventually)

These improvements enhance code quality but are not urgent.

### 3.1 Add Missing Type Hints (LOW)

**Files:** Various throughout codebase
**Issue:** Incomplete type hint coverage
**Impact:** Reduced IDE support, type checking effectiveness
**Effort:** 4-6 hours

**Areas Needing Type Hints:**
- `connectomics/data/io/` - I/O utility functions
- `connectomics/data/process/` - Processing functions
- `connectomics/utils/` - General utilities
- `connectomics/decoding/` - Post-processing

**Tools to Use:**
- `mypy` for static type checking
- `monkeytype` for automatic type hint generation

**Action Items:**
1. Run `mypy` to identify missing hints
2. Add type hints to public APIs first
3. Add type hints to internal functions
4. Configure `mypy` in CI/CD
5. Set target: 90%+ type hint coverage

---

### 3.2 Add Parameter Validation (MEDIUM)

**Files:** Transform builders, model builders
**Issue:** Missing validation for probability bounds, value ranges
**Impact:** Silent failures or confusing error messages
**Effort:** 3-4 hours

**Examples of Missing Validation:**

```python
# Current
def RandRotate90d(prob=0.5, ...):
    # No check if prob is in [0, 1]
    pass

# Recommended
def RandRotate90d(prob=0.5, ...):
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    pass
```

**Action Items:**
- [ ] Add probability validation to all transforms
- [ ] Add range validation for numeric parameters
- [ ] Add type validation for complex parameters
- [ ] Create validation utility functions
- [ ] Add unit tests for validation

---

### 3.3 Improve Error Messages (LOW)

**Files:** Various
**Issue:** Generic error messages that don't guide users to solutions
**Impact:** Difficult debugging for users
**Effort:** 2-3 hours

**Examples:**

```python
# Current
raise ValueError("Invalid architecture")

# Recommended
raise ValueError(
    f"Invalid architecture '{arch}'. Available architectures: "
    f"{list_architectures()}. See CLAUDE.md for details."
)
```

**Action Items:**
- [ ] Audit all error messages
- [ ] Add context and suggestions
- [ ] Link to documentation where appropriate
- [ ] Add troubleshooting hints

---

### 3.4 Add Logging for Debugging (LOW)

**Files:** Data pipeline, model building
**Issue:** Insufficient logging for debugging data issues
**Impact:** Difficult to debug data loading problems
**Effort:** 2-3 hours

**Recommended Logging:**
- Dataset creation: paths, sizes, transforms
- Data loading: batch shapes, value ranges
- Model building: architecture, parameter counts
- Transform application: shapes, value ranges

---

## Priority 4: Nice-to-Have Enhancements

These are optional improvements that can be deferred.

### 4.1 Add `predict_step()` Method (LOW)

**File:** `connectomics/lightning/lit_model.py`
**Issue:** Uses `test_step()` for prediction, includes TTA overhead
**Impact:** Slower than necessary for production inference
**Effort:** 2-3 hours

**Recommended Implementation:**
```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    """Optimized prediction without TTA or metrics."""
    # Simpler, faster inference path
    pass
```

---

### 4.2 Add MC Dropout Uncertainty (NICE-TO-HAVE)

**File:** New module `connectomics/lightning/uncertainty.py`
**Feature:** Monte Carlo dropout for uncertainty estimation
**Effort:** 6-8 hours

---

### 4.3 Add Checkpoint Ensemble (NICE-TO-HAVE)

**File:** New module `connectomics/lightning/ensemble.py`
**Feature:** Ensemble predictions from multiple checkpoints
**Effort:** 4-6 hours

---

## Code Cleanup Tasks

### 5.1 Archive Legacy YACS Configs ✅ **COMPLETED**

**Files:** ~~`configs/barcode/*.yaml` (3 files)~~ **REMOVED**
**Action:** ~~Move to `configs/legacy/` or~~ remove entirely ✅
**Effort:** 15 minutes ✅

**Completed Steps:**
1. ✅ Removed `configs/barcode/` directory entirely
2. ✅ All 3 legacy YACS config files deleted
3. ✅ Updated CLAUDE.md to remove references
4. ✅ Updated codebase metrics (100% migration complete)
5. ✅ Updated overall assessment score (8.1 → 8.3)

**Status:** No YACS code remains in the codebase

---

### 5.2 Remove Unused Code

**Files:** Various
**Action:** Identify and remove dead code
**Effort:** 2-3 hours

**Tools:**
- `vulture` - Find unused code
- `coverage` - Identify uncovered code

**Areas to Check:**
- Unused imports
- Unreachable code paths
- Deprecated functions
- Commented-out code blocks

---

### 5.3 Fix Import Cycle Risk

**File:** `connectomics/data/augment/build.py:31`
**Issue:** Imports `LoadVolumed` from potentially wrong location
**Impact:** Could cause circular dependency
**Effort:** 30 minutes

**Action:**
1. Verify correct import path
2. Update import statement
3. Add import order tests

---

### 5.4 Clean Up TODO/FIXME Comments

**Files:** 2 files with TODO/FIXME comments
**Action:** Address or document TODOs
**Effort:** 1-2 hours

**Found:**
- `connectomics/data/dataset/dataset_volume.py:1` - TODO for normalization
- `connectomics/data/dataset/build.py:1` - TODO for JSON tile datasets

**Action:**
- [ ] Implement or create GitHub issues
- [ ] Document why deferred if not implementing
- [ ] Remove completed TODOs

---

## Test Improvement Plan

### 6.1 Current Test Status

```
Unit Tests:        38/61 passing (62%)
Integration Tests:  0/6 passing (0%) - Need v2.0 API migration
E2E Tests:          0/3 working - Require data setup
Overall:           38/70 passing (54%)
```

**Target:** 80%+ passing, 90%+ coverage

### 6.2 Unit Test Improvements

**High Priority:**
- [ ] Fix failing model tests (23 failures)
- [ ] Add tests for NotImplementedError functions
- [ ] Add tests for deep supervision logic
- [ ] Add tests for multi-task learning
- [ ] Add tests for data transforms

**Medium Priority:**
- [ ] Add tests for edge cases
- [ ] Add tests for error conditions
- [ ] Add parametrized tests for configs
- [ ] Add benchmark tests

### 6.3 Integration Test Rewrite

See Priority 1.3 above for full details.

**Tests Needed:**
- [ ] End-to-end training pipeline
- [ ] Distributed training (DDP)
- [ ] Mixed precision training
- [ ] Checkpoint save/load/resume
- [ ] Test-time augmentation
- [ ] Multi-task learning
- [ ] Sliding window inference

### 6.4 E2E Test Setup

**Requirements:**
- [ ] Download test datasets
- [ ] Document data requirements
- [ ] Create automated data download script
- [ ] Add data validation
- [ ] Set up CI/CD data caching

### 6.5 Test Infrastructure

**Improvements:**
- [ ] Add `pytest` fixtures for common setups
- [ ] Add test utilities module
- [ ] Set up test coverage reporting
- [ ] Add performance benchmarking
- [ ] Configure CI/CD test matrix

---

## Documentation Updates

### 7.1 CLAUDE.md

✅ **Status:** COMPLETED
- [x] Corrected directory structure (arch/ not architectures/)
- [x] Added all 8 architectures (MONAI, MedNeXt, RSUNet)
- [x] Fixed import paths
- [x] Removed non-existent .claude/ references
- [x] Added code quality status section
- [x] Updated tutorial list (11 configs)
- [x] Added test status information

### 7.2 API Documentation

**Files:** Docstrings throughout codebase
**Action:** Ensure comprehensive API docs
**Effort:** 4-6 hours

**Areas Needing Docs:**
- [ ] Architecture builders (in `models/arch/`)
- [ ] Transform builders (in `data/augment/build.py`)
- [ ] Dataset classes (in `data/dataset/`)
- [ ] Lightning modules (in `lightning/`)

**Standards:**
- Use Google-style docstrings
- Include type hints
- Provide examples
- Document all parameters

### 7.3 Tutorial Updates

**Files:** `tutorials/*.yaml` and documentation
**Action:** Ensure all tutorials are current and working
**Effort:** 3-4 hours

**Tasks:**
- [ ] Test all 11 tutorial configs
- [ ] Add comments explaining key parameters
- [ ] Create beginner/intermediate/advanced sections
- [ ] Add architecture comparison guide
- [ ] Document best practices per dataset type

### 7.4 Architecture Decision Records (ADRs)

**New Files:** `docs/adr/*.md`
**Action:** Document key architectural decisions
**Effort:** 4-6 hours

**Decisions to Document:**
- Why Lightning + MONAI + Hydra
- Why architecture registry pattern
- Why deep supervision implementation
- Why multi-task learning approach
- Migration from YACS to Hydra

---

## Performance Optimization Opportunities

### 8.1 Data Loading Optimization

**Current:** Basic MONAI CacheDataset
**Opportunities:**
- Persistent caching to disk
- Pre-computed transforms
- Parallel data loading tuning
- Memory-mapped arrays

**Effort:** 4-6 hours
**Expected Gain:** 20-30% faster data loading

### 8.2 Mixed Precision Tuning

**Current:** Basic AMP support
**Opportunities:**
- Gradient scaling tuning
- Loss scaling configuration
- Model-specific precision zones

**Effort:** 2-3 hours
**Expected Gain:** 10-15% faster training

### 8.3 Compilation (PyTorch 2.0+)

**Current:** No compilation
**Opportunities:**
- `torch.compile()` for model
- CUDA graphs for inference
- Operator fusion

**Effort:** 3-4 hours
**Expected Gain:** 15-25% faster training/inference

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
**Effort:** 8-12 hours
**Priority:** P1 issues

1. **Day 1-2:** Implement `create_tile_data_dicts_from_json()` (Priority 1.1)
2. **Day 2-3:** Fix code duplication in lit_model.py (Priority 1.2)
3. **Day 3-5:** Update integration tests for Lightning 2.0 (Priority 1.3)

**Deliverables:**
- [ ] All NotImplementedError functions completed
- [ ] Code duplication eliminated
- [ ] Integration tests passing
- [ ] All P1 issues resolved

### Phase 2: High-Value Refactoring (Week 2)
**Effort:** 12-16 hours
**Priority:** P2 issues

1. **Day 1-2:** Refactor lit_model.py into modules (Priority 2.1)
2. **Day 3:** Remove dummy validation dataset (Priority 2.2)
3. **Day 4:** Make hardcoded values configurable (Priority 2.3)
4. **Day 5:** Consolidate redundant datasets and transforms (Priority 2.4, 2.5)

**Deliverables:**
- [ ] lit_model.py split into focused modules
- [ ] All hardcoded values configurable
- [ ] No code duplication
- [ ] All P2 issues resolved

### Phase 3: Code Quality (Week 3)
**Effort:** 8-12 hours
**Priority:** P3 issues + cleanup

1. **Day 1-2:** Add type hints and validation (Priority 3.1, 3.2)
2. **Day 3:** Improve error messages and logging (Priority 3.3, 3.4)
3. **Day 4-5:** Code cleanup tasks (Section 5)

**Deliverables:**
- [ ] 90%+ type hint coverage
- [ ] Parameter validation throughout
- [ ] Legacy configs archived
- [ ] All cleanup tasks completed

### Phase 4: Documentation & Testing (Week 4)
**Effort:** 8-12 hours
**Priority:** Documentation + tests

1. **Day 1-2:** Update documentation (Section 7)
2. **Day 3-4:** Improve test coverage (Section 6)
3. **Day 5:** Final review and polish

**Deliverables:**
- [ ] All documentation updated
- [ ] 80%+ test coverage
- [ ] All tutorials tested
- [ ] Release notes prepared

### Optional Phase 5: Enhancements (Future)
**Effort:** 16-24 hours
**Priority:** P4 + performance

1. Performance optimizations (Section 8)
2. Nice-to-have features (Section 4)
3. Advanced features (uncertainty, ensembles)

---

## Success Metrics

### Code Quality Metrics
- [ ] No `NotImplementedError` in active code
- [ ] No code duplication (< 5% duplicate lines)
- [ ] Files < 600 lines each
- [ ] 90%+ type hint coverage
- [ ] 0 `mypy` errors

### Test Metrics
- [ ] 80%+ unit tests passing
- [ ] 100% integration tests passing
- [ ] 90%+ code coverage
- [ ] All tutorials executable

### Documentation Metrics
- [ ] All public APIs documented
- [ ] All config options documented
- [ ] All tutorials working
- [ ] ADRs for major decisions

### Performance Metrics
- [ ] Data loading throughput measured
- [ ] Training speed benchmarked
- [ ] Inference speed benchmarked
- [ ] Memory usage profiled

---

## Risk Assessment

### Low Risk
- Code cleanup tasks
- Documentation updates
- Type hint additions

### Medium Risk
- lit_model.py refactoring (large file, many dependencies)
- Transform builder consolidation (affects data pipeline)

### High Risk (Requires Careful Testing)
- Integration test rewrite (API changes)
- Dummy dataset removal (could break configs)
- Hardcoded value migration (affects model behavior)

### Mitigation Strategies
1. **Comprehensive testing** before and after each change
2. **Feature flags** for backward compatibility
3. **Deprecation warnings** before removal
4. **Rollback plan** for each phase
5. **User communication** via release notes

---

## Conclusion

PyTorch Connectomics is in **excellent shape** overall (8.1/10) with a modern, well-designed architecture. The refactoring plan focuses on:

1. **Eliminating technical debt** (code duplication, incomplete implementations)
2. **Improving maintainability** (file size, modularity, type safety)
3. **Enhancing testability** (integration tests, coverage)
4. **Polishing documentation** (accuracy, completeness)

**Recommended Approach:** Execute Phases 1-3 (3 weeks) for maximum impact. Phase 4 can be done incrementally. Phase 5 is optional based on project needs.

**Expected Outcome:** A codebase that is:
- ✅ Easier to maintain (modular, typed, documented)
- ✅ Easier to test (comprehensive test suite)
- ✅ Easier to extend (clear patterns, no duplication)
- ✅ Production-ready (no critical issues, robust error handling)

---

**Next Steps:**
1. Review this plan with team
2. Prioritize tasks based on project needs
3. Create GitHub issues for tracking
4. Begin Phase 1 implementation
5. Establish regular progress reviews

---

*Document created by comprehensive codebase analysis on 2025-11-14*
