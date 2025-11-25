# Mode Integration Implementation Status

## ✅ Completed

### 1. CLI Arguments Extended
- ✅ Added `--mode` choices: `tune`, `tune+test`, `infer`
- ✅ Added `--params` for parameter file override
- ✅ Added `--param-source` for parameter source selection
- ✅ Added `--tune-trials` for Optuna trial count override

### 2. Mode Handling in main.py
- ✅ Added early exit for `tune` and `tune+test` modes with helpful messages
- ✅ Added `infer` mode (alias for `test`)
- ✅ Updated all mode checks to include new modes
- ✅ Stub implementation with clear development status

### 3. justfile Commands
- ✅ `just tune <dataset> <ckpt>` - Tune decoding parameters
- ✅ `just tune-test <dataset> <ckpt>` - Tune then test
- ✅ `just tune-quick <dataset> <ckpt>` - Quick tuning (20 trials)
- ✅ `just test-with-params <dataset> <ckpt> <params>` - Use specific params
- ✅ `just infer <dataset> <ckpt>` - Inference (alias for test)

### 4. Documentation
- ✅ `.claude/MODE_INTEGRATION_DESIGN.md` - Full design document
- ✅ `tutorials/MODE_INTEGRATION_SUMMARY.md` - User-friendly summary
- ✅ `tutorials/optuna_decoding_tuning.yaml` - Example config
- ✅ `tutorials/unified_inference_tuning.yaml` - Unified config
- ✅ All Optuna documentation (9 files, 113KB)

## 🚧 In Progress

### Core Implementation Needed
- ⬜ `connectomics/decoding/optuna_tuner.py` - OptunaDecodingTuner class
- ⬜ Implement `run_tuning_workflow()` in main.py
- ⬜ Implement `run_tuning_and_inference_workflow()` in main.py
- ⬜ Parameter loading/saving utilities
- ⬜ Integration with existing decoders

### Config System
- ⬜ Add Optuna config dataclasses to `hydra_config.py`
- ⬜ Parameter space validation
- ⬜ Config merging for parameter sources

## Usage (Current Status)

### ✅ Working Now
```bash
# Traditional modes (fully functional)
just train hydra-lv
just test hydra-lv checkpoints/best.ckpt
just infer hydra-lv checkpoints/best.ckpt  # Alias for test

# Help for new modes (under development)
just tune hydra-lv checkpoints/best.ckpt
just tune-test hydra-lv checkpoints/best.ckpt
```

### 🚧 Coming Soon
```bash
# Will work after implementation
just tune hydra-lv checkpoints/best.ckpt
# → Runs Optuna optimization
# → Saves best_params.yaml
# → Generates plots

just tune-test hydra-lv checkpoints/best.ckpt
# → Stage 1: Optimize on validation
# → Stage 2: Test with best params
```

## Testing

### Test Commands
```bash
# Test mode recognition
python scripts/main.py --mode tune

# Test with config
python scripts/main.py --config tutorials/hydra-lv.yaml --mode tune --checkpoint dummy.ckpt

# Test justfile commands
just tune hydra-lv dummy.ckpt
just tune-test hydra-lv dummy.ckpt
just infer hydra-lv dummy.ckpt
```

### Expected Output (tune mode)
```
================================================================================
🎯 TUNE MODE
================================================================================

⚠️  This mode is under development.

To use Optuna parameter tuning:
  1. Install dependencies: pip install -e .[optim]
  2. See: tutorials/optuna_decoding_tuning.yaml
  3. See: tutorials/unified_inference_tuning.yaml
  4. See: .claude/MODE_INTEGRATION_DESIGN.md

For now, use:
  • --mode test with manual parameters in config

💡 Implementation tracked in:
  • .claude/MODE_INTEGRATION_DESIGN.md
  • .claude/OPTUNA_DECODING_DESIGN.md
================================================================================
```

## Next Steps

### Priority 1: Core Optuna Integration
1. Create `connectomics/decoding/optuna_tuner.py`
   - OptunaDecodingTuner class
   - Parameter sampling
   - Objective function
   - Study management

2. Implement tune workflow in `main.py`
   - Load model
   - Run inference on validation
   - Run Optuna optimization
   - Save results

3. Test with simple example
   - Small dataset
   - Few trials (5-10)
   - Verify end-to-end

### Priority 2: Config System
1. Add Optuna dataclasses to `hydra_config.py`
2. Add parameter_source handling
3. Add parameter loading utilities

### Priority 3: Full Integration
1. Implement tune+test workflow
2. Add parameter file loading
3. Add visualization generation
4. Update documentation

## Files Modified

### Core Files
- ✅ `scripts/main.py` - Added modes, CLI args, stub implementation
- ✅ `justfile` - Added tune commands

### Documentation Files (New)
- ✅ `.claude/MODE_INTEGRATION_DESIGN.md`
- ✅ `.claude/OPTUNA_DECODING_DESIGN.md`
- ✅ `tutorials/MODE_INTEGRATION_SUMMARY.md`
- ✅ `tutorials/optuna_decoding_tuning.yaml`
- ✅ `tutorials/unified_inference_tuning.yaml`
- ✅ `tutorials/OPTUNA_QUICKSTART.md`
- ✅ `tutorials/optuna_comparison.md`
- ✅ `tutorials/UNIFIED_CONFIG_GUIDE.md`
- ✅ `tutorials/README_OPTUNA.md`
- ✅ `tutorials/OPTUNA_SUMMARY.md`
- ✅ `tutorials/optuna_architecture_diagram.txt`

## Summary

**Status:** Foundation complete, ready for core implementation

**What Works:**
- ✅ All CLI arguments recognized
- ✅ Mode dispatch working
- ✅ justfile commands functional
- ✅ Helpful error messages
- ✅ Comprehensive documentation

**What's Next:**
- Implement OptunaDecodingTuner class
- Implement workflow functions in main.py
- Add config system support
- Test end-to-end

**Estimated Implementation Time:**
- Core Optuna integration: 4-8 hours
- Config system: 2-4 hours
- Testing and refinement: 2-4 hours
- **Total: 1-2 days of focused work**
