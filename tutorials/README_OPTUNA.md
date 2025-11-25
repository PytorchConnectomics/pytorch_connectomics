# Optuna Parameter Optimization for PyTorch Connectomics

## 📚 Documentation Overview

This directory contains comprehensive documentation for Optuna-based hyperparameter optimization of decoding/post-processing parameters.

### Quick Navigation

1. **[OPTUNA_QUICKSTART.md](OPTUNA_QUICKSTART.md)** - Start here!
   - 5-minute introduction to Optuna parameter tuning
   - Step-by-step usage guide
   - Before/after comparison

2. **[optuna_comparison.md](optuna_comparison.md)** - Detailed comparison
   - Manual grid search vs Optuna
   - Visual diagrams and examples
   - Efficiency analysis

3. **[optuna_decoding_tuning.yaml](optuna_decoding_tuning.yaml)** - Full example config
   - Complete YAML configuration template
   - All options documented with comments
   - Ready to customize and use

4. **[.claude/OPTUNA_DECODING_DESIGN.md](../.claude/OPTUNA_DECODING_DESIGN.md)** - Technical design
   - Architecture and implementation details
   - Advanced features (multi-objective, distributed, etc.)
   - For developers and advanced users

## 🚀 TL;DR - Get Started in 3 Steps

### Step 1: Copy and customize the example config
```bash
cp tutorials/optuna_decoding_tuning.yaml my_tuning_config.yaml
# Edit: data paths, model checkpoint, parameter ranges
```

### Step 2: Run optimization
```bash
python scripts/tune_decoding.py --config my_tuning_config.yaml
```

### Step 3: Use optimized parameters
```bash
# Copy best parameters from outputs/optuna_decoding_tuning/best_params.yaml
# to your main config and run inference
python scripts/main.py --config my_config.yaml --mode test
```

## 🎯 What Problem Does This Solve?

### Before Optuna ❌
```yaml
inference:
  decoding:
    kwargs:
      binary_threshold: 0.9     # Guess
      contour_threshold: 0.8    # Trial and error
      min_instance_size: 16     # Default
      # ❌ Don't know if these are optimal
      # ❌ Don't know which parameters matter
      # ❌ Tedious manual tuning
```

### After Optuna ✅
```yaml
inference:
  decoding:
    kwargs:
      binary_threshold: 0.87    # Optimized by Optuna!
      contour_threshold: 0.94   # Found automatically
      min_instance_size: 28     # Data-driven choice
      # ✅ Proven optimal for your data
      # ✅ Know parameter importance
      # ✅ Automated tuning
```

**Result:** Better segmentation quality with less effort!

## 📖 Key Concepts

### 1. Parameter Search Space
Instead of manually picking values, define a range:
```yaml
parameter_space:
  parameters:
    binary_threshold:
      type: float
      range: [0.5, 0.95]  # Search this range
```

### 2. Optimization Objective
Tell Optuna what to optimize:
```yaml
optuna:
  optimization:
    single_objective:
      metric: adapted_rand  # Maximize this
      direction: maximize
```

### 3. Intelligent Search
Optuna uses Bayesian optimization (TPE sampler):
- Learns from previous trials
- Focuses on promising regions
- 10-50x more efficient than grid search

## 📊 Example Results

### Parameter Importance
```
Which parameters matter most for adapted_rand?

Contour Threshold:   ████████████████████ 45%
Binary Threshold:    ████████████ 30%
Min Instance Size:   █████ 15%
Distance Threshold:  ██ 10%
```

### Optimization History
```
Trial  | Binary | Contour | Size | Adapted Rand
-------|--------|---------|------|-------------
   1   |  0.75  |   0.85  |  32  |    0.823
   5   |  0.80  |   0.90  |  28  |    0.856  📈
  10   |  0.85  |   0.92  |  32  |    0.891  📈
  20   |  0.87  |   0.94  |  28  |    0.912  📈
  50   |  0.87  |   0.95  |  32  |    0.919  🎯 BEST
```

## 🔧 Customization Guide

### 1. Choose Parameters to Optimize

**High Priority (always optimize):**
- `binary_threshold`: Foreground/background separation
- `contour_threshold`: Instance boundary detection
- `distance_threshold`: Seed placement for watershed

**Medium Priority (optimize if time permits):**
- `min_instance_size`: Small object removal
- `min_seed_size`: Minimum watershed seed size

**Low Priority (usually keep fixed):**
- `use_numba`: Always true (performance)
- `scale_factors`: Usually [1, 1, 1]

### 2. Choose Optimization Metric

**Volume-based metrics:**
- `adapted_rand`: Good general-purpose metric (0-1, higher better)
- `voi_sum`: Variation of Information (lower better)
- `precision`, `recall`, `f1_score`: Standard classification metrics

**Skeleton-based metrics** (requires skeleton file):
- `nerl`: Normalized Expected Run Length (0-1, higher better)
- `erl`: Expected Run Length

### 3. Set Number of Trials

| Use Case | Trials | Time |
|----------|--------|------|
| Quick test | 20-30 | 1-2 hours |
| Standard | 50-100 | 4-8 hours |
| Thorough | 100-200 | 8-16 hours |
| Production | 200+ | 16+ hours |

### 4. Single vs Multi-Objective

**Single-Objective** (easier, faster):
```yaml
optimization:
  mode: single
  single_objective:
    metric: adapted_rand
    direction: maximize
```

**Multi-Objective** (find trade-offs):
```yaml
optimization:
  mode: multi
  multi_objective:
    objectives:
      - metric: adapted_rand
        direction: maximize
      - metric: voi_sum
        direction: minimize
```

## 🎓 Learning Resources

### For Beginners
1. Read: [OPTUNA_QUICKSTART.md](OPTUNA_QUICKSTART.md)
2. Try: Use provided [optuna_decoding_tuning.yaml](optuna_decoding_tuning.yaml)
3. Experiment: Modify parameter ranges for your data

### For Advanced Users
1. Read: [OPTUNA_DECODING_DESIGN.md](../.claude/OPTUNA_DECODING_DESIGN.md)
2. Implement: Custom objective functions
3. Explore: Multi-objective optimization, distributed optimization

### External Resources
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [TPE Paper](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)

## 💡 Tips and Best Practices

### 1. Start with Reasonable Ranges
```yaml
# ❌ Too wide
binary_threshold:
  range: [0.0, 1.0]  # Wastes trials on obviously bad values

# ✅ Reasonable
binary_threshold:
  range: [0.5, 0.95]  # Based on domain knowledge
```

### 2. Use Log Scale for Wide Ranges
```yaml
# For parameters spanning orders of magnitude
min_instance_size:
  type: int
  range: [8, 1024]
  log: true  # Sample evenly in log space
```

### 3. Cache Predictions
```yaml
inference:
  run_inference: false  # Don't re-run inference
  prediction_path: "outputs/cached_predictions.h5"
```
Saves time if predictions are already computed!

### 4. Resume Interrupted Studies
```yaml
optuna:
  study_name: "my_optimization"
  storage: "sqlite:///optuna_study.db"
  load_if_exists: true  # Resume if interrupted
```

### 5. Run Parallel Workers
```bash
# All workers share the same database
python scripts/tune_decoding.py --config config.yaml &
python scripts/tune_decoding.py --config config.yaml &
python scripts/tune_decoding.py --config config.yaml &
```

## 🐛 Troubleshooting

### Issue: No improvement after many trials
**Solution:**
- Check parameter ranges are reasonable
- Increase number of trials (50 → 100+)
- Verify metric is computed correctly
- Try different sampler (TPE → CmaEs)

### Issue: Optimization very slow
**Solution:**
- Cache predictions (don't re-run inference)
- Use smaller validation set
- Run parallel workers
- Reduce number of trials

### Issue: Results not reproducible
**Solution:**
- Set `system.seed` in config
- Use persistent storage (`storage: "sqlite://..."`)
- Save study after optimization

## 📁 File Structure

```
tutorials/
├── optuna_decoding_tuning.yaml       # Example config (start here!)
├── OPTUNA_QUICKSTART.md              # Quick start guide
├── optuna_comparison.md               # Detailed comparison
└── README_OPTUNA.md                   # This file

.claude/
└── OPTUNA_DECODING_DESIGN.md          # Technical design doc

scripts/
└── tune_decoding.py                   # CLI tool (to be implemented)

connectomics/decoding/
├── auto_tuning.py                     # Existing threshold tuning
└── optuna_tuner.py                    # New Optuna tuner (to be implemented)
```

## 🚧 Implementation Status

### ✅ Completed
- Design document
- Example configuration
- Quick start guide
- Comparison analysis

### 🚧 In Progress
- `scripts/tune_decoding.py` (CLI tool)
- `connectomics/decoding/optuna_tuner.py` (core implementation)
- Integration with existing decoders
- Visualization and reporting

### 📋 TODO
- Unit tests
- Integration tests
- Example notebooks
- Tutorial video
- CLAUDE.md updates

## 🤝 Contributing

Want to contribute? Areas that need help:
1. Custom objective functions for specific use cases
2. New decoder implementations
3. Visualization improvements
4. Documentation improvements
5. Example notebooks

## 📝 Citation

If you use Optuna parameter optimization in your research, please cite:

```bibtex
@inproceedings{akiba2019optuna,
  title={Optuna: A next-generation hyperparameter optimization framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery \& data mining},
  pages={2623--2631},
  year={2019}
}
```

## 📞 Support

- **Questions?** Open an issue on GitHub
- **Bug reports?** Use issue tracker
- **Feature requests?** Submit via GitHub discussions
- **Slack community:** [Join here](https://join.slack.com/t/pytorchconnectomics/shared_invite/zt-obufj5d1-v5_NndNS5yog8vhxy4L12w)

---

**Happy optimizing! 🎯🚀**
