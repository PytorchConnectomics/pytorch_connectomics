# Parameter Tuning: Manual vs Optuna

## Current Manual Approach (Grid Search)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Define parameter grid                              │
│                                                             │
│  binary_threshold:   [0.7, 0.8, 0.9]                       │
│  contour_threshold:  [0.6, 0.8, 1.0]                       │
│  min_instance_size:  [16, 32, 64]                          │
│                                                             │
│  Total combinations: 3 × 3 × 3 = 27 experiments            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Run ALL 27 experiments manually                    │
│                                                             │
│  [1/27] binary=0.7, contour=0.6, size=16  → metric=0.72    │
│  [2/27] binary=0.7, contour=0.6, size=32  → metric=0.75    │
│  [3/27] binary=0.7, contour=0.6, size=64  → metric=0.71    │
│  [4/27] binary=0.7, contour=0.8, size=16  → metric=0.78    │
│  ...                                                        │
│  [27/27] binary=0.9, contour=1.0, size=64 → metric=0.85    │
│                                                             │
│  ⏱️  Time: ~5 min/exp × 27 = 135 minutes (2+ hours)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Find best result                                   │
│                                                             │
│  Best: binary=0.85, contour=0.9, size=32                   │
│  Metric: 0.89                                               │
│                                                             │
│  ❌ Problems:                                               │
│     - Tested EVERY combination (wasteful)                  │
│     - No insight into parameter importance                 │
│     - Hard to extend to more parameters                    │
│     - Might miss optimal value between grid points         │
└─────────────────────────────────────────────────────────────┘
```

**Grid Search with 5 parameters:**
- 3 values each: 3^5 = 243 experiments
- 5 values each: 5^5 = 3,125 experiments
- 10 values each: 10^5 = 100,000 experiments 😱

---

## New Optuna Approach (Bayesian Optimization)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Define parameter ranges (not grid!)               │
│                                                             │
│  binary_threshold:   [0.5, 0.95]   (continuous range)     │
│  contour_threshold:  [0.6, 1.2]    (continuous range)     │
│  min_instance_size:  [8, 128]      (integer range)        │
│                                                             │
│  Search space: INFINITE combinations                        │
│  Optuna will sample intelligently                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Optuna runs smart experiments                      │
│                                                             │
│  Trial 1: Random exploration                                │
│    binary=0.75, contour=0.85, size=32  → metric=0.82       │
│                                                             │
│  Trial 2: Random exploration                                │
│    binary=0.68, contour=0.92, size=48  → metric=0.79       │
│                                                             │
│  Trial 3: Learning from trials 1-2                          │
│    binary=0.78, contour=0.88, size=28  → metric=0.85 📈    │
│                                                             │
│  Trial 4: Exploiting promising region                       │
│    binary=0.82, contour=0.91, size=32  → metric=0.88 📈    │
│                                                             │
│  Trial 5: Exploring binary_threshold higher                 │
│    binary=0.87, contour=0.89, size=36  → metric=0.90 📈    │
│                                                             │
│  ...                                                        │
│                                                             │
│  Trial 50: Converged to optimal region                      │
│    binary=0.85, contour=0.95, size=32  → metric=0.92 🎯    │
│                                                             │
│  ⏱️  Time: ~5 min/trial × 50 = 250 minutes (4 hours)       │
│                                                             │
│  But: Found BETTER result (0.92 vs 0.89) with FEWER        │
│       informative trials! 🚀                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Analyze results                                    │
│                                                             │
│  Best: binary=0.85, contour=0.95, size=32                  │
│  Metric: 0.92 (better than grid search!)                   │
│                                                             │
│  ✅ Benefits:                                               │
│     - Parameter importance: contour > binary > size        │
│     - Optimal region identified: binary ∈ [0.82, 0.88]    │
│     - Can easily add more parameters                       │
│     - Finds values between grid points                     │
│     - Visualization of parameter interactions              │
└─────────────────────────────────────────────────────────────┘
```

---

## Visual Comparison: Search Strategy

### Grid Search (Exhaustive)
```
Parameter Space (2D example)

Contour Threshold
    1.2 │   X   X   X   X   X
    1.0 │   X   X   X   X   X
    0.8 │   X   X   X   X   X
    0.6 │   X   X   X   X   X
        └─────────────────────
        0.5  0.6  0.7  0.8  0.9
           Binary Threshold

Legend:
  X = Tested point

Total: 20 tests (every grid point)
❌ Wastes tests on clearly bad regions
❌ Can't test between grid points
```

### Optuna TPE (Intelligent)
```
Parameter Space (2D example)

Contour Threshold
    1.2 │           o
    1.0 │   o       ●   o
    0.8 │       ●   ●   ●   o
    0.6 │   o       o
        └─────────────────────
        0.5  0.6  0.7  0.8  0.9
           Binary Threshold

Legend:
  o = Early random trials
  ● = Focused trials (high-value region)

Total: 15 tests (same budget)
✅ Focuses on promising regions
✅ Tests any values (continuous)
✅ Learns from previous results
```

---

## Efficiency Comparison

### Scenario: 5 parameters, find good solution

| Method | Trials Needed | Time | Result Quality |
|--------|--------------|------|----------------|
| **Random Search** | 500+ | 40+ hours | Poor |
| **Grid Search (3 vals/param)** | 243 | 20 hours | Good |
| **Grid Search (5 vals/param)** | 3,125 | 260 hours | Better |
| **Optuna TPE** | 50-100 | 4-8 hours | Best |

**Optuna Advantage:**
- 5-50x fewer trials than grid search
- Better results (continuous search space)
- Provides parameter insights

---

## Parameter Importance Analysis

### What Grid Search Tells You:
```
Binary Threshold: ???
Contour Threshold: ???
Min Instance Size: ???

❌ No insight into which parameters matter most
```

### What Optuna Tells You:
```
Parameter Importance (Optuna)

Contour Threshold:   ████████████████████ 45%
Binary Threshold:    ████████████ 30%
Min Instance Size:   █████ 15%
Distance Threshold:  ██ 10%

✅ Now you know:
   - Focus tuning efforts on contour_threshold
   - Binary_threshold is important too
   - Min_instance_size matters less
   - Can fix less important params to save time
```

---

## Multi-Objective Example

### Goal: Maximize Precision AND Recall

**Grid Search:**
```
1. Run grid search optimizing precision
   → Find best for precision: binary=0.9, contour=0.8

2. Run another grid search optimizing recall
   → Find best for recall: binary=0.7, contour=0.95

3. Manually try to balance?
   → ??? No systematic way to find trade-offs
```

**Optuna Multi-Objective:**
```
1. Define two objectives:
   - Maximize precision
   - Maximize recall

2. Run Optuna NSGA-III sampler
   → Finds Pareto front automatically

3. Get multiple optimal trade-offs:

   Pareto Front:
   ┌─────────────────────────────────────┐
   │  Point A: precision=0.95, recall=0.75│
   │  Point B: precision=0.88, recall=0.85│
   │  Point C: precision=0.82, recall=0.92│
   └─────────────────────────────────────┘

   Choose based on your priority!
```

---

## When to Use Each Method

### Use Grid Search When:
- ✅ Very few parameters (1-2)
- ✅ Small search space
- ✅ Need to test specific discrete values
- ✅ Simple baseline comparison

### Use Optuna When:
- ✅ Many parameters (3+)
- ✅ Large search space
- ✅ Want parameter importance analysis
- ✅ Need multi-objective optimization
- ✅ Limited computational budget
- ✅ Production deployment (need best results)

---

## Example: Hydra Large Vesicle Segmentation

### Current Manual Config:
```yaml
# You manually chose these values
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs:
        binary_threshold: [0.9, 0.85]   # How did you pick this?
        contour_threshold: [0.8, 1.1]   # Trial and error?
        distance_threshold: [0.5, 0]    # Guess?
        min_instance_size: 16           # Default?
        min_seed_size: 8                # ???
```

### After Optuna Optimization:
```yaml
# Optuna found these optimal values
inference:
  decoding:
    - name: decode_binary_contour_distance_watershed
      kwargs:
        binary_threshold: 0.87      # Optimized for your data!
        contour_threshold: 0.94     # Validated on held-out set
        distance_threshold: 0.42    # Found to work best
        min_instance_size: 28       # Not 16 or 32, but 28!
        min_seed_size: 12           # Goldilocks value

# Improvement: adapted_rand = 0.89 → 0.93 (+4.5%)
```

---

## Summary

| Aspect | Manual/Grid | Optuna |
|--------|------------|--------|
| **Setup effort** | Low | Medium |
| **Runtime** | Hours/Days | Hours |
| **Result quality** | Good | Best |
| **Parameter insights** | None | Rich |
| **Scalability** | Poor | Excellent |
| **Reproducibility** | Manual | Automatic |
| **Recommended for** | Quick tests | Production |

**Bottom Line:**
- Use Optuna for any serious parameter optimization
- Saves time, finds better parameters, provides insights
- Essential for production-quality segmentation
