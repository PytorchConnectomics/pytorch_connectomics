# Phase 8: Weighted Dataset Mixing

## Summary

Implements multi-dataset utilities for mixing synthetic and real data with controllable proportions, enabling domain adaptation and transfer learning.

## Motivation

Training on **mixed datasets** improves generalization:
- **Synthetic data:** Unlimited labeled data, but may have domain gap
- **Real data:** Limited labels, but matches target domain
- **Mixed:** Combines advantages of both

BANIS demonstrated that mixing 80% synthetic + 20% real data achieves better performance than either alone.

## Implementation

### Files Added

1. **`connectomics/data/dataset/dataset_multi.py` (227 lines)**
   - `WeightedConcatDataset` - weighted random sampling
   - `StratifiedConcatDataset` - round-robin balanced sampling
   - `UniformConcatDataset` - size-proportional sampling

2. **`tests/test_dataset_multi.py` (310 lines)**
   - 18 comprehensive test cases
   - Tests for all 3 dataset classes
   - Integration tests for complete workflows

3. **`tutorials/mixed_data_example.yaml`**
   - Example configuration for mixed training
   - Documents all mixing strategies

### Files Modified

**`connectomics/data/dataset/__init__.py`**
- Added exports for new dataset classes

## Key Components

### 1. WeightedConcatDataset (Recommended)

**Purpose:** Sample from datasets with specified weights

```python
from connectomics.data.dataset import WeightedConcatDataset

# 80% synthetic, 20% real
mixed_dataset = WeightedConcatDataset(
    datasets=[synthetic_dataset, real_dataset],
    weights=[0.8, 0.2],
    length=1000  # Samples per epoch
)
```

**Features:**
- Weights must sum to 1.0 (validated)
- Random sampling according to weights
- Configurable epoch length
- Independent of dataset sizes

**Use case:** Domain adaptation (more synthetic, less real)

### 2. StratifiedConcatDataset

**Purpose:** Balanced round-robin sampling

```python
from connectomics.data.dataset import StratifiedConcatDataset

# Alternate between datasets
mixed_dataset = StratifiedConcatDataset(
    datasets=[dataset1, dataset2, dataset3]
)
# Returns: dataset1[0], dataset2[0], dataset3[0], dataset1[1], ...
```

**Features:**
- Cycles through datasets sequentially
- Ensures balanced representation
- Length = sum of dataset lengths (by default)

**Use case:** Equal contribution from each dataset

### 3. UniformConcatDataset

**Purpose:** Random sampling proportional to dataset sizes

```python
from connectomics.data.dataset import UniformConcatDataset

# Random sampling, larger datasets contribute more
mixed_dataset = UniformConcatDataset(
    datasets=[large_dataset, small_dataset]
)
```

**Features:**
- Samples uniformly from combined pool
- Larger datasets have higher sampling probability
- Similar to `torch.utils.data.ConcatDataset`

**Use case:** Natural mixing without manual weighting

## Usage

### Example 1: Domain Adaptation (Synthetic + Real)

```yaml
# tutorials/mixed_data_example.yaml
data:
  # Enable mixed datasets
  use_mixed_datasets: true
  mixing_strategy: weighted  # or stratified, uniform

  # Dataset weights (for 'weighted' strategy)
  dataset_weights: [0.8, 0.2]  # 80% synthetic, 20% real

  # Dataset paths
  datasets:
    - type: synthetic
      image: "datasets/synthetic/train_image.h5"
      label: "datasets/synthetic/train_label.h5"

    - type: real
      image: "datasets/real/train_image.h5"
      label: "datasets/real/train_label.h5"

  # Samples per epoch
  samples_per_epoch: 1000

  # Other data config...
  batch_size: 4
  num_workers: 8
```

### Example 2: Programmatic Usage

```python
from connectomics.data.dataset import WeightedConcatDataset
from torch.utils.data import DataLoader

# Create individual datasets
synthetic_dataset = create_dataset(
    image_path="datasets/synthetic/train_image.h5",
    label_path="datasets/synthetic/train_label.h5",
    transforms=train_transforms,
)

real_dataset = create_dataset(
    image_path="datasets/real/train_image.h5",
    label_path="datasets/real/train_label.h5",
    transforms=train_transforms,
)

# Mix datasets (80% synthetic, 20% real)
mixed_dataset = WeightedConcatDataset(
    datasets=[synthetic_dataset, real_dataset],
    weights=[0.8, 0.2],
    length=1000,  # 1000 samples per epoch
)

# Create data loader
train_loader = DataLoader(
    mixed_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
)

# Train
for epoch in range(100):
    for batch in train_loader:
        # 80% of batches will be from synthetic data
        # 20% of batches will be from real data
        train_step(batch)
```

### Example 3: Multiple Datasets

```python
# Mix 3 datasets with different weights
mixed_dataset = WeightedConcatDataset(
    datasets=[dataset_A, dataset_B, dataset_C],
    weights=[0.5, 0.3, 0.2],  # 50%, 30%, 20%
    length=2000,
)
```

## Testing

### Run Tests
```bash
# All dataset mixing tests
pytest tests/test_dataset_multi.py -v

# Specific test class
pytest tests/test_dataset_multi.py::TestWeightedConcatDataset -v

# With coverage
pytest tests/test_dataset_multi.py --cov=connectomics.data.dataset.dataset_multi
```

### Test Coverage

**18 Tests across 5 test classes:**
1. **TestWeightedConcatDataset** (7 tests)
   - Basic functionality
   - Weight validation
   - Sampling distribution
   - Custom length
   - Edge cases

2. **TestStratifiedConcatDataset** (5 tests)
   - Round-robin sampling
   - Balanced representation
   - Different dataset sizes

3. **TestUniformConcatDataset** (4 tests)
   - Uniform sampling
   - Size-proportional distribution
   - Edge cases

4. **TestIntegration** (2 tests)
   - DataLoader integration
   - End-to-end training workflow

### Example Test
```python
def test_weighted_distribution():
    """Test that sampling follows specified weights."""
    dataset1 = DummyDataset(value=1.0, size=100)
    dataset2 = DummyDataset(value=2.0, size=100)

    mixed = WeightedConcatDataset(
        datasets=[dataset1, dataset2],
        weights=[0.8, 0.2]
    )

    # Sample 1000 times
    samples = [mixed[i] for i in range(1000)]

    count_1 = sum(1 for s in samples if s == 1.0)
    count_2 = sum(1 for s in samples if s == 2.0)

    # Should be approximately 80/20
    assert 750 < count_1 < 850  # 80% ± 5%
    assert 150 < count_2 < 250  # 20% ± 5%
```

## Benefits

1. **Domain Adaptation** - Leverage unlimited synthetic data + limited real data
2. **Transfer Learning** - Pre-train on synthetic, fine-tune on real
3. **Data Augmentation** - Increase dataset diversity
4. **Controllable Mixing** - Precise control over dataset proportions
5. **Multiple Strategies** - Weighted, stratified, uniform sampling
6. **PyTorch Compatible** - Works with standard DataLoader
7. **Well-Tested** - 18 comprehensive tests

## Comparison with BANIS

### BANIS Implementation
```python
# BANIS: Basic weighted concatenation
class WeightedConcatDataset:
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights

    def __getitem__(self, index):
        dataset_idx = np.random.choice(len(self.datasets), p=self.weights)
        sample_idx = np.random.randint(len(self.datasets[dataset_idx]))
        return self.datasets[dataset_idx][sample_idx]
```

### PyTC Implementation

**Improvements over BANIS:**
- ✅ **3 mixing strategies** (weighted, stratified, uniform)
- ✅ **Weight validation** (must sum to 1.0)
- ✅ **Configurable epoch length**
- ✅ **Comprehensive testing** (18 tests)
- ✅ **Example configuration**
- ✅ **Better error messages**
- ✅ **Integration with Lightning**

## Use Cases

### 1. Domain Adaptation
**Problem:** Limited real labeled data
**Solution:** Mix 80% synthetic + 20% real
```python
mixed = WeightedConcatDataset(
    [synthetic_data, real_data],
    weights=[0.8, 0.2]
)
```

### 2. Curriculum Learning
**Problem:** Model struggles with difficult examples
**Solution:** Start with easy dataset, gradually increase hard dataset
```python
# Epoch 1-10: 90% easy, 10% hard
# Epoch 11-20: 70% easy, 30% hard
# Epoch 21+: 50% easy, 50% hard
weights = get_curriculum_weights(epoch)
mixed = WeightedConcatDataset([easy_data, hard_data], weights)
```

### 3. Multi-Organ Segmentation
**Problem:** Different organs have different amounts of data
**Solution:** Balance contribution from each organ
```python
mixed = StratifiedConcatDataset([
    liver_data,    # 500 samples
    kidney_data,   # 200 samples
    spleen_data,   # 100 samples
])
# Each organ contributes equally despite size differences
```

### 4. Transfer Learning
**Problem:** Pre-training on large dataset, fine-tuning on small target
**Solution:**
```python
# Stage 1: Pre-train on large dataset
train(large_dataset, epochs=100)

# Stage 2: Fine-tune with mixed data
mixed = WeightedConcatDataset(
    [large_dataset, target_dataset],
    weights=[0.5, 0.5]  # Gradually reduce large_dataset weight
)
train(mixed, epochs=50)
```

## Documentation

- Docstrings with examples for all classes
- Example configuration: `tutorials/mixed_data_example.yaml`
- Updated `.claude/CLAUDE.md` with usage examples
- Integration guide in documentation

## Checklist

- [x] Implementation complete (3 dataset classes)
- [x] Tests passing (18/18)
- [x] Documentation complete
- [x] Example configuration created
- [x] Integration with DataLoader verified
- [ ] Create GitHub issue for tracking
- [ ] Merge to main branch

## Related

- Implements BANIS_PLAN.md Phase 8
- Part of BANIS integration (Phases 6-10)
- Complements augmentation pipeline (Phase 6)
- Works with Lightning DataModule

## Future Enhancements

1. **Dynamic Weighting** - Adjust weights based on validation performance
2. **Class Balancing** - Weight by class distribution
3. **Hard Example Mining** - Oversample difficult examples
4. **Multi-Modal Mixing** - Mix different imaging modalities
5. **Caching Support** - Integrate with MONAI CacheDataset
6. **Distributed Sampling** - Coordinate sampling across GPUs
