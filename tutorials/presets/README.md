# Augmentation Presets

Ready-to-use augmentation configurations for different use cases.

## Available Presets

### 1. Light Augmentation (`aug_light.yaml`)
**Use case:** Quick experiments, large datasets, fast iteration

**Features:**
- Missing sections (low probability)
- Misalignment (translation only)
- Minimal overhead

**Training speed:** ⚡⚡⚡ Fast
**Robustness:** ⭐⭐ Basic

```bash
python scripts/main.py --config tutorials/presets/aug_light.yaml
```

---

### 2. Realistic Augmentation (`aug_realistic.yaml`)
**Use case:** Replicating BANIS augmentation strategy

**Features:**
- Missing sections (better than BANIS)
- Misalignment (better than BANIS)
- Motion blur
- Intensity augmentation

**Training speed:** ⚡⚡ Medium
**Robustness:** ⭐⭐⭐⭐ Good

```bash
python scripts/main.py --config tutorials/presets/aug_realistic.yaml
```

**Note:** This config replicates BANIS augmentation but with superior PyTC implementations.

---

### 3. Heavy Augmentation (`aug_heavy.yaml`)
**Use case:** Small datasets, maximum robustness, production models

**Features:**
- ALL EM-specific transforms (8 total)
- High augmentation probability (70%)
- Intensity + spatial augmentation

**Training speed:** ⚡ Slower
**Robustness:** ⭐⭐⭐⭐⭐ Excellent

```bash
python scripts/main.py --config tutorials/presets/aug_heavy.yaml
```

**Note:** Requires more training epochs (150+) due to heavy augmentation.

---

### 4. Super-Resolution (`aug_superres.yaml`)
**Use case:** Training for super-resolution or multi-scale learning

**Features:**
- CutBlur (primary augmentation)
- Motion blur
- Mild misalignment

**Training speed:** ⚡⚡ Medium
**Robustness:** ⭐⭐⭐⭐ Excellent for resolution variations

```bash
python scripts/main.py --config tutorials/presets/aug_superres.yaml
```

---

### 5. Instance Segmentation (`aug_instance.yaml`)
**Use case:** Neuron instance segmentation, object detection

**Features:**
- Copy-paste augmentation
- Mixup regularization
- Geometric transforms (applied to both image and label)

**Training speed:** ⚡ Slower (copy-paste is expensive)
**Robustness:** ⭐⭐⭐⭐⭐ Excellent for instance segmentation

```bash
python scripts/main.py --config tutorials/presets/aug_instance.yaml
```

**Note:** Requires labels for copy-paste augmentation.

---

## Customization

### Modify Data Paths
All presets use placeholder paths. Update these for your dataset:

```yaml
data:
  train_image: "datasets/YOUR_DATASET/train_image.h5"
  train_label: "datasets/YOUR_DATASET/train_label.h5"
  val_image: "datasets/YOUR_DATASET/val_image.h5"
  val_label: "datasets/YOUR_DATASET/val_label.h5"
```

### Adjust Augmentation Probability
Increase or decrease `prob` for each transform:

```yaml
# More aggressive
data:
  augmentation:
    missing_section:
      enabled: true
      prob: 0.9  # Increased from 0.5

# Less aggressive
data:
  augmentation:
    missing_section:
      enabled: true
      prob: 0.2  # Decreased from 0.5
```

### Mix Presets
Combine augmentations from different presets:

```yaml
# Custom config
data:
  augmentation:
    preset: some
    # From aug_realistic.yaml
    missing_section:
      enabled: true
      prob: 0.5
      num_sections: 2
    # From aug_superres.yaml
    cut_blur:
      enabled: true
      prob: 0.5
      length_ratio: 0.25
```

---

## Comparison

| Preset | Transforms | Aug Prob | Speed | Robustness | Use Case |
|--------|-----------|----------|-------|------------|----------|
| **Light** | 2 | 30% | ⚡⚡⚡ | ⭐⭐ | Quick experiments |
| **Realistic** | 6 | 50% | ⚡⚡ | ⭐⭐⭐⭐ | BANIS-style |
| **Heavy** | 9 | 70% | ⚡ | ⭐⭐⭐⭐⭐ | Production |
| **SuperRes** | 5 | 60% | ⚡⚡ | ⭐⭐⭐⭐ | Multi-scale |
| **Instance** | 8 | 50% | ⚡ | ⭐⭐⭐⭐⭐ | Instance seg |

---

## See Also

- [EM Augmentation Guide](../../.claude/EM_AUGMENTATION_GUIDE.md) - Comprehensive documentation
- [BANIS Comparison](../../.claude/PHASE6_COMPARISON.md) - PyTC vs BANIS augmentations
- [MONAI Transforms](https://docs.monai.io/en/stable/transforms.html) - Standard augmentations
