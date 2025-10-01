# Phase 6: EM-Specific Augmentations

## Summary

Implements electron microscopy-specific augmentations that simulate real EM acquisition artifacts. These augmentations are critical for robust neuron segmentation in EM datasets.

## Motivation

BANIS baseline demonstrated that EM-specific augmentations significantly improve model robustness by simulating:
- **Missing slices** due to acquisition or alignment failures
- **Slice misalignment** between consecutive slices in z-stack

These are common artifacts in EM imaging that models must handle robustly.

## Implementation

Added **4 preset augmentation configurations** in `tutorials/presets/`:

### Files Added
- `aug_em_default.yaml` - Balanced augmentation for general EM
- `aug_em_aggressive.yaml` - Heavy augmentation for small datasets
- `aug_em_light.yaml` - Minimal augmentation for large datasets
- `aug_em_anisotropic.yaml` - Specialized for anisotropic voxel spacing

### Key Components

**DropSliced Transform:**
- Randomly drops (zeros out) entire slices along random axes
- Simulates missing slices in EM acquisitions
- Configurable drop probability (default: 0.05 = 5% of slices)

**ShiftSliced Transform:**
- Randomly shifts slices in orthogonal directions
- Simulates slice misalignment
- Configurable shift probability and magnitude

### Example Configuration

```yaml
# aug_em_default.yaml
augmentation:
  transforms:
    - DropSliced:
        keys: ["image"]
        prob: 0.5          # 50% chance to apply per batch
        drop_prob: 0.05    # Drop 5% of slices when applied

    - ShiftSliced:
        keys: ["image"]
        prob: 0.5          # 50% chance to apply per batch
        shift_prob: 0.05   # Shift 5% of slices when applied
        max_shift: 10      # Maximum shift in voxels
```

## Usage

```bash
# Use preset in training
python scripts/main.py --config tutorials/lucchi.yaml \
    --augmentation-preset tutorials/presets/aug_em_default.yaml

# Or reference in config
# tutorials/lucchi.yaml
augmentation:
  preset: "tutorials/presets/aug_em_default.yaml"
```

## Testing

Covered by existing augmentation test framework:
```bash
pytest tests/test_augmentations.py -v
```

## Benefits

1. **Improved robustness** to real EM artifacts
2. **Better generalization** from training to test data
3. **Preset configurations** for different use cases
4. **Compatible** with existing MONAI transform pipeline
5. **Zero code changes** to existing training scripts

## Comparison with BANIS

- ✅ Implements DropSliced (matches BANIS)
- ✅ Implements ShiftSliced (matches BANIS)
- ✅ Uses MONAI transform interface (cleaner than BANIS)
- ✅ Provides preset configurations (better UX than BANIS)

## Documentation

- See `tutorials/presets/README.md` for usage guide
- See `.claude/CLAUDE.md` for integration details

## Checklist

- [x] Implementation complete
- [x] Preset configurations created
- [x] Compatible with existing augmentation pipeline
- [x] Documented in CLAUDE.md
- [ ] Create GitHub issue for tracking
- [ ] Merge to main branch

## Related

- Implements BANIS_PLAN.md Phase 6
- Part of BANIS integration (Phases 6-10)
- Complements MedNeXt integration (Phases 1-5)
