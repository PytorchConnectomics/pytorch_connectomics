# Phase 6 vs Existing RandMis* Functions: Comparison

## Summary

**Conclusion: Phase 6 is largely REDUNDANT.** PyTorch Connectomics already has excellent EM-specific slice augmentations that are MORE sophisticated than BANIS's implementations.

---

## Feature-by-Feature Comparison

### 1. Missing Slices / Slice Dropout

| Feature | BANIS `DropSliced` | PyTC `RandMissingSectiond` | Winner |
|---------|-------------------|---------------------------|---------|
| **Functionality** | Set random slices to zero | Delete random slices from volume | **PyTC** |
| **Implementation** | Sets to 0 (preserves shape) | Removes slices (changes shape) | **PyTC** (more realistic) |
| **Probability Control** | Per-slice probability | Number of sections to remove | **Tie** (different approaches) |
| **MONAI Compliance** | ✅ Uses MapTransform | ✅ Uses MapTransform | ✅ Both |
| **Code Quality** | Simple, clear | Simple, clear | ✅ Both |

**BANIS:**
```python
# Set random slices to zero based on per-slice probability
drop_mask = self.R.rand(n_slices) < self.drop_prob
for i in range(n_slices):
    if drop_mask[i]:
        img[..., i, :, :] = 0  # Set to zero
```

**PyTC:**
```python
# Actually remove slices (more realistic simulation)
indices_to_remove = self.R.choice(
    np.arange(1, img.shape[0] - 1),
    size=num_to_remove,
    replace=False
)
return np.delete(img, indices_to_remove, axis=0)
```

**Verdict:** **PyTC is better** - actually removes slices (more realistic) vs just zeroing them.

---

### 2. Slice Shifting / Misalignment

| Feature | BANIS `ShiftSliced` | PyTC `RandMisAlignmentd` | Winner |
|---------|-------------------|--------------------------|---------|
| **Shift Type** | Circular shift (roll) | Translation with cropping | **PyTC** |
| **Rotation Support** | ❌ No | ✅ Yes (with rotation matrix) | **PyTC** |
| **Modes** | Single mode | "slip" and "translation" modes | **PyTC** |
| **Realism** | Circular wrap (unrealistic) | Proper translation (realistic) | **PyTC** |
| **Flexibility** | Per-slice shifts | Configurable slip/translation | **PyTC** |

**BANIS:**
```python
# Uses torch.roll (circular shift - unrealistic)
slice_data = torch.roll(slice_data, shifts=shift, dims=roll_axis)
# Problem: Wraps pixels from right to left (not realistic)
```

**PyTC:**
```python
# Translation mode: crops and translates
output[:idx] = img[:idx, y0:y0+out_shape[1], x0:x0+out_shape[2]]
output[idx:] = img[idx:, y1:y1+out_shape[1], x1:x1+out_shape[2]]

# Rotation mode: uses cv2.warpAffine (proper geometric transform)
M = cv2.getRotationMatrix2D((height/2, height/2), rand_angle, 1)
img[idx] = cv2.warpAffine(img[idx], M, (height, width), ...)
```

**Verdict:** **PyTC is MUCH better** - realistic translation, rotation support, multiple modes.

---

### 3. Missing Parts / Holes

| Feature | BANIS | PyTC `RandMissingPartsd` | Winner |
|---------|-------|-------------------------|---------|
| **Functionality** | ❌ Not in BANIS | ✅ Rectangular holes | **PyTC** |
| **Hole Size** | N/A | Configurable ratio (0.1-0.3) | **PyTC** |
| **Location** | N/A | Random section + position | **PyTC** |

**Verdict:** **PyTC has it**, BANIS doesn't.

---

### 4. Additional PyTC Features Not in BANIS

| Transform | Description | Value |
|-----------|-------------|-------|
| **RandMotionBlurd** | Directional blur (horizontal/vertical kernels) | ✅ HIGH |
| **RandCutNoised** | Add noise to random cuboid regions | ✅ MEDIUM |
| **RandCutBlurd** | Downsample cuboid regions (super-resolution) | ✅ HIGH |
| **RandMixupd** | Mix samples for regularization | ✅ MEDIUM |
| **RandCopyPasted** | Copy-paste objects with rotation | ✅ HIGH |

**None of these are in BANIS.**

---

## Code Quality Comparison

### BANIS Strengths
✅ **Simpler implementation** (easier to understand)
✅ **Independent per-slice probability** (fine-grained control)
✅ **Preserves volume shape** (no shape changes)

### BANIS Weaknesses
❌ **Less realistic** (circular shifts, zero-filling)
❌ **Limited functionality** (no rotation, no modes)
❌ **Fewer features** (only 2 augmentations vs PyTC's 8)

### PyTC Strengths
✅ **More realistic** (proper translation, rotation, deletion)
✅ **More features** (8 transforms vs BANIS's 2)
✅ **Better modes** (slip/translation, rotation/translation)
✅ **Higher quality** (cv2 transforms, proper geometric operations)

### PyTC Weaknesses
⚠️ **More complex** (harder to understand)
⚠️ **Changes shapes** (deletion changes volume size)

---

## Recommendation

### ❌ DO NOT implement Phase 6 as originally proposed

**Reasons:**
1. **PyTC already has better implementations** of the same concepts
2. **PyTC has MORE features** than BANIS (8 vs 2)
3. **PyTC's implementations are MORE realistic** (proper geometric transforms)
4. **No value added** - would just duplicate existing functionality

### ✅ INSTEAD, do this:

#### Option A: Document Existing Features (Recommended)
Create a guide showing how to use PyTC's existing augmentations for EM data:

**File:** `tutorials/em_augmentation_guide.md`
```markdown
# EM-Specific Augmentation Guide

PyTorch Connectomics has comprehensive EM-specific augmentations:

## Missing Slices
Use `RandMissingSectiond` to simulate missing sections:
```yaml
data:
  augmentation:
    transforms:
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.5
          num_sections: 2
```

## Slice Misalignment
Use `RandMisAlignmentd` for realistic misalignment:
```yaml
      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.5
          displacement: 16
          rotate_ratio: 0.5  # 50% rotation, 50% translation
```

## Additional Features
- Motion blur: `RandMotionBlurd`
- Missing parts: `RandMissingPartsd`
- CutBlur (super-resolution): `RandCutBlurd`
- Copy-paste: `RandCopyPasted`
```

#### Option B: Minor Enhancements (Optional)
If you want to add BANIS-style per-slice probability control:

**File:** `connectomics/data/augment/monai_transforms.py`

Add parameter to `RandMisAlignmentd`:
```python
class RandMisAlignmentd(RandomizableTransform, MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        displacement: int = 16,
        rotate_ratio: float = 0.0,
        per_slice_prob: float = None,  # NEW: BANIS-style per-slice
        allow_missing_keys: bool = False,
    ) -> None:
        # ... existing code ...
        self.per_slice_prob = per_slice_prob

    def _apply_misalignment_translation(self, img: np.ndarray) -> np.ndarray:
        if self.per_slice_prob is not None:
            # BANIS-style: apply to each slice independently
            for i in range(1, img.shape[0] - 1):
                if self.R.rand() < self.per_slice_prob:
                    # Apply shift to this slice
                    # ... existing shift logic ...
        else:
            # Original PyTC behavior
            # ... existing code ...
```

---

## Updated Phase 6 Plan

### Phase 6 (Revised): EM Augmentation Documentation & Minor Enhancements

**Week 6 Tasks:**

1. **Create EM Augmentation Guide** (2 days)
   - Document all existing EM augmentations
   - Provide usage examples
   - Show recommended configs for different datasets
   - Compare to other frameworks (BANIS, nnUNet)

2. **Add Configuration Presets** (1 day)
   - `tutorials/augmentation_presets/em_light.yaml`
   - `tutorials/augmentation_presets/em_heavy.yaml`
   - `tutorials/augmentation_presets/em_realistic.yaml`

3. **Optional: Add Per-Slice Probability** (2 days)
   - Add `per_slice_prob` parameter to `RandMisAlignmentd`
   - Add `per_slice_prob` parameter to `RandMissingSectiond`
   - Maintain backward compatibility

4. **Add Tests** (0.5 days)
   - Test all existing EM augmentations
   - Ensure transforms work with Lightning

5. **Update Documentation** (0.5 days)
   - Update `CLAUDE.md` with augmentation details
   - Add examples to README

**Total: 6 days instead of 5 days, with better value**

---

## Comparison Table: BANIS vs PyTC Augmentations

| Augmentation | BANIS | PyTC | PyTC Quality |
|--------------|-------|------|--------------|
| Slice dropout | `DropSliced` (zeros) | `RandMissingSectiond` (deletes) | ⭐⭐⭐⭐⭐ BETTER |
| Slice shifting | `ShiftSliced` (circular) | `RandMisAlignmentd` (translation+rotation) | ⭐⭐⭐⭐⭐ MUCH BETTER |
| Missing parts | ❌ | `RandMissingPartsd` | ⭐⭐⭐⭐ |
| Motion blur | ❌ | `RandMotionBlurd` | ⭐⭐⭐⭐ |
| Cut noise | ❌ | `RandCutNoised` | ⭐⭐⭐ |
| Cut blur | ❌ | `RandCutBlurd` | ⭐⭐⭐⭐⭐ |
| Mixup | ❌ | `RandMixupd` | ⭐⭐⭐ |
| Copy-paste | ❌ | `RandCopyPasted` | ⭐⭐⭐⭐⭐ |

**Score: PyTC 8, BANIS 2** (and PyTC's 2 are better quality)

---

## Example Configs

### BANIS-Style Config (using PyTC transforms)
```yaml
# tutorials/banis_style_augmentation.yaml
data:
  augmentation:
    transforms:
      # Missing sections (better than BANIS's DropSliced)
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.5
          num_sections: 2  # More explicit than per-slice prob

      # Misalignment (better than BANIS's ShiftSliced)
      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.5
          displacement: 16  # BANIS uses max_shift=10
          rotate_ratio: 0.0  # Set to 0 for pure translation like BANIS

      # Additional features BANIS doesn't have
      - RandMotionBlurd:
          keys: ["image"]
          prob: 0.5
          sections: 2
          kernel_size: 11

      - RandCutBlurd:
          keys: ["image"]
          prob: 0.5
          length_ratio: 0.25
```

### Heavy EM Augmentation
```yaml
# tutorials/augmentation_presets/em_heavy.yaml
data:
  augmentation:
    transforms:
      - RandMissingSectiond:
          keys: ["image"]
          prob: 0.7
          num_sections: 3

      - RandMisAlignmentd:
          keys: ["image"]
          prob: 0.7
          displacement: 20
          rotate_ratio: 0.5

      - RandMissingPartsd:
          keys: ["image"]
          prob: 0.5
          hole_range: [0.1, 0.3]

      - RandMotionBlurd:
          keys: ["image"]
          prob: 0.5
          sections: [1, 3]
          kernel_size: 11

      - RandCutBlurd:
          keys: ["image"]
          prob: 0.5
          length_ratio: 0.25
          down_ratio_range: [2.0, 8.0]
```

---

## Conclusion

**PyTorch Connectomics already has SUPERIOR EM augmentations compared to BANIS.**

### Actions:
1. ✅ **Skip** implementing new `DropSliced` and `ShiftSliced` (redundant)
2. ✅ **Document** existing PyTC augmentations (better approach)
3. ✅ **Create** example configs showing how to replicate BANIS augmentation strategy
4. ⚠️ **Optional** Add per-slice probability if users request it

### Updated Refactoring Priority:
- ~~Phase 6: Slice Augmentations~~ → **SKIP (already exists)**
- **Phase 6 (NEW): Documentation & Presets** → **DO THIS**
- Phase 7: Numba Connected Components → **HIGH PRIORITY** (not in PyTC)
- Phase 8: Weighted Datasets → **MEDIUM PRIORITY** (not in PyTC)
- Phase 9: Skeleton Metrics → **MEDIUM PRIORITY** (not in PyTC)
- Phase 10: Auto-Config → **HIGH PRIORITY** (not in PyTC)

**Time saved: ~1 week, redirect to more valuable features!**
