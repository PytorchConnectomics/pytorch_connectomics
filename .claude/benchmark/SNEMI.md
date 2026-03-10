# SNEMI3D Benchmark: PyTC vs DeepEM Gap Analysis

Reference: Lee et al., "Superhuman Accuracy on the SNEMI3D Connectomics Challenge" (NIPS 2017)
Config: `tutorials/neuron_snemi.yaml`
Code ref: `lib/DeepEM/` (original implementation)

## Paper Results (Target)

| Configuration | Rand Error | TTA Variants | Params |
|---|---|---|---|
| aug3-long (TTA=16) | **0.02576** | 16 | 1.5M |
| aug3-long (TTA=8) | 0.02590 | 8 | 1.5M |
| aug3-long (mean aggl.) | 0.03332 | 1 | 1.5M |
| Human accuracy | 0.05998 | - | - |

## Discrepancies (Ranked by Expected Impact)

### 1. CRITICAL: Training Duration (5-7x too short)

| | DeepEM (paper) | PyTC |
|---|---|---|
| **Total iterations** | 500K - 700K | 100K (100 epochs x 1000 steps) |

The model has not converged. This is likely the single largest contributor to the performance gap.

**Fix:** Increase to at least 500K iterations. Either:
- `max_epochs: 500` with `n_steps_per_epoch: 1000` (= 500K)
- Or `max_epochs: 200` with `n_steps_per_epoch: 3000` (= 600K)

### 2. CRITICAL: Optimizer Configuration (completely different regime)

| | DeepEM (paper, Sec 6.3) | PyTC |
|---|---|---|
| **Optimizer** | Adam | AdamW |
| **Learning rate** | **0.01** | 0.0003 |
| **epsilon** | **0.01** | 1e-8 |
| **betas** | (0.9, 0.999) | (0.9, 0.999) |
| **weight_decay** | 0 | 0.01 |

The paper uses an unusually high `eps=0.01` which dampens Adam's adaptive step sizes, making it behave more like SGD with momentum. Combined with `lr=0.01` (33x higher than PyTC), this creates a very specific optimization regime that the authors tuned for convergence on SNEMI.

**Fix:** Create a dedicated optimizer profile for the paper recipe:
```yaml
optimizer:
  name: Adam         # Not AdamW
  lr: 0.01           # Paper's learning rate
  betas: [0.9, 0.999]
  eps: 0.01           # Critical: large epsilon dampens adaptive LR
  # No weight_decay (plain Adam, not AdamW)
```

### 3. CRITICAL: Learning Rate Schedule

| | DeepEM (paper) | PyTC |
|---|---|---|
| **Schedule** | Halve LR when val loss plateaus, up to 4x | WarmupCosineLR |
| **LR trajectory** | 0.01 → 0.005 → 0.0025 → 0.00125 → 0.000625 | 3e-4 → cosine decay → 1e-6 |

The paper's plateau-based halving allows the model to train at high LR for extended periods, then make sharp step-downs. Cosine annealing decays continuously from the start, spending most training at lower LRs.

**Fix:**
```yaml
optimization:
  optimizer:
    name: Adam
    lr: 0.01
    eps: 0.01
  scheduler:
    name: ReduceLROnPlateau
    monitor: val/loss
    mode: min
    factor: 0.5        # Halve
    patience: 50        # In epochs; tune based on val frequency
    min_lr: 0.000625    # 0.01 / 2^4
```

### 4. HIGH: Architecture Depth (missing finest-scale level)

| | DeepEM (paper Fig. 1) | PyTC |
|---|---|---|
| **Width** | [18, 36, 48, 64, 80] | [36, 48, 64, 80] |
| **Levels** | 5 (depth=4) | 4 (depth=3) |
| **Downsampling steps** | 4 | 3 |
| **IO kernel** | (1, 5, 5) | (1, 3, 3) via depth_2d |

PyTC is missing the 18-channel finest-scale level. This means:
- ~20% fewer parameters and reduced model capacity
- One fewer encoder/decoder layer pair → smaller effective receptive field
- The paper's bottleneck sees 10x10 spatial features (160/2^4), PyTC sees 28x28 (224/2^3) — proportionally less global context

The paper's finest level (18 channels) uses exclusively 2D convolutions where anisotropy is maximal. PyTC's `depth_2d=1` on the 36-channel level partially compensates but at different capacity.

**Fix:**
```yaml
model:
  rsunet:
    width: [18, 36, 48, 64, 80]    # Full 5-level architecture
    depth_2d: 1                      # Finest level = 2D convs
    down_factors: [[1,2,2], [1,2,2], [1,2,2], [1,2,2]]  # 4 downsampling steps
```

### 5. HIGH: Patch Size Mismatch

| | DeepEM (paper Sec 6.3) | PyTC |
|---|---|---|
| **Patch size** | 160 x 160 x 18 | 224 x 224 x 16 |
| **Batch size** | 1 | 4 (default) |

The paper uses 18 slices in Z (vs 16 in PyTC) and 160x160 in XY (vs 224x224). With the 5-level architecture, 160x160 maps to a 10x10 bottleneck — well-matched to the model depth. PyTC's 224x224 with only 4 levels gives a 28x28 bottleneck — underutilizing the depth.

The paper also uses batch_size=1. With batch_size=4, the effective gradient is averaged over 4 samples, changing the noise level and interacting with the learning rate.

**Fix:**
```yaml
data:
  dataloader:
    patch_size: [18, 160, 160]    # Match paper
    batch_size: 1                  # Match paper
model:
  input_size: [18, 160, 160]
  output_size: [18, 160, 160]
```

### 6. MODERATE: Loss Function ✅

| | DeepEM | PyTC (before fix) | PyTC (after fix) |
|---|---|---|---|
| **Loss** | BCE only | BCE + DiceLoss(weight=0.5) | BCE only ✅ |
| **Class balancing** | Per-edge: w_pos = n_neg/(n_pos+n_neg), w_neg = n_pos/(n_pos+n_neg) | pos_weight = n_neg/n_pos (capped at 10) | Per-channel pos_weight = n_neg/n_pos (capped at 10) ✅ |
| **Loss structure** | 12 independent per-edge losses summed | Single loss over all 12 channels | 12 independent per-channel losses via `PerChannelBCEWithLogitsLoss` ✅ |

**Fix implemented** — new `PerChannelBCEWithLogitsLoss` loss class:
- Computes BCE independently per output channel, then sums — matching DeepEM's per-edge structure
- Vectorized per-channel `pos_weight` computation: counts pos/neg voxels per channel over batch+spatial dims, computes `min(n_neg/n_pos, 10.0)` per channel, broadcasts via `(1, C, 1, ..., 1)` shape
- Accepts `weight` mask from the orchestrator (affinity valid mask flows through correctly per channel)
- Per-channel reduction ensures each channel's mean is computed only over its own valid voxels
- DiceLoss removed (BCE only, matching paper)

Files changed:
- `connectomics/models/loss/losses.py`: added `PerChannelBCEWithLogitsLoss`
- `connectomics/models/loss/build.py`: registered in loss registry
- `connectomics/models/loss/metadata.py`: added metadata with `spatial_weight_arg="weight"`
- `connectomics/training/loss/plan.py`: defaults `pos_weight` to `1.0` for this loss (prevents orchestrator from double-weighting; the loss handles class balancing internally)
- `tutorials/bases/loss_profiles.yaml`: new `loss_per_channel` profile (single entry)
- `tutorials/bases/pipeline_profiles.yaml`: `affinity-12` pipeline now uses `loss_per_channel` profile

Config (via `affinity-12` pipeline profile → `loss_per_channel` loss profile):
```yaml
# loss_profiles.yaml
loss_per_channel:
  - function: PerChannelBCEWithLogitsLoss
    weight: 1.0
    kwargs: {auto_pos_weight: true}
```
No per-tutorial loss config needed — the pipeline profile handles it.

### 7. HIGH: Affinity Border Masking (DeepEM `get_pair` logic)

| | DeepEM | PyTC (before fix) | PyTC (after fix) |
|---|---|---|---|
| **Border handling** | Per-channel `get_pair` crop + mask | Uniform `deepem_crop` (max-offset spatial crop) | Per-channel valid mask ✅ |
| **Augment padding on labels** | Mask propagated through augmentation | `RandAffined` reflection padding on labels → false affinities | Per-channel mask excludes border artifacts ✅ |

**Problem:** Two interacting issues caused border artifacts in affinity targets:

1. **Reflection padding on labels during augmentation**: `RandAffined` and `RandElasticd` use `padding_mode="reflection"` for all keys including labels. When spatial transforms rotate/scale/shear a patch, border pixels are filled with reflected label values. Computing affinity from these reflected labels creates false affinities — especially visible for long-range channels like ch11 (offset `0-27-0`) where the reflected region spans 27 voxels.

2. **Uniform spatial crop vs per-channel masking**: The old `deepem_crop` computed the **union** of all offsets' invalid borders and uniformly cropped all channels to this smallest valid region. For the SNEMI 12-channel offsets, this meant cropping (4, 27, 27) from all channels — even short-range channels that only need 1 voxel cropped. This wasted ~35% of training data for short-range channels.

**DeepEM's approach**: In DeepEM, `get_pair(arr, edge)` extracts two aligned crops per channel, computing affinity only in the overlap region. A separate mask (propagated through augmentation) excludes padded regions from the loss. Each channel has its own valid region.

**Fix implemented** (`affinity.py`, `orchestrator.py`):
- Added `compute_affinity_valid_mask(offsets, spatial_shape)` — builds a per-channel binary mask matching DeepEM's `get_pair` valid regions
- The loss orchestrator now uses this mask instead of uniform spatial cropping
- The mask is merged into `combined_mask_tensor` (for `masked_fill` path) and folded into `spatial_weight_tensor` after pos_weight computation (preserving class-balanced weighting)
- Per-channel data efficiency with patch size [16, 224, 224]:

| Channel | Offset | Valid (per-channel mask) | Valid (old uniform crop) |
|---|---|---|---|
| Ch0 | 0-0-1 | 99.6% | 65.5% |
| Ch2 | 1-0-0 | 93.8% | 65.5% |
| Ch5 | 4-0-0 | 75.0% | 65.5% |
| Ch11 | 0-27-0 | 87.9% | 65.5% |

### 8. MODERATE: Augmentation Differences

#### 8a. Mutual Exclusion of Defect Augmentations

DeepEM applies misalignment and missing section as **mutually exclusive** via `Blend(mutex)`:
```python
# DeepEM aug_v2.py
mutex = [
    Blend([Misalign(0,17), SlipMisalign(0,17), None], props=[0.5, 0.2, 0.3]),
    MixedMissingSection(maxsec=5, individual=True, random=True, skip=0.1)
]
augs.append(Blend(mutex))  # Only ONE fires per sample
```

PyTC applies all three (misalignment, missing section, motion blur) **independently** with prob=1.0 each. This means every training sample gets all three defect types simultaneously, which is much more aggressive than the paper and could corrupt training signal.

#### 8b. Warping/Elastic Deformation

DeepEM uses `Warp(skip=0.3, do_twist=False, rot_max=45.0)` — a smooth elastic warping.
The `aug_em_neuron` profile includes elastic deformation, but it's unclear whether the neuron_snemi.yaml overrides properly preserve it.

#### 8c. Grayscale/Contrast Augmentation

DeepEM: `MixedGrayscale2D(contrast_factor=0.5, brightness_factor=0.5, prob=1, skip=0.3)` — aggressive ±50% contrast and brightness.
PyTC default intensity: `contrast_range: [0.9, 1.1]` — much milder.

**Fix:** Make misalignment/missing/blur mutually exclusive (apply only one per sample), verify elastic and grayscale augmentation strengths match.

### 9. LOW-MODERATE: Inference Blending

| | DeepEM (paper Sec 2.2) | PyTC |
|---|---|---|
| **Blending** | Bump function: f(r) = exp(Σ[r_a(p_a-r_a)]^(-t_a)) | MONAI Gaussian |
| **Parameters** | t_x=t_y=t_z=1.5 | sigma_scale=0.125 |
| **Overlap** | 50% | 50% |

The bump function has sharper edges than Gaussian blending, which can reduce boundary artifacts. PyTC has a `blend_bump()` implementation in `connectomics/data/process/blend.py` but doesn't use it in the inference pipeline.

**Fix:** Consider switching to bump blending, but this is lower priority than training-side fixes.

## Recommended Action Plan

### Phase 1: Match the Training Recipe (highest ROI)

These three changes together should close most of the gap:

```yaml
# In neuron_snemi.yaml
train:
  optimization:
    # 1. Match optimizer
    optimizer:
      name: Adam
      lr: 0.01
      eps: 0.01
      betas: [0.9, 0.999]
    # 2. Match LR schedule
    scheduler:
      name: ReduceLROnPlateau
      monitor: val/loss
      mode: min
      factor: 0.5
      patience: 50
      min_lr: 0.000625
    # 3. Match training duration
    max_epochs: 600
    n_steps_per_epoch: 1000

  data:
    dataloader:
      batch_size: 1
      patch_size: [18, 160, 160]
```

### Phase 2: Match the Architecture

```yaml
default:
  model:
    arch:
      profile: null
      type: rsunet
    rsunet:
      width: [18, 36, 48, 64, 80]
      norm: batch              # Paper uses BN (batch_size=1 → effectively instance norm)
      activation: elu
      depth_2d: 1
      kernel_2d: [1, 3, 3]
      down_factors: [[1,2,2], [1,2,2], [1,2,2], [1,2,2]]
    input_size: [18, 160, 160]
    output_size: [18, 160, 160]
    # Loss handled by affinity-12 pipeline profile → loss_per_channel → PerChannelBCEWithLogitsLoss
```

### Phase 3: Match Augmentation ✅

All three items implemented and tested (32/32 augmentation tests pass).

1. **Mutual exclusion of defect augmentations** — Added `defect_mutex: bool` to `AugmentationConfig`. When `true`, misalignment/missing_section/motion_blur are wrapped in MONAI `OneOf` so only one fires per sample (matching DeepEM `Blend(mutex)`). Backward-compatible default (`false`).
   - `connectomics/config/schema/data.py`: added `defect_mutex` field
   - `connectomics/data/augment/build.py`: collects defect transforms into `OneOf` when enabled
   - `tests/unit/test_em_augmentations.py`: 2 new tests for mutex on/off

2. **Elastic deformation enabled** — `prob=0.7`, `sigma_range=[4.0, 8.0]`, `magnitude_range=[8.0, 16.0]` (matches `aug_em_neuron` profile, tuned for anisotropic EM).

3. **Contrast/brightness ±50%** — `contrast_range=[0.5, 1.5]`, `shift_intensity_offset=0.2` (matches DeepEM `MixedGrayscale2D`).

All settings live in the `aug_em_neuron` profile (`tutorials/bases/augmentation_profiles.yaml`),
which is applied automatically via the `affinity-12` pipeline profile in `tutorials/bases/pipeline_profiles.yaml`.
No inline augmentation overrides are needed in `neuron_snemi.yaml`.

### Phase 4: Inference Improvements

- Enable TTA (16 variants: z-flip, y-flip, x-flip, xy-transpose)
- Consider bump function blending
- Consider mean affinity agglomeration post-processing

## Discrepancy Summary Table

| # | Category | Discrepancy | Impact | Status |
|---|---|---|---|---|
| 1 | Training duration | 100K vs 500-700K iters | **CRITICAL** | Config change needed |
| 2 | Optimizer | AdamW(lr=3e-4, eps=1e-8) vs Adam(lr=0.01, eps=0.01) | **CRITICAL** | Config change needed |
| 3 | LR schedule | Cosine vs plateau halving | **CRITICAL** | Config change needed |
| 4 | Architecture | 4 vs 5 levels, missing 18-ch level | **HIGH** | **Fixed** ✅ |
| 5 | Patch size | 16x224x224 vs 18x160x160 | **HIGH** | Config change needed |
| 6 | Loss function | BCE+Dice vs BCE-only, per-channel balancing | **MODERATE** | **Fixed** ✅ |
| 7 | Border masking | Uniform crop vs per-channel `get_pair` mask | **HIGH** | **Fixed** ✅ |
| 8 | Augmentation | Independent vs mutually exclusive defects | **MODERATE** | Code change needed |
| 9 | Batch size | 4 vs 1 | **MODERATE** | Config change needed |
| 10 | Blending | MONAI Gaussian vs bump function | **LOW** | Code change needed |

## Notes

- The DeepEM paper trained on Caffe, not PyTorch. Minor numerical differences in BN, convolution, and optimizer implementations may exist but should be negligible.
- GroupNorm (PyTC) vs BatchNorm (paper) may actually favor PyTC at batch_size=1, since BN statistics are noisy with single samples. Consider keeping GroupNorm.
- The paper's ε=0.01 for Adam is unusually high. This effectively reduces the adaptive nature of Adam, preventing very large per-parameter step sizes. This may be important for training stability at lr=0.01.
- The paper splits training data 80/20 (80 slices train, 20 val). PyTC currently trains on the full training volume without a val split, which means the ReduceLROnPlateau schedule would need a validation set to monitor.
