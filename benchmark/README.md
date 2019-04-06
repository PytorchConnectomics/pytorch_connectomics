# Benchmark Results in Connectomics

## SNEMI 3D

| Augmentation  | Unet-v0    | Unet-v1    | Unet-v2    | Unet-v3    | FPN        |
| ------------- | ---------  | ---------- | ---------- | ---------- | ---------- |
|               | 28x160x160 | 12x160x160 | 12x160x160 | 12x160x160 | 12x160x160 |
| Baseline      |            |            |            |            |            |
| Augment 1     |            |            |            |            |            |
| Augment 2     |            |            |            |            |            |

**Baseline:** no augmentation

**Augment 1:** rotation, flip, brightness, contrast, invert, wraping

**Augment 2:** missing section, misalignment, out-of-focus section