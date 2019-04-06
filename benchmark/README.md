# Benchmark Results in Connectomics

## SNEMI 3D

| Augmentation  | Unet-v0    | Unet-v1    | Unet-v2    | Unet-v3    | FPN        |
| ------------- | ---------  | ---------- | ---------- | ---------- | ---------- |
| Baseline      |            |            |            |            |            |
| Augment 1     |            |            |            |            |            |
| Augment 2     |            |            |            |            |            |

We use a mini-batch size of 8 with patches of size (18,160,160). The initial learning rate is 0.001.

**Baseline:** no augmentation

**Augment 1:** rotation, flip, brightness, contrast, invert, wraping

**Augment 2:** missing section, misalignment, out-of-focus section (+ **Augment 1**)
