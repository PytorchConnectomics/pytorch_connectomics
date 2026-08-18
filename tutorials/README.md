# Tutorial Configs

Tutorial configs are in this tree and are intended to be runnable with:

```bash
python scripts/main.py --config tutorials/<config>.yaml
```

## Active configs

- `tutorials/mito_lucchi++/mito_lucchi++.yaml`: Lucchi++ semantic mitochondria
  segmentation (MedNeXt-S).
- `tutorials/mitoEM/H.yaml`: MitoEM-Human (EM30-H) instance segmentation (MedNeXt, SDT).
- `tutorials/mitoEM/R.yaml`: MitoEM-Rat (EM30-R) instance segmentation (MedNeXt, SDT).
- `tutorials/mitoEM/HR.yaml`: Joint EM30-H + EM30-R training (MedNeXt, SDT).
- `tutorials/mito_mitolab.yaml`: CEM-MitoLab 2D mitochondria segmentation (MedNeXt).
- `tutorials/mito_betaseg.yaml`: BetaSeg mitochondria instance segmentation (MedNeXt, affinity+SDT).
- `tutorials/neuron_snemi/neuron_snemi.yaml`: SNEMI3D neuron instance segmentation
  (MedNeXt-S, 12-channel affinity, waterz).
- `tutorials/nuc_nucmm-z.yaml`: NucMM zebrafish nuclei segmentation (MONAI UNet, multi-task).
- `tutorials/fiber_linghu26.yaml`: Fiber segmentation (MedNeXt, binary+boundary+distance).

## Shared recipes (not runnable on their own)

- `tutorials/banis.yaml`: BANIS neuron-affinity recipe (MedNeXt-L/k3, 6-channel affinity,
  128-cube, 50k steps) with **no** data paths.
- `tutorials/banis+.yaml`: `banis.yaml` + the ML-ops deltas (per-channel class-balanced BCE,
  EMA, label erosion=2, 200k steps).

Dataset tutorials inherit one of these and add only their own data:
`tutorials/neuron_nisb/base_banis+.yaml` = `banis+.yaml` + `neuron_nisb/dataset.yaml`;
`tutorials/neuron_j0126/infer_affinity.yaml` = `banis+.yaml` + its own zebrafinch block.

- `tutorials/mito_betaseg_base.yaml`: the betaSeg benchmark — data splits, label caching,
  sparse-crop sampling, and the shared watershed decode + Adapted-Rand metric. The four
  `mito_betaseg_banis_{v0,plus,v1,v2}.yaml` recipes inherit it and add only their model
  and schedule; `_plus` is `_v0` plus its documented deltas (MedNeXt-L, PerChannelBCE,
  erosion=2, EMA). These are a separate lineage from `banis.yaml` — 7-channel aff+SDT
  multi-head, not 6-channel affinity — so they do not share it.

## Config composition (`_base_`)

Top-level configs now use inheritance via `_base_`:

- `connectomics/config/all_profiles.yaml`: Canonical registry index loaded by top-level tutorials.
- `connectomics/config/profiles/*.yaml`: Section-level registries selected by `*.profile`.
- `connectomics/config/templates/*.yaml`: Explicit list-item templates, currently used for top-level `decoding`.

`_base_` supports:

- A single file path (`_base_: ../connectomics/config/all_profiles.yaml`)
- A list of files (`_base_: [a.yaml, b.yaml]`) with left-to-right merge order
- Relative paths resolved from the current config file

Merge semantics:

- Profile payloads are merged into the destination section first.
- Explicit keys in the tutorial override profile keys.
- Explicit lists replace profile lists; they are not additive.
- Canonical decoding syntax is explicit list templating: `- template: decoding_waterz`.

## Validation

Validate top-level tutorial configs:

```bash
python scripts/validate_tutorial_configs.py
```

Include nested tutorial families, including Lucchi++:

```bash
python scripts/validate_tutorial_configs.py --glob 'tutorials/**/*.yaml'
```

This check fails if a config cannot load or if legacy keys reappear (`inference.data`, `data.augmentation.enabled`, or `inference.test_time_augmentation.act`).
