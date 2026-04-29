# Tutorial Configs

Top-level tutorial configs are in this folder and are intended to be runnable with:

```bash
python scripts/main.py --config tutorials/<config>.yaml
```

## Active top-level configs

- `tutorials/mito_lucchi++.yaml`: Lucchi++ mitochondria segmentation (MONAI UNet).
- `tutorials/mitoEM/H.yaml`: MitoEM-Human (EM30-H) instance segmentation (MedNeXt, SDT).
- `tutorials/mitoEM/R.yaml`: MitoEM-Rat (EM30-R) instance segmentation (MedNeXt, SDT).
- `tutorials/mitoEM/HR.yaml`: Joint EM30-H + EM30-R training (MedNeXt, SDT).
- `tutorials/mito_mitolab.yaml`: CEM-MitoLab 2D mitochondria segmentation (MedNeXt).
- `tutorials/mito_betaseg.yaml`: BetaSeg mitochondria instance segmentation (MedNeXt, affinity+SDT).
- `tutorials/neuron_snemi.yaml`: SNEMI3D neuron segmentation (RSUNet, affinities).
- `tutorials/nuc_nucmm-z.yaml`: NucMM zebrafish nuclei segmentation (MONAI UNet, multi-task).
- `tutorials/fiber_linghu26.yaml`: Fiber segmentation (MedNeXt, binary+boundary+distance).

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

This check fails if a config cannot load or if legacy keys reappear (`inference.data`, `data.augmentation.enabled`, or `inference.test_time_augmentation.act`).
