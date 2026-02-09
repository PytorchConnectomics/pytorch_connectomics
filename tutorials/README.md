# Tutorial Configs

Top-level tutorial configs are in this folder and are intended to be runnable with:

```bash
python scripts/main.py --config tutorials/<config>.yaml
```

## Active top-level configs

- `tutorials/mito_lucchi++.yaml`: Lucchi++ mitochondria segmentation (MONAI UNet).
- `tutorials/mito_mitoEM.yaml`: MitoEM mitochondria instance segmentation (MedNeXt, SDT).
- `tutorials/mito_mitolab.yaml`: CEM-MitoLab 2D mitochondria segmentation (MedNeXt).
- `tutorials/mito_betaseg.yaml`: BetaSeg mitochondria instance segmentation (MedNeXt, affinity+SDT).
- `tutorials/neuron_snemi.yaml`: SNEMI3D neuron segmentation (RSUNet, affinities).
- `tutorials/nuc_nucmm-z.yaml`: NucMM zebrafish nuclei segmentation (MONAI UNet, multi-task).
- `tutorials/fiber_linghu26.yaml`: Fiber segmentation (MedNeXt, binary+boundary+distance).

## Config composition (`_base_`)

Top-level configs now use inheritance via `_base_`:

- `tutorials/bases/common.yaml`: Shared defaults across top-level tutorials.
- `tutorials/bases/mednext.yaml`: MedNeXt-specific shared settings.
- `tutorials/bases/monai_unet.yaml`: MONAI UNet shared settings.
- `tutorials/bases/rsunet.yaml`: RSUNet shared settings.

`_base_` supports:

- A single file path (`_base_: bases/common.yaml`)
- A list of files (`_base_: [a.yaml, b.yaml]`) with left-to-right merge order
- Relative paths resolved from the current config file

## Validation

Validate top-level tutorial configs:

```bash
python scripts/validate_tutorial_configs.py
```

This check fails if a config cannot load or if legacy keys reappear (`inference.data`, `data.augmentation.enabled`, or `inference.test_time_augmentation.act`).
