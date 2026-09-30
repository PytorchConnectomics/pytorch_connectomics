# Tutorial configs

Run a workflow from the repository root after installing PyTC and obtaining its data:

```bash
python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml
```

| Workflow | Config |
|---|---|
| Synthetic CPU demo | `tutorials/minimal.yaml` (`pytc --demo`) |
| Lucchi++ mitochondria | `tutorials/mito_lucchi++/mito_lucchi++.yaml` |
| MitoEM human, rat, joint | `tutorials/mitoEM/H.yaml`, `tutorials/mitoEM/R.yaml`, `tutorials/mitoEM/HR.yaml` |
| MitoLab mitochondria | `tutorials/mito_mitolab.yaml` |
| BetaSeg mitochondria | `tutorials/mito_betaseg.yaml` |
| SNEMI3D neurons | `tutorials/neuron_snemi/neuron_snemi.yaml` |
| NISB neurons | `tutorials/neuron_nisb/base_banis.yaml`, `tutorials/neuron_nisb/base_banis+.yaml` |
| NucMM nuclei | `tutorials/nuc_nucmm-z.yaml` |
| CREMI synapses | `tutorials/syn_cremi.yaml` |
| Vesicles | `tutorials/vesicle_xm.yaml` |
| External ABISS decoding | `tutorials/basics/decoding_abiss.yaml` |
| waterz decoding | `tutorials/waterz_decoding.yaml` |

Use `python scripts/download_data.py --list` for available public downloads.
Downloader layouts are relative to `datasets/`; replace `/path/to/` placeholders
in other workflows with your local data and external-tool locations.

Shared recipes `tutorials/banis.yaml`, `tutorials/banis+.yaml`,
`tutorials/mitoEM/common.yaml`, and `tutorials/neuron_nisb/dataset.yaml` supply
base settings and are composed by their dataset workflows.

## Config composition

`_base_` accepts a path or a list of paths, resolved relative to the current YAML.
Top-level recipes inherit `connectomics/config/all_profiles.yaml` for the shared
section registries in `connectomics/config/profiles/` and list templates in
`connectomics/config/templates/`. Select section profiles at `*.profile` and
list templates with `template:`. Explicit keys override profile values; explicit
lists replace profile lists.

## Validation

The validator checks every tutorial YAML by default, including shared recipes.
It rejects unknown or removed schema keys and absolute paths outside `/path/to/`.
No workflow is exempt from loading and runtime-coherence checks.

```bash
python scripts/validate_tutorial_configs.py
python scripts/validate_tutorial_configs.py --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'
```

Supplying `--glob` replaces the default patterns.
