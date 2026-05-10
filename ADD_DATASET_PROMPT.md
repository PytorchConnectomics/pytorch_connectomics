# PyTorch Connectomics — Agent Add-Dataset Prompt

You are adding a new EM volume / dataset to this repo. Follow the
steps below; only edit the files explicitly listed under "Files to
touch".

## Inputs to ask the user for first

- New dataset short name (e.g. `mito_my_volume`).
- Train image volume path (HDF5 / TIFF / Zarr).
- Train label volume path.
- Optional: validation image / label paths.
- Task type: semantic segmentation or instance segmentation.

If any required input is missing, stop and ask.

## Steps

1. Pick the closest existing tutorial as your template:
   `tutorials/mito_lucchi++.yaml` for semantic mito,
   `tutorials/syn_cremi.yaml` for synapse,
   `tutorials/neuron_snemi/*.yaml` for instance neuron,
   `tutorials/nuc_nucmm-z.yaml` for nuclei.

2. `cp tutorials/<closest>.yaml tutorials/<new>.yaml`.

3. Edit only the **stage-specific** data paths inside the new YAML.
   Canonical tutorials nest data under `train:` / `test:` / `tune:`,
   e.g. `train.data.train.{image,label}` and
   `train.data.val.{image,label}`. A top-level `data:` block does
   *not* override these. Do not invent new keys.

4. Validate:
   `python scripts/validate_tutorial_configs.py --glob 'tutorials/<new>.yaml'`
   (`--glob` is additive over default `tutorials/*.yaml`; only treat
   errors that mention the new path as yours; otherwise stop).

5. Smoke-train one batch:
   `python scripts/main.py --config tutorials/<new>.yaml --fast-dev-run`.
   Non-zero exit = stop and report.

## Verification

- `python scripts/main.py --config tutorials/<new>.yaml --fast-dev-run`
  exits 0. (If the YAML keys land in the wrong place, this fails
  with `FileNotFoundError` pointing at the default path, not yours —
  that's the actual signal that the edit took effect.)

## Stop conditions

- Required input missing → stop and ask.
- New file format (not HDF5/TIFF/Zarr) → stop; do not edit
  `connectomics/data/io/io.py` without the user's say-so.
- Strict-key validation error → stop and report; do not invent
  schema fields.
- Same fast-dev-run failure twice → stop and report.

## Files to touch

- `tutorials/<new>.yaml` only.
