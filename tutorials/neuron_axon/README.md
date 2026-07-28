# Axon decoding

Decode any 3-channel CZYX affinity volume containing axons two ways, then evaluate
the result with NERL or without ground truth:

- **`waterz_baseline.yaml`** — the fixed `naive_waterz` reference recipe (`decode_waterz`
  in 80-slice chunks with border stitching).
- **`axon_decode.yaml`** — the staged axon decoder: volume-unique 2D sections → conservative
  tracklet linking → false-merge splitting → completion/merge/weak-gap bridging.
- **`tube_analysis.yaml`** — the staged decoder plus GT-free tube diagnostics: how many
  instances are long, complete, geometrically single, parallel, disconnected, or locally
  enlarged.

The premise is that in connectomics **a split is cheaper than a merge** (a split is fixable by
proofreading; a merge corrupts two neurons), so the pipeline first converts false merges into
splits — raising the false-merge-free ceiling — and then re-links the splits back up under it.

The two decoder YAMLs are **self-contained** — no `_base_` include — so each is the complete
description of one run from affinity to final segmentation. Every decode parameter is
spelled out at its starting default.
`tube_analysis.yaml` inherits `axon_decode.yaml`, changes only the output path and evaluation
block, and explicitly clears the GT label.

## Run

First replace the placeholder `decoding.load_prediction_path`. For NERL runs, also replace
`data.test.label` and set `data.test.resolution`; the GT-free tube run needs only the affinity.

```bash
python scripts/main.py --config tutorials/neuron_axon/waterz_baseline.yaml --mode test
python scripts/main.py --config tutorials/neuron_axon/axon_decode.yaml     --mode test
python scripts/main.py --config tutorials/neuron_axon/tube_analysis.yaml   --mode test
```

All three are decode-only: no checkpoint and no GPU. The tube-analysis run also needs no GT
or skeleton graph. Runtime and memory scale with the input volume and number of instances.
Set `num_workers` on the `sections` node (`-1` = every available CPU; it reads the cgroup
mask, so it honours `sbatch -c N`).

## Inputs

| example path | what |
|---|---|
| `path/to/affinity.h5` | 3-channel CZYX affinity ordered as z, y, x |
| `path/to/label.h5` | optional instance-label volume used only by the two NERL runs |

For NERL, **no prebuilt skeleton file is needed.** The graph is derived from the label volume
on first use and cached beside it. Set `data.test.resolution` to the physical voxel size in
z, y, x array-axis order.

Evaluation is optional — delete the `evaluation:` block to write only the segmentation.
Use `tube_analysis.yaml` when no GT is available.

## GT-free tube analysis

`tube_analysis.yaml` requests `evaluation.metrics: [tube]` and sets
`data.test.label: null`. It writes:

- `eval_*.txt` — aggregate counts, count-weighted and volume-weighted fractions, quality
  flags, and the largest incomplete tubes.
- `eval_*_tube_instances.npz` — one row per substantial label, including z span and
  occupancy, border contacts, cross-section area, bump count, parallel-strand evidence,
  3D component counts, and the complete/single/valid flags.

The default denominator is a **decent** tube: at least 20,000 voxels and spanning at least
25% of the volume depth. A tube is **complete** when both of its z-directed terminal
cross-sections reach any relaxed volume face. It is **valid** when complete and neither
persistently multi-stranded nor split into multiple substantial 3D components. Every
threshold is explicit under `evaluation.tube` in the YAML.

These are segmentation diagnostics, not biological truth. A real axon may terminate inside
the crop, and a smooth-looking predicted tube may still be a false end-to-end merge. Compare
decoders on the same crop and thresholds; do not interpret `complete` as proof that an axon
identity is correct.

## Interpreting NERL

`nerl` measures the achieved error-free run length. `nerl_oracle_merge` relabels every
predicted fragment to its majority-overlap GT and estimates the ceiling after false merges
are excluded. Read them together: a conservative split stage can lower base NERL while
raising the oracle-merge ceiling, and a later merge stage should recover run length while
preserving as much of that ceiling as possible.

## Stopping early

Change one line — `graph.output` in `axon_decode.yaml` — to `sections`, `tracklets` or
`split`. The graph prunes execution to that node's ancestors, and the artifact cache tag
changes with it, so an early-stopped run cannot silently reuse a later artifact.

## Tuning

Each graph node lists its kwargs at the starting default. Two important controls are:

- **`merge_iou`** (0.45) — cross-section IoU needed to consider a partner at a z-seam.
  Selecting by IoU rather than affinity is what made the merge stage work; the seam affinity
  (`aff_lo`) is only a background floor and must never rank partners.
- **`margin`** (0.15) — the best partner must beat the runner-up by this gap, or the pair is
  left split. Raise it to be more conservative.

`small: 0` on the sections node avoids silently removing whole thin tubes before linking.

## Why `nerl_merge_threshold: 10`

ERL is sensitive to contamination: a low threshold can classify a small foreign-node graze
as a merge. The decode does not change when this evaluation threshold changes, so scores
quoted with different thresholds are not directly comparable. The YAML uses 10 as a starting
point; retune it for the density of your skeleton graph.

## Using your volume

Point `decoding.load_prediction_path` at your affinity volume. For NERL, also point
`data.test.{label,resolution}` at your GT; for tube analysis, leave `data.test.label: null`
and retune `evaluation.tube` for the crop dimensions and object caliber. The 2D-section seed
assumes the in-plane resolution resolves an axon cross-section, while the splitter's
caliber/collinearity gates assume tube-like objects. Retune the defaults for different voxel
spacing, axon caliber, affinity calibration, or non-tubular data.

The optional completion-radius link (`prefer_length: true`) is **off** by default. It does
reach lateral-drift gaps that shape matching structurally cannot, but can lose oracle-merge
ceiling when correct links are not separable from false ones by proximity, caliber, and
trajectory alone. Enable it only when run length matters more than merge safety.
