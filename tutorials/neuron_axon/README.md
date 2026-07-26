# Axon decoding (MIT-LiCONN DL288B)

Decode the same 3-channel CZYX affinity volume two ways and score both with NERL:

- **`waterz_baseline.yaml`** — the fixed `naive_waterz` reference recipe (`decode_waterz`
  in 80-slice chunks with the validated border stitching).
- **`axon_decode.yaml`** — the staged axon decoder: volume-unique 2D sections → conservative
  tracklet linking → false-merge splitting → completion/merge/weak-gap bridging.

The premise is that in connectomics **a split is cheaper than a merge** (a split is fixable by
proofreading; a merge corrupts two neurons), so the pipeline first converts false merges into
splits — raising the false-merge-free ceiling — and then re-links the splits back up under it.

## Run

```bash
python scripts/main.py --config tutorials/neuron_axon/waterz_baseline.yaml --mode test
python scripts/main.py --config tutorials/neuron_axon/axon_decode.yaml     --mode test
```

Both read `datasets/mit-liconn/raw_x1_head-aff_r1.h5`, score against the same 943-label
proofread GT and strong-foreground ERL graph, and report **base NERL** plus **oracle-merge
NERL** (every predicted fragment relabelled to its majority-overlap GT — the attainable
split-recovery ceiling, i.e. what is left once all false merges are excluded).

Neither YAML exposes tuning knobs: the validated constants and the gate order live with the
operations, not in config.

## Results

Full 800×1024×1024 volume, `nerl_merge_threshold: 10`, 943-label GT:

| decode output | NERL base | NERL oracle-merge |
|---|---:|---:|
| naive waterz | 0.6530 | 0.7580 |
| tracklets (`output: tracklets`) | 0.8284 | 0.9424 |
| split (`output: split`) | 0.7302 | 0.9631 |
| **merged (`output: merged`, default)** | **0.8434** | **0.9525** |

The split stage intentionally *lowers* base NERL — it is buying ceiling (om 0.9424 → 0.9631)
that the merge stage then converts back into run length. Judge a split stage by om, and a
merge stage by the base it recovers while holding om.

Wall time on one 8-core node: ~14 min for the baseline, ~41 min for the staged decode (~7 min
of which is the 800-slice 2D watershed).

## Stopping early

Change one line — `graph.output` in `axon_decode.yaml` — to `sections`, `tracklets` or
`split`. The graph prunes execution to that node's ancestors, and the artifact cache tag
changes with it, so an early-stopped run cannot silently reuse a later artifact.

```yaml
    graph:
      nodes:
        - {name: sections,  op: seg_2d,       inputs: [raw]}
        - {name: tracklets, op: branch_link,  inputs: [raw, sections]}
        - {name: split,     op: branch_split, inputs: [raw, tracklets]}
        - {name: merged,    op: branch_merge, inputs: [raw, split]}
      output: merged        # <- sections | tracklets | split | merged
```

## Why `nerl_merge_threshold: 10`

ERL is brutally sensitive to contamination: at `merge_threshold: 1`, as few as **two** foreign
nodes count a segment as merged and halve a long neuron's ERL. Sweeping it on the split stage:
1 → om 0.9037, 2 → 0.9203, 5 → 0.9447, **10 → 0.9631**. Roughly 90% of the apparent oracle gap
at thr=1 was that artifact, so 10 is the fair yardstick here. Retune it for a dataset with
different skeleton density.

## Adapting to another dataset

Point `decoding.load_prediction_path` at your affinity volume and `data.test.{label, skeleton,
resolution}` at your GT. The gates were tuned on 25×9×9 nm anisotropic axons: the 2D-section
seed assumes the in-plane resolution resolves a cross-section while the z-step may not, and the
splitter's caliber/collinearity gates assume tube-like objects. On isotropic or non-tubular
data, expect to retune rather than reuse.

The optional completion-radius link (`branch_merge(..., prefer_length=True)`) is **off** by
default. It does reach the lateral-drift gaps that shape matching structurally cannot, but
every configuration tested lost oracle-merge ceiling: the correct links were not separable from
the false ones by proximity, caliber and trajectory alone. Enable it only when run length
matters more than merge safety.

## Provenance

Algorithm and measurements: `dev/mit_liconn/PIPELINE.md`. The packaged ops in
`connectomics/decoding/decoders/branch/` reproduce that research pipeline **voxel-exactly**
(0 differing voxels at both the baseline and the final stage), and their module docstrings
carry the CUE LADDERs — which signals were measured to be trustworthy for deciding "same
neuron?" and which were measured to be useless.
