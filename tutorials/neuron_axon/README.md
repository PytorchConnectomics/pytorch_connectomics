# Axon decoding (MIT-LiCONN DL288B)

Decode the same 3-channel CZYX affinity volume two ways and score both with NERL:

- **`waterz_baseline.yaml`** — the fixed `naive_waterz` reference recipe (`decode_waterz`
  in 80-slice chunks with the validated border stitching).
- **`axon_decode.yaml`** — the staged axon decoder: volume-unique 2D sections → conservative
  tracklet linking → false-merge splitting → completion/merge/weak-gap bridging.

The premise is that in connectomics **a split is cheaper than a merge** (a split is fixable by
proofreading; a merge corrupts two neurons), so the pipeline first converts false merges into
splits — raising the false-merge-free ceiling — and then re-links the splits back up under it.

Both YAMLs are **self-contained** — no `_base_` include — so each file is the complete,
reproducible description of one run from affinity to final segmentation. Every decode
parameter is spelled out at its validated default with the measurement behind it, so tuning
means editing a value rather than reading the source.

## Run

```bash
python scripts/main.py --config tutorials/neuron_axon/waterz_baseline.yaml --mode test
python scripts/main.py --config tutorials/neuron_axon/axon_decode.yaml     --mode test
```

Decode-only: no checkpoint and no GPU. On 8 CPU cores the staged decode takes ~8 min and
~26 GB; the baseline ~14 min. Set `num_workers` on the `sections` node (`-1` = every
available CPU; it reads the cgroup mask, so it honours `sbatch -c N`).

## Inputs

| path | what |
|---|---|
| `datasets/mit-liconn/raw_x1_head-aff_r1.h5` | 3-channel CZYX affinity (z,y,x), float32, 800×1024×1024 at 25×9×9 nm |
| `datasets/mit-liconn/gt_label_clean_v3.h5` | proofread GT, 943 axons (iteratively audited — the original release over-split axons) |

**No skeleton file is needed.** The ERL graph is derived from the label volume on first use
(kimimaro at a (2,4,4) downsample, anisotropy scaled to match so vertex coordinates stay in
the full-resolution frame) and cached as `gt_label_clean_v3.erlgraph.npz` beside it: 909
skeletons, 98,260 nodes, 5686 µm.

Scoring is optional — delete the `evaluation:` and `test:` blocks to decode without ground
truth, and the run writes only the segmentation.

## Results

Full 800×1024×1024 volume, `nerl_merge_threshold: 10`, scored against the derived graph.
`om` is oracle-merge: every predicted fragment relabelled to its majority-overlap GT, i.e.
the ceiling that remains once all false merges are excluded.

| decode output | segments | NERL base | NERL om |
|---|---:|---:|---:|
| naive waterz | 1,223 | 0.6143 | 0.7130 |
| tracklets (`output: tracklets`) | 27,836 | 0.7773 | 0.8673 |
| split (`output: split`) | 28,147 | 0.6857 | 0.8856 |
| **merged (`output: merged`, default)** | 22,071 | **0.7914** | **0.8759** |

Read the middle two rows together: the split stage *lowers* base NERL (0.7773 → 0.6857) and
raises the ceiling (0.8673 → 0.8856). It is buying headroom, and the merge stage converts
that headroom back into run length — ending above the tracklet base on both counts. Judge a
split stage by om, and a merge stage by the base it recovers while holding om. (Merging can
only lower om, which is why the final 0.8759 sits just under the split's 0.8856.)

## Stopping early

Change one line — `graph.output` in `axon_decode.yaml` — to `sections`, `tracklets` or
`split`. The graph prunes execution to that node's ancestors, and the artifact cache tag
changes with it, so an early-stopped run cannot silently reuse a later artifact.

## Tuning

Each graph node lists its kwargs at the validated default. The two that move the result most:

- **`merge_iou`** (0.45) — cross-section IoU needed to consider a partner at a z-seam.
  Selecting by IoU rather than affinity is what made the merge stage work; the seam affinity
  (`aff_lo`) is only a background floor and must never rank partners.
- **`margin`** (0.15) — the best partner must beat the runner-up by this gap, or the pair is
  left split. Raise it to be more conservative.

`small: 0` on the sections node is load-bearing: waterz's own default of 150 silently removed
4.2% of confident skeleton coverage, whole thin tubes included.

## Why `nerl_merge_threshold: 10`

ERL is brutally sensitive to contamination — at `merge_threshold: 1`, as few as two foreign
nodes count a segment as merged. Sweeping it on the split stage:

| merge_threshold | base | om |
|---:|---:|---:|
| 1 | 0.6641 | 0.8301 |
| 2 | 0.6699 | 0.8464 |
| 5 | 0.6792 | 0.8687 |
| **10** | **0.6857** | **0.8856** |

The ceiling moves 0.055 across that range while the decode does not change at all, so a
number quoted without its threshold is not comparable. 10 is the fair yardstick here; retune
it for a dataset with different skeleton density.

## Adapting to another dataset

Point `decoding.load_prediction_path` at your affinity volume and `data.test.{label,
resolution}` at your GT. The gates were tuned on 25×9×9 nm anisotropic axons: the 2D-section
seed assumes the in-plane resolution resolves a cross-section while the z-step may not, and
the splitter's caliber/collinearity gates assume tube-like objects. On isotropic or
non-tubular data, expect to retune rather than reuse.

The optional completion-radius link (`prefer_length: true`) is **off** by default. It does
reach the lateral-drift gaps that shape matching structurally cannot, but every configuration
tested lost oracle-merge ceiling: the correct links were not separable from the false ones by
proximity, caliber and trajectory alone. Enable it only when run length matters more than
merge safety.

## Provenance

Algorithm and measurements: `dev/mit_liconn/PIPELINE.md`. The packaged ops in
`connectomics/decoding/decoders/branch/` reproduce that research pipeline **voxel-exactly**
(0 differing voxels at both the baseline and the final stage), and their module docstrings
carry the CUE LADDERs — which signals were measured to be trustworthy for deciding "same
neuron?" and which were measured to be useless.
