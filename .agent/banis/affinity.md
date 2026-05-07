# Affinity Modes

Affinity targets must declare one explicit convention:

```yaml
data:
  label_transform:
    targets:
      - name: affinity
        kwargs:
          offsets: ["0-0-1", "0-1-0", "1-0-0"]
          affinity_mode: deepem  # or banis
```

There is no legacy crop flag. The mode controls target voxel placement,
valid-border masking, visualization crop, and test-time prediction crop.

## Modes

| Mode | Edge storage | Valid side for positive offsets | Intended use |
| --- | --- | --- | --- |
| `deepem` | Destination voxel `v + offset` | Leading border invalid | DeepEM/SNEMI-style targets, zwatershed/ABISS destination-index affinity |
| `banis` | Source voxel `v` | Trailing border invalid | BANIS-compatible targets and source-index connected-component decoding |

For a positive x offset `0-0-1`, `deepem` produces valid affinities at
`x >= 1`; `banis` produces valid affinities at `x < W - 1`.

## Training

Training does not crop every affinity channel to the largest common valid
interior. It keeps prediction and target shapes unchanged and applies a
per-channel valid mask before loss evaluation. This preserves short-range edge
supervision while excluding convention-dependent padded borders.

Mixed affinity modes in one stacked label tensor are rejected. If multiple
affinity target groups are ever needed, they must share the same
`affinity_mode`.

## Inference And Decoding

Test-time affinity crops use the same mode as training:

```python
compute_affinity_crop_pad(offsets, affinity_mode="deepem")
compute_affinity_crop_pad(offsets, affinity_mode="banis")
```

The crop is resolved after `inference.select_channel` and output-head target
slices. If decoding keeps only short-range channels from a larger affinity
target stack, the automatic affinity crop must use only those selected offsets.

`decode_affinity_cc` has an independent `edge_offset` knob for the numba
backend:

| Target mode | `decode_affinity_cc.kwargs.edge_offset` |
| --- | --- |
| `deepem` | `1` |
| `banis` | `0` |

The `cc3d` backend ignores directed edge placement and only thresholds
foreground connectivity, so this matters mainly for `backend: numba`.

## Config Policy

Use `affinity_mode: deepem` for DeepEM/SNEMI/LiConn-style configs.

Use `affinity_mode: banis` for BANIS/NISB reproduction configs and any config
whose target should match `lib/banis/data.py::comp_affinities`.
