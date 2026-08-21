# Decoding Tutorials

## Single-volume ABISS

[`decoding_abiss.yaml`](decoding_abiss.yaml) runs ABISS directly on one saved
affinity volume through `scripts/run_abiss_volume.py`. It produces one HDF5
segmentation and does not use a chunk hierarchy or cross-chunk stitching.

Run it with:

```bash
python scripts/main.py \
  --config tutorials/decoding/decoding_abiss.yaml \
  --mode test
```

In `tutorials/decoding/decoding_abiss.yaml`, change these fields for a different
dataset or experiment:

- `default.decoding.load_prediction_path`: input HDF5 containing a `CZYX` affinity array.
- `default.decoding.save_path`: output directory for the decoded HDF5.
- `steps[0].kwargs.input_dataset`: dataset key inside the input HDF5.
- `steps[0].kwargs.channels`: three nearest-neighbor affinity channels in ABISS `XYZ` order. Reorder them here if the model saved another order.
- `steps[0].kwargs.command`: single-volume ABISS command. Keep the input/output placeholders. Add `--abiss-home /absolute/path/to/abiss` when ABISS is not under this repository's `lib/abiss`.
- `cli_args.ws_high_threshold`: confident watershed seed threshold.
- `cli_args.ws_low_threshold`: lower affinity accepted while growing the watershed. This is the ABISS parameter corresponding to `aff_low`.
- `cli_args.ws_size_threshold`: size threshold used during watershed merging.
- `cli_args.ws_dust_threshold`: fragments below this size are removed.
- `cli_args.ws_merge_threshold`: affinity cutoff for size-based region merging. Increase it to merge more conservatively.
- `test.data.test.name`: stable volume name used in output filenames.
- `test.data.test.resolution`: voxel size in `ZYX` array-axis order.
- `test.data.test.label`: optional ground-truth instance HDF5; leave it `null` for decode-only use.

For volumes that must be divided and stitched across multiple logical chunks,
use the separate multi-chunk ABISS workflow instead.

## Postprocessing: shape smoothing

The second decode step, `shape_smooth`, cleans up the ABISS output using
geometry alone — it never reads the affinity back. It has three parts, and they
are not equally useful:

1. **`fastmorph` label opening** — erode then dilate each label, which removes
   hairline necks and one-voxel protrusions. Label identity never moves, and the
   result is clamped to the original support so the opening cannot *add* voxels.
2. **`cc3d` relabel** — a neck that the opening removed becomes two instances
   here. This is where the actual split happens; the opening only makes it
   possible.
3. **Cross-z outlier split** (`split`, off by default) — where a label's
   per-slice area *steps* up, carve the extra region out with a two-marker
   distance watershed: the eroded previous cross-section seeds "the tube", the
   part it does not explain seeds "what joined it". The kept part seeds the next
   slice, so the carve tracks the tube instead of drifting.

### 2D or 3D opening

`open_plane: 2d` opens each z-slice independently; `3d` uses a 3×3×3 element.

Prefer **2d when labels are thin and densely packed**. A 3D erosion attacks every
label–label interface, not just necks, and at such an interface there is no
background to regrow from — so a z-thin label is deleted outright rather than
merely thinned. In the unit tests a label wedged between two others in z goes
from 288 voxels to 0 under a 3D opening and survives intact under 2D.

That failure is not hypothetical. Run on a fine 517-label decode of this same
volume, the 3D opening removed ~11% of the foreground and drove COMPLETE from
71.1% down to 61.3%: of the eroded voxels, 326k could only be reclaimed by a
*different* label, meaning their own label had no surviving seed anywhere.

### What the parts are worth

Measured on this volume with the GT-free tube report
(`connectomics/metrics/tube.py`), starting from 139 ABISS labels:

| variant | labels | decent | COMPLETE | VALID | disconnected | foreground |
|---|---|---|---|---|---|---|
| baseline | 139 | 38 | 26 (68.4%) | 15 (39.5%) | 0 | — |
| open 2d + cc3d | 269 | 39 | 28 (71.8%) | 17 (43.6%) | 0 | −1.7% |
| open 3d + cc3d | 241 | 46 | 33 (71.7%) | 20 (43.5%) | 0 | −3.4% |
| open 3d + cc3d + split | 253 | 46 | 33 (71.7%) | 20 (43.5%) | 4 | −3.4% |

The opening plus relabel is what earns the gain. The carve added no completeness
on this volume and raised `disconnected` from 0 to 4, because a carved id can end
up as two 3D pieces — hence `split: false` in the shipped config. Turn it on when
you can see sustained cross-section jumps that the opening does not resolve, and
check `disconnected` afterwards.

Two knobs that are traps rather than tuning parameters:

- `open_radius` selects a spherical element instead of the cube. Radii below
  `2.0` erode **zero** voxels on the voxel lattice, making the whole stage a
  silent no-op, so values in `(0, 2)` raise instead of being accepted.
- `anchor_border` refuses to carve a cross-section sitting on a volume z-face.
  Such a cross-section is truncated by the face, so the area step beside it is a
  boundary artifact. Without it, 8 of 15 carves started at `z1` and one ran the
  full 63 slices.
