# Low-res MICrONS segmentation at 256 x 256 x 320 nm

BANIS+ trained on downsampled MICrONS (phase 1 and phase 2) images against the public automatic
segmentation on the same grid. At this resolution a thin neurite is smaller than a voxel, so the
learnable target is the soma, the nucleus region and the thick proximal dendrites; the use is
telling cell types (neuron vs glia, excitatory vs inhibitory) from a cheap whole-volume pass.

| YAML | Deep learning | Decoding |
|---|---|---|
| `overfit.yaml` | `banis+.yaml` for 5k steps on one aligned crop per volume, tested on the same crops | ABISS ws |
| `banis+.yaml` | BANIS+ (MedNeXt-S/k3, 6-channel ABISS-layout affinity, class-balanced BCE, EMA, erosion 1, 128 cube, 200k steps) with the MedNeXt speed knobs | ABISS ws |

## Results

Pending: nothing has been trained. Run `overfit.yaml` first; train `banis+.yaml` only once the
overfit crops decode back to their labels.

## Data

The volumes are not public downloads in this layout. They are cut by the research workbench job
`projects/seg120nm-2611/scripts/valid_region.py`:

- `align` finds the integer offset between each downsampled image and its segmentation (FFT
  cross-correlation of a label "interior" map with image maps, checked block by block and, where
  a nucleus label volume exists, by nuclei lying inside one segment), estimates the region where
  both are non-zero, and writes `overfit/<volume>_img.h5`, `_seg.h5` (160 x 640 x 640) and an
  overlay PNG.
- `export` cuts 256 x 512 x 512 chunks from that region into `train/` and `tune/` (the highest-x
  column of chunks, one chunk column away from `train/`).

Images are uint8, labels uint32 relabelled per chunk with 0 as background. Point the three
`path:` entries of `dataset.yaml` (or a run-local config that inherits it) at the folders.

## Recipe

`banis+.yaml` inherits `../neuron_segneuron/banis+.yaml`, so the affinities are in ABISS's layout
(channels x, y, z; edge (i-1, i) stored at voxel i) and go to `decode_abiss` unchanged. The
changes are in its header: MedNeXt-S, the 128 cube with 10-voxel long-range offsets on every axis
(near-isotropic voxels), label erosion 1, and `aug_light` in place of section-defect
augmentation sized for 4-8 nm pixels.

## Overfit check

```bash
python scripts/main.py --config tutorials/neuron_seg256nm/overfit.yaml
python scripts/main.py --config tutorials/neuron_seg256nm/overfit.yaml --mode test \
    --checkpoint <run>/checkpoints/last.ckpt
python scripts/evaluate_emneuron.py --abiss-home /path/to/abiss --volume <volume> \
    --pred <the affinity file the test run wrote for that volume> --gt /path/to/seg256nm/overfit/<volume>_seg.h5 \
    --out overfit_scores.tsv
```

A pair that is mis-registered, a wrong affinity layout or a broken target shows up as a training
loss that will not fall or as predicted boundaries offset from the labels. The score is against
the uneroded labels, so it is bounded by what erosion 1 leaves: read it next to the same decode
of the ground-truth affinities, not against zero.
