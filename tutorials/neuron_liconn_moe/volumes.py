#!/usr/bin/env python3
"""The eight ExPID96 "moe" volumes and how each is put on the training grid.

Single source of truth for the batch: `prepare_volume.py` invocations,
per-volume config generation, the ABISS sweep, and the neuroglancer upload all
read this. Run it directly to print the table.

THE ONE DECISION ENCODED HERE. The banis+ checkpoint was trained at
[24, 18, 18] nm ZYX (biological nm -- moe spacings already have the expansion
factor divided out, so matching nm matches neurite caliber in voxels). The eight
volumes sit at four expansions and none is a native match:

    18x  [22.22, 9.03, 9.03]   ->  factor to [24,18,18] = [1.08, 1.99, 1.99]
    22x  [18.18, 7.39, 7.39]   ->                         [1.32, 2.44, 2.44]
    28x  [14.29, 5.80, 5.80]   ->                         [1.68, 3.10, 3.10]
    32x  [12.50, 5.08, 5.08]   ->                         [1.92, 3.54, 3.54]

The 18x pair takes `factor` -- an exact (1,2,2) block average landing on
[22.22, 18.06, 18.06], within 0.3% of the training XY and 7.4% of its Z, with no
interpolation at all. That is the recipe validated on
`ExPID96_2ndgel_S1_40XW001_18x` (affinity QC + merge-threshold sweep + published
layer), so the second 18x volume inherits it unchanged.

The other six take `target`: an area-average resample to [24, 18, 18] exactly.
Interpolating Z by 1.32-1.92 is the price; the alternative -- rounding to the
nearest integer factor -- would leave XY 18-26% off the training scale, and the
whole reason this workflow resamples at all is that the model is a conv net for
which voxel size is not a free parameter. Their actual output spacing is
`n*s/round(n/f)`, within ~0.1% of [24,18,18]; `prepare_volume.py` records it.

There is NO GROUND TRUTH for any moe volume, so nothing below is validated
against labels. See README.md.
"""

from __future__ import annotations

from pathlib import Path

MOE_ROOT = Path("/projects/weilab/dataset/liconn/moe/preprocessed/clip_percentile_1_99")
SRC_ZARR = MOE_ROOT / "zarr"
PREPARED = MOE_ROOT / "prepared_train_grid"

TRAIN_GRID_ZYX = (24.0, 18.0, 18.0)

# The IST-LICONN banis+ 200k checkpoint applied cross-sample to every moe volume.
REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
CKPT = REPO / "outputs/liconn_final_banis_plus_tube/20260728_032436/checkpoints/step=00200000.ckpt"
# In --mode test `runtime/checkpoint_dispatch.py` overwrites `inference.save_path`
# with <ckpt run dir>/test_<ckpt stem>/, so this is where results actually land --
# the `save_path:` in the step YAMLs is ignored. The per-volume leaf is the stem
# of the image path, which is why `prepare_volume.py` writes `<vol>.h5` and not
# `<vol>.zarr/0` (every zarr volume would share the leaf "0" and overwrite).
TEST_OUT = CKPT.parent.parent / f"test_{CKPT.stem}"
OUT_ROOT = REPO / "outputs/neuron_liconn_moe"

# Publication target: the PUBLIC bucket, which already mirrors all eight OME-Zarr
# image groups (and the first segmentation) under the same prefix, so layers land
# next to the images they overlay.
#
# The `clip_percentile_1_99` component is load-bearing and must not be dropped: a
# second clip variant with IDENTICAL dataset names exists at
# `preprocessed/zarr/`, so a flat `liconn/moe/` would collide the moment the
# other variant is published. See [liconn_expid96_moe_preprocessed].
GCS_BUCKET = "donglai_public"
GCS_PREFIX = "liconn/moe/clip_percentile_1_99"
GCS_FOLDER = f"gs://{GCS_BUCKET}/{GCS_PREFIX}"

# ngauth server for the private `donglai` bucket, kept for reference. Whether it
# is authorised for `donglai_public` has not been established.
NGAUTH = f"gs+ngauth+https://sunny-catalyst-506019-a2.ue.r.appspot.com/{GCS_BUCKET}"


def layer_url(layer: str) -> str:
    """Neuroglancer source for a published layer.

    Plain `gs://` assumes the bucket is anonymously readable. As of 2026-09-04 it
    is NOT: an unauthenticated
    `https://storage.googleapis.com/storage/v1/b/donglai_public/o` returns 401,
    the same as the known-private `donglai` bucket, where a public bucket answers
    200. Until `allUsers:objectViewer` is granted, use the `NGAUTH` form instead
    (and confirm that server is authorised for this bucket).
    """
    return f"precomputed://gs://{GCS_BUCKET}/{GCS_PREFIX}/{layer}"


# name -> prep recipe. `factor` = exact block average; `target` = area resample.
#
# APPEND ONLY. The array index of every SLURM step is a position in PENDING, so
# inserting anywhere but the end silently re-points a rerun at a different volume.
VOLUMES: dict[str, dict] = {
    "ExPID96_2ndgel_S1_40XW001_18x": {"factor": (1, 2, 2), "published": True},
    "ExPID96_S1_40XW002_18x": {"factor": (1, 2, 2)},
    "ExPID96_2ndgel_S2_40XW002_22x": {"target": TRAIN_GRID_ZYX},
    "ExPID96_2ndgel_S2_40XW_22x": {"target": TRAIN_GRID_ZYX},
    "ExPID96_2ndgel_S3_40XW004_28xx": {"target": TRAIN_GRID_ZYX},
    "ExPID96_2ndgel_S3_40XW_28x": {"target": TRAIN_GRID_ZYX},
    "ExPID96_2ndgel_S4_40XW002_32x": {"target": TRAIN_GRID_ZYX},
    "ExPID96_2ndgel_S4_40XW003_32x": {"target": TRAIN_GRID_ZYX},
    # Added 2026-09-04, PENDING index 7. Same 32x grid as the S4 pair, so the
    # same `target` resample -- but a DIFFERENT SAMPLE SERIES AND REGION
    # (ExPID99, cerebellum) from the eight ExPID96 fields above. Everything the
    # README says about this batch being out of domain applies here more, not
    # less: the checkpoint saw IST cortical neuropil, and cerebellar cortex has
    # its own morphology (granule-cell packing, parallel fibers). Read its
    # affinity QC before trusting its segmentation at all.
    "ExPID99_32x_2_cerebellum": {"target": TRAIN_GRID_ZYX},
    # Added 2026-09-05, PENDING indices 8-10: the rest of the ExPID99 Cerebellum
    # Drive folder. Same caveat as the row above and then some -- cerebellar
    # cortex, and the checkpoint saw IST cortical neuropil.
    #
    # The two `_18x_2_` volumes are DIFFERENT acquisitions that share one Drive
    # name (ExPID99_18x_2.nd2, 9.9 GB and 11.9 GB); the tail is the Drive file-id
    # prefix, and which is which is unknown. They take the exact (1,2,2) block
    # average like the ExPID96 18x pair -- no interpolation at all.
    "ExPID99_18x_2_cerebellum_1Byqvupl": {"factor": (1, 2, 2)},
    "ExPID99_18x_2_cerebellum_1m2f9z4J": {"factor": (1, 2, 2)},
    "ExPID99_32x_1_cerebellum": {"target": TRAIN_GRID_ZYX},
}

# The volumes this batch runs: everything except the one already on GCS.
PENDING = [k for k, v in VOLUMES.items() if not v.get("published")]


# The first volume was run before the `<vol>.h5` naming fix and its artifacts
# still sit under the NGFF level-index leaf "0", off the `zarr_ds1-2-2` copy.
# Recorded rather than rerun: it is already published, and its numbers are the
# reference the rest of the batch is read against.
LEGACY: dict[str, dict] = {
    "ExPID96_2ndgel_S1_40XW001_18x": {
        "image": MOE_ROOT / "zarr_ds1-2-2/ExPID96_2ndgel_S1_40XW001_18x.zarr/0",
        "merge_threshold": 0.60,
        "sweep_grid": [0.47, 0.55, 0.60, 0.65, 0.70, 0.75],
        # Already on GCS under the pre-percentile naming; do not regenerate.
        "layer": "ExPID96_2ndgel_S1_40XW001_18x_seg_abiss_mt060",
    }
}


def source_zarr(name: str) -> Path:
    return SRC_ZARR / f"{name}.zarr"


def prepared_h5(name: str) -> Path:
    return PREPARED / f"{name}.h5"


def prepared_image(name: str) -> Path:
    """The path handed to the model as `data.test.image`."""
    if name in LEGACY:
        return LEGACY[name]["image"]
    return prepared_h5(name)



def affinity_h5(name: str) -> Path:
    """Step 1 output for `name` (3, Z, Y, X) float16."""
    if name in LEGACY:
        return TEST_OUT / "0/raw_x1_ch0-1-2.h5"
    return TEST_OUT / name / "raw_x1_ch0-1-2.h5"


def work_dir(name: str) -> Path:
    return OUT_ROOT / name


def sweep_dir(name: str) -> Path:
    """Where `sweep_merge_threshold.py` puts its per-threshold segmentations.
    The first volume's sweep predates the per-volume layout."""
    if name in LEGACY:
        return OUT_ROOT / "mt_sweep"
    return work_dir(name) / "mt_sweep"


def layer_name(name: str, merge_threshold: float) -> str:
    """Published layer name. Three decimals, because the threshold is no longer a
    round number -- it is the percentile-matched value for this volume, and two
    digits would round 0.5714 and 0.5749 onto the same layer."""
    if name in LEGACY and "layer" in LEGACY[name]:
        return LEGACY[name]["layer"]
    return f"{name}_seg_abiss_mt{merge_threshold:.3f}".replace(".", "")


def plan(name: str) -> dict:
    """Resolve a volume's prep recipe against the source zarr's own metadata."""
    import numpy as np
    import zarr

    rec = VOLUMES[name]
    g = zarr.open_group(str(source_zarr(name)), mode="r")
    shape = tuple(int(v) for v in g["0"].shape)
    native = np.asarray(g.attrs["spacing_nm_zyx"], dtype=np.float64)

    if "factor" in rec:
        factor = tuple(rec["factor"])
        out_shape = tuple(s // f for s, f in zip(shape, factor))
        spacing = native * np.asarray(factor, dtype=np.float64)
        args = ["--factor", *[str(f) for f in factor]]
    else:
        target = np.asarray(rec["target"], dtype=np.float64)
        out_shape = tuple(max(1, int(round(n / (t / s))))
                          for n, t, s in zip(shape, target, native))
        spacing = native * np.asarray(shape, dtype=np.float64) / np.asarray(out_shape)
        args = ["--target-spacing", *[f"{v:g}" for v in target]]

    return {
        "name": name,
        "source": source_zarr(name),
        "output": prepared_h5(name),
        "native_shape": shape,
        "native_spacing_zyx": native.tolist(),
        "shape": out_shape,
        # ZYX nm of the prepared volume. The neuroglancer layer must use this
        # (reversed to XYZ) or it will not overlay the native image group.
        "spacing_zyx": spacing.tolist(),
        "prepare_args": args,
    }


if __name__ == "__main__":
    print(f"{'volume':34s} {'native shape':>20s} {'-> prepared':>20s} "
          f"{'spacing ZYX nm':>28s}  {'Mvox':>6s}  recipe")
    for n in VOLUMES:
        p = plan(n)
        mv = p["shape"][0] * p["shape"][1] * p["shape"][2] / 1e6
        tag = "PUBLISHED" if VOLUMES[n].get("published") else " ".join(p["prepare_args"])
        print(f"{n:34s} {str(p['native_shape']):>20s} {str(p['shape']):>20s} "
              f"{str([round(v, 4) for v in p['spacing_zyx']]):>28s}  {mv:6.0f}  {tag}")
