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

import os
from pathlib import Path

MOE_ROOT = Path("/projects/weilab/dataset/liconn/moe/preprocessed/clip_percentile_1_99")
SRC_ZARR = MOE_ROOT / "zarr"
# GRID SCOPING. Which model scale the volumes are prepared for. `train` is the
# mip1 banis+ grid; `mip0` is the finer grid of the mip0 checkpoints.
#
#   MOE_GRID=mip0 MOE_CKPT=<mip0 ckpt> MOE_OUT_ROOT=<...>/mip0_eb8 MOE_GCS_KIND=mip0_eb8
#
# THE 18x VOLUMES ARE THE ONLY ONES WHERE mip0 IS INTERESTING, and for one
# specific reason: their native spacing is [22.22, 9.03, 9.03] nm, so their XY is
# ALREADY at the mip0 model's 9 nm to within 0.3%. The mip1 recipe throws that
# away with a (1,2,2) block average. At mip0 the model sees real, un-averaged XY.
#
# The cost is Z: 22.22 -> 12 nm is a 1.852x INTERPOLATION, i.e. invented planes.
# The closest precedent in this batch is the ExPID99 32x pair, which interpolates
# Z by 1.92x and lands worst on coverage (0.550/0.633) -- the README attributes
# that to the resample rather than the biology. So mip0-on-18x runs a real XY
# gain against a known-bad Z interpolation, and the two are confounded. Read any
# result accordingly; it is not a clean "is mip0 better" test.
_GRIDS = {
    "train": ("prepared_train_grid", (24.0, 18.0, 18.0)),
    "mip0": ("prepared_mip0_grid", (12.0, 9.0, 9.0)),
}
MOE_GRID = os.environ.get("MOE_GRID", "train")
if MOE_GRID not in _GRIDS:
    raise ValueError(f"MOE_GRID must be one of {tuple(_GRIDS)}")
_dirname, TRAIN_GRID_ZYX = _GRIDS[MOE_GRID]
PREPARED = MOE_ROOT / _dirname

# The IST-LICONN banis+ 200k checkpoint applied cross-sample to every moe volume.
REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")
#
# MODEL SCOPING. Both of the following are env-overridable so a SECOND checkpoint
# can be decoded without touching the first one's artifacts. They must move
# together:
#
#   MOE_CKPT=<other ckpt> MOE_OUT_ROOT=<other tree> python sweep_merge_threshold.py ...
#
# Changing MOE_CKPT alone is a data-loss bug, not a shortcut. sweep_merge_threshold.py
# writes <work_dir>/mt_sweep.json at a FIXED name and does final.unlink() before
# hard-linking the chosen segmentation -- and for the LEGACY volume layer_name()
# returns a fixed string. So a second model decoded into the default OUT_ROOT
# would delete the first model's published segmentation and keep its filename,
# leaving an artifact whose name says one model and whose contents are another.
CKPT = Path(os.environ.get(
    "MOE_CKPT",
    REPO / "outputs/liconn_final_banis_plus_tube/20260728_032436/checkpoints/step=00200000.ckpt"))
# In --mode test `runtime/checkpoint_dispatch.py` overwrites `inference.save_path`
# with <ckpt run dir>/test_<ckpt stem>/, so this is where results actually land --
# the `save_path:` in the step YAMLs is ignored. The per-volume leaf is the stem
# of the image path, which is why `prepare_volume.py` writes `<vol>.h5` and not
# `<vol>.zarr/0` (every zarr volume would share the leaf "0" and overwrite).
TEST_OUT = CKPT.parent.parent / f"test_{CKPT.stem}"
# Default is the eb2 subtree: artifacts were reorganised 2026-09-16 so each
# model owns a sibling directory under outputs/neuron_liconn_moe/ rather than
# sharing one namespace. Everything derived from OUT_ROOT (work_dir, the
# precomputed staging tree, the qc output) moves with it.
DEFAULT_OUT_ROOT = REPO / "outputs/neuron_liconn_moe/eb2"
OUT_ROOT = Path(os.environ.get("MOE_OUT_ROOT", DEFAULT_OUT_ROOT))

# Publication target: the PUBLIC bucket (not anonymously readable -- reads and
# writes need the donglai@mindspan.org account), which holds both the OME-Zarr
# image groups and the segmentation layers, so layers sit next to the images
# they overlay.
GCS_BUCKET = "donglai_public"
GCS_ROOT = f"gs://{GCS_BUCKET}/liconn/moe"

# LAYOUT (2026-09-16). Objects are grouped by sample series, then by what they
# are: gs://donglai_public/liconn/moe/<family>/<kind>/<name>
#   family  expid96 | expid99
#   kind    image | mip1_eb2 | mip1_eb8 | mip0_eb8
# `mip1` is the model scale: every moe volume is resampled onto the checkpoint's
# [24,18,18] nm training grid, so a mip1 model is what produced these layers.
#
# THIS DROPPED THE `clip_percentile_1_99` PATH COMPONENT, which the previous
# layout carried deliberately: a second clip variant with IDENTICAL dataset
# names exists at preprocessed/zarr/, and nothing in the new path records which
# variant an image came from. If that variant is ever published it collides in
# <family>/image/. Record the clip variant in the image group's own metadata, or
# reintroduce it as a suffix -- do not rely on the path to disambiguate it.
GCS_KINDS = ("image", "mip1_eb2", "mip1_eb8", "mip0_eb8")


def family(name: str) -> str:
    """Sample series a volume belongs to -- the first path component on GCS."""
    if name.startswith("ExPID96"):
        return "expid96"
    if name.startswith("ExPID99"):
        return "expid99"
    raise ValueError(f"unknown sample series for {name!r}")


def gcs_kind() -> str:
    """Which model tree is being published, derived from the scoped OUT_ROOT."""
    explicit = os.environ.get("MOE_GCS_KIND")
    if explicit:
        if explicit not in GCS_KINDS:
            raise ValueError(f"MOE_GCS_KIND must be one of {GCS_KINDS}")
        return explicit
    model = OUT_ROOT.name                      # eb2 | eb8, see MODEL SCOPING above
    if model not in ("eb2", "eb8"):
        raise ValueError(
            f"cannot infer the GCS kind from OUT_ROOT {OUT_ROOT}; set MOE_GCS_KIND")
    return f"mip1_{model}"


def gcs_folder(name: str, kind: str | None = None) -> str:
    return f"{GCS_ROOT}/{family(name)}/{kind or gcs_kind()}"

# ngauth server for the private `donglai` bucket, kept for reference. Whether it
# is authorised for `donglai_public` has not been established.
NGAUTH = f"gs+ngauth+https://sunny-catalyst-506019-a2.ue.r.appspot.com/{GCS_BUCKET}"


def layer_url(layer: str, volume: str, kind: str | None = None) -> str:
    """Neuroglancer source for a published layer.

    Plain `gs://` assumes the bucket is anonymously readable. As of 2026-09-04 it
    is NOT: an unauthenticated
    `https://storage.googleapis.com/storage/v1/b/donglai_public/o` returns 401,
    the same as the known-private `donglai` bucket, where a public bucket answers
    200. Until `allUsers:objectViewer` is granted, use the `NGAUTH` form instead
    (and confirm that server is authorised for this bucket).
    """
    return f"precomputed://{gcs_folder(volume, kind)}/{layer}"


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
    """The path handed to the model as `data.test.image`.

    The LEGACY pin is a mip1 artifact: it points at the `zarr_ds1-2-2` copy that
    predates `prepare_volume.py` writing `<vol>.h5`. It is the (1,2,2) block
    average, i.e. the TRAIN grid, so honouring it on any other grid would feed
    the model a downsampled volume while the config says otherwise.
    """
    if name in LEGACY and MOE_GRID == "train":
        return LEGACY[name]["image"]
    return prepared_h5(name)



def affinity_h5(name: str) -> Path:
    """Step 1 output for `name` (3, Z, Y, X) float16.

    The `0/` leaf is NOT a property of the volume, it is a property of the image
    path it was run from: `inference/output.py::resolve_output_filenames` names
    the leaf from the last path component, and the LEGACY pin above ends in
    `.zarr/0` ("0" is not in `_UNINFORMATIVE_STEMS`, so the walk-up-to-parent
    rule never fires). Both mip1 runs of this volume therefore wrote to `0/`.
    `make_volume_config.py` does not consult `prepared_image()` at all -- it
    writes `image: {name}.h5` from `plan()` -- so on any other grid the leaf is
    the volume's own name, and redirecting to `0/` raises FileNotFoundError on a
    path that only ever existed under the mip1 checkpoints. Observed 2026-09-21
    on MOE_GRID=mip0, whose affinity is (3, 585, 2304, 2304) under the NAMED
    leaf. Scope the redirect to the grid that produced it.
    """
    if name in LEGACY and MOE_GRID == "train":
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
    # The pinned LEGACY name exists ONLY to keep the already-published eb2 layer
    # reachable on GCS. It is a two-decimal name from before this convention, and
    # it hardcodes a threshold. Applying it to any other model produces a file
    # whose name states a threshold that model did not choose -- observed
    # 2026-09-16, when eb8 picked 0.6079 and still got a file called `mt060`.
    # So honour the pin only in the default (eb2) tree.
    if name in LEGACY and "layer" in LEGACY[name] and OUT_ROOT == DEFAULT_OUT_ROOT:
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

    # The `factor` entries are hand-picked for the TRAIN grid only -- (1,2,2) on
    # an 18x volume lands on [22.22, 18.06, 18.06], which is meaningless for any
    # other target. On a non-train grid every volume resamples to that grid.
    use_factor = "factor" in rec and MOE_GRID == "train"

    # mip0: feed the 18x volumes NATIVE. Their XY is already 9.03 nm, within
    # 0.3% of the mip0 model's 9 nm, so there is nothing to gain by resampling
    # XY and the mip1 (1,2,2) average would throw the detail away. Z is left at
    # its native 22.22 nm rather than interpolated up to 12: a 1.852x Z
    # interpolation invents planes, and the closest precedent in this batch (the
    # 32x pair, Z interpolated 1.92x) lands worst on coverage. So the model gets
    # real XY at its trained scale and a Z step ~1.85x coarser than it saw in
    # training -- an honest mismatch instead of manufactured data.
    if MOE_GRID == "mip0":
        use_factor = True
        rec = dict(rec, factor=(1, 1, 1))

    if use_factor:
        factor = tuple(rec["factor"])
        out_shape = tuple(s // f for s, f in zip(shape, factor))
        spacing = native * np.asarray(factor, dtype=np.float64)
        args = ["--factor", *[str(f) for f in factor]]
    else:
        target = np.asarray(rec.get("target", TRAIN_GRID_ZYX) if MOE_GRID == "train"
                            else TRAIN_GRID_ZYX, dtype=np.float64)
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
