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

import math
import os
from pathlib import Path


def _root(env: str, default) -> Path:
    """A storage root, overridable by environment variable.

    Every path below defaults to the BC checkout, so a SLURM submission is
    unaffected. The Google Cloud driver (`gcloud/`) runs the identical scripts
    inside a container where none of these paths exist, and sets each of these
    variables to its staged equivalent instead. That is the whole mechanism:
    there is no second copy of the recipe table, the prep script, the sweep or
    the uploader, so BC and GCP cannot drift apart.
    """
    return Path(os.environ.get(env) or default)


MOE_ROOT = _root("LICONN_MOE_ROOT",
                 "/projects/weilab/dataset/liconn/moe/preprocessed/clip_percentile_1_99")
SRC_ZARR = _root("LICONN_MOE_SRC_ZARR", MOE_ROOT / "zarr")
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
PREPARED = _root("LICONN_MOE_PREPARED", MOE_ROOT / _dirname)

# How far an EXACT integer block average may sit from the training grid before
# the interpolated area resample is preferred instead. 8% is not arbitrary: the
# published reference volume, ExPID96_2ndgel_S1_40XW001_18x, runs at Z 22.22 nm
# against the grid's 24 -- a 7.41% deviation -- and produced the batch's best
# result. So deviations up to about 8% are demonstrated-tolerable on this
# checkpoint, and below that an exact block average beats an interpolation that
# is nominally perfect. Above it, hit the grid.
INTEGER_TOL = 0.08


def choose_axis(native: float, target: float, integer_tol: float = INTEGER_TOL) -> dict:
    """Pick the best downsample for ONE axis, considering every option.

    The model is a conv net trained at [24, 18, 18] nm ZYX, so voxel size is not
    a free parameter: neurite caliber measured in voxels is part of what it
    learned. Each axis is therefore driven to the training grid independently,
    and the choice is between three kinds of answer:

    * **block** -- an integer block average. Exact, no interpolation, but only
      lands on spacings `native * f`. Preferred whenever one of those is within
      `integer_tol` of the target.
    * **area** -- a fractional `cv2.INTER_AREA` resample that hits the target
      exactly, at the cost of interpolating.
    * **native** -- the axis is left alone because reaching the target would
      require *more* samples than the microscope took. Downsampling removes
      information that is there; upsampling manufactures information that never
      was, and closes no part of the sampling gap. This is the refusal recorded
      in `lessons/liconn_expansion_factor.md`; the deviation is reported rather
      than papered over, because the volume genuinely is off-grid on that axis
      and any result has to be read in that light.

    Mixing modes across axes is normal and is usually the best answer: a volume
    can be an exact factor from the grid in Z and nowhere near it in XY.
    """
    ideal = target / native
    best = None
    for f in {math.floor(ideal), math.ceil(ideal), int(round(ideal))}:
        f = max(1, int(f))
        got = native * f
        dev = abs(got - target) / target
        if best is None or dev < best[2]:
            best = (f, got, dev)
    f_int, got_int, dev_int = best

    if ideal < 1.0:
        return {"mode": "native", "factor": 1, "achieved": native,
                "deviation": (native - target) / target,
                "why": "target is FINER than the acquisition; refusing to upsample"}
    if dev_int <= integer_tol:
        return {"mode": "block", "factor": f_int, "achieved": got_int,
                "deviation": (got_int - target) / target,
                "why": f"exact block average, within {integer_tol:.0%} of the grid"}
    return {"mode": "area", "factor": ideal, "achieved": target, "deviation": 0.0,
            "why": f"no integer factor is within {integer_tol:.0%}; resample onto the grid"}


def auto_recipe(shape, native, target=TRAIN_GRID_ZYX) -> dict:
    """Per-axis `choose_axis` for a whole volume, plus the CLI args to run it."""
    axes = [choose_axis(n, t) for n, t in zip(native, target)]
    if all(a["mode"] == "block" for a in axes):
        # Every axis is an exact integer factor: use the true block average.
        args = ["--factor", *[str(int(a["factor"])) for a in axes]]
    else:
        # Mixed, or any fractional axis. `--target-spacing` takes the ACHIEVED
        # spacing per axis, not the nominal grid -- for an integer-factor axis
        # that reproduces the block average to within one gray level (see
        # prepare_volume.py), and for the others it is the resample itself.
        args = ["--target-spacing", *[f"{a['achieved']:.6f}" for a in axes]]
    return {"axes": axes, "prepare_args": args,
            "achieved": tuple(a["achieved"] for a in axes),
            "worst_deviation": max(abs(a["deviation"]) for a in axes),
            "upsample_refused": [ax for ax, a in zip("ZYX", axes) if a["mode"] == "native"]}

# The IST-LICONN banis+ 200k checkpoint applied cross-sample to every moe volume.
REPO = _root("LICONN_MOE_REPO", "/projects/weilab/weidf/lib/pytorch_connectomics")
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
GCS_BUCKET = os.environ.get("LICONN_MOE_GCS_BUCKET") or "donglai_public"
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
    if name.startswith("ExPID108"):
        return "expid108"
    if name.startswith("ExPID71"):
        return "expid71"
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
    # Added 2026-09-21, PENDING index 11. THE FIRST VOLUME RUN ON GOOGLE CLOUD
    # rather than BC, and the first whose source is not on `/projects`: it
    # arrived already published at
    # `gs://donglai_public/liconn/moe/expid108/image/ExPID108_32x_Cortex_L1_01.zarr`
    # at [12.5, 5.078125, 5.078125] nm ZYX -- the same 32x grid as the S4 pair,
    # so the same `target` resample. `gcloud/run_volume.sh` stages that group to
    # `$LICONN_MOE_SRC_ZARR/ExPID108_32x_Cortex_L1_01.zarr`, which is why the
    # recipe below needs no cloud-specific branch.
    #
    # Sample series ExPID108, cortex layer 1. Out of domain for the same reasons
    # as every row above (checkpoint saw IST ExPID82_1 cortical neuropil) plus
    # the 32x caveat: the 32x rows in this batch have the lowest coverage and the
    # lowest affinity median, and the ExPID99 pair argues that is the fractional
    # Z resample (1.92x here) rather than contrast. Read it as provisional.
    "ExPID108_32x_Cortex_L1_01": {"target": TRAIN_GRID_ZYX},
    # Added 2026-09-21, PENDING index 12. From the ExPID71 z-step sweep at
    # `gs://donglai_public/liconn/moe/expid71/image/`. THIS IS THE ONLY VOLUME IN
    # THAT DROP THE STANDARD RECIPE CAN HONESTLY TAKE, and the reason is worth
    # stating because six siblings look equally runnable and are not.
    #
    # Every ExPID71 volume is 18x (XY 9.0278 nm = 162.5/18), so XY is the usual
    # ~2x reduction to the 18 nm training grid. Z is where they differ, and Z is
    # the swept variable:
    #
    #   z-step   biological Z   Z:XY    factor to 24 nm   direction
    #    300 nm    16.667 nm    1.85         1.440        DOWNsample -- fine
    #    500 nm    27.778 nm    3.08         0.864        UPSAMPLE x1.16
    #    600 nm    33.333 nm    3.69         0.720        UPSAMPLE x1.39
    #
    # Resampling a 500 or 600 nm acquisition onto [24,18,18] INTERPOLATES Z
    # UPWARD: it manufactures planes the microscope never sampled and closes no
    # part of the sampling gap. That is the refusal already recorded in
    # `lessons/liconn_expansion_factor.md` and in msi_liconn_deploy/spec.md task
    # 2, whose open question is precisely how to treat those volumes. Do NOT add
    # them here with `target` until that question is answered.
    #
    # At 300 nm the factor is 1.44 the other way -- an ordinary area-average
    # downsample, no invention -- so `target` is correct and needs no waiver.
    # `factor (1,2,2)` would be wrong here despite being the 18x default: it
    # leaves Z at 16.67 nm, 31% finer than the grid the checkpoint learned,
    # where the ExPID96 18x pair's (1,2,2) landed at 22.22 nm, within 7.4%.
    #
    # Prepared: (583, 1027, 1027) = 615 Mvoxel, 28.6% of ABISS's uint32
    # watershed cap, ~44 GB peak RSS at the measured ~71 GB/Gvoxel -- too tight
    # for a 64 GB machine, so run it on g2-standard-32 (1x L4, 128 GB).
    #
    # Hippocampus, and the checkpoint saw IST cortical neuropil: out of domain
    # on region as well as on sample.
    # `auto` rather than a hand-picked recipe: `choose_axis` drives each axis to
    # the training grid on its own, and here that is genuinely mixed -- Z has no
    # integer factor anywhere near 24 nm (1 -> 16.67, 2 -> 33.33) so it takes
    # the fractional resample, while XY is an exact x2 block average landing at
    # 18.056 nm, 0.31% off. Hand-picking one mode for all three axes would give
    # up one or the other.
    "ExPID71_Hippocampus_300nm_40XW01": {"auto": True},
    # Added 2026-09-22, PENDING indices 13-18. The rest of the ingested set.
    # All `auto`, so each axis is chosen against the training grid on its own.
    #
    # READ THE GRID DEVIATION BEFORE READING ANY RESULT FROM THESE. Six of the
    # seven are Z OFF-GRID, and `auto` does not hide it -- it refuses to
    # upsample, leaves Z native, and reports how far off that leaves the volume:
    #
    #   ExPID108_32x_Cortex_L1_02        Z  +4.2%   on-grid, exact x2 block avg
    #   ExPID71_2Hippocampus_500nm       Z +15.7%   Z native, 500 nm step
    #   ExPID71_*_600nm_* (four of them) Z +38.9%   Z native, 600 nm step
    #
    # +38.9% is further from the checkpoint's geometry than ANY volume in the
    # published ExPID96/99 batch, whose worst was the 32x pair at a fractional
    # Z resample. A poor result on a 600 nm volume is therefore a statement
    # about sampling, not about the tissue, and must not be read as one about
    # the tissue. The alternative -- interpolating Z up onto [24,18,18] -- would
    # invent planes and produce a cleaner-looking number that means less; see
    # `msi_liconn_deploy/spec.md` task 2, whose FILL is still open.
    #
    # The `_e030` / `_e120` pairs are the SAME plane at two exposures, so they
    # are two conditions rather than two channels, and differ only in SNR.
    # Comparing them is the point of the sweep and belongs to MSIDEPLOY-EVAL-002,
    # not to this pipeline: running them here yields layers to look at.
    "ExPID108_32x_Cortex_L1_02": {"auto": True},
    "ExPID71_2Hippocampus_500nm_40XW": {"auto": True},
    "ExPID71_Hippocampus_600nm_40XW02": {"auto": True},
    "ExPID71_120ms-30ms_600nm_40XW01_e030": {"auto": True},
    "ExPID71_120ms-30ms_600nm_40XW01_e120": {"auto": True},
    "ExPID71_120ms-30ms_600nm_40XW02_e030": {"auto": True},
    "ExPID71_120ms-30ms_600nm_40XW02_e120": {"auto": True},
    # Added 2026-09-22, second wave. Two more ExPID71 600 nm exposures (note the
    # underscore spelling `120ms_30ms`, a different acquisition from the
    # hyphenated `120ms-30ms` pair above), and the ExPID107 14.5x group.
    "ExPID71_120ms_30ms_600nm_40XW_e030": {"auto": True},
    "ExPID71_120ms_30ms_600nm_40XW_e120": {"auto": True},
    # ExPID107 is 14.5x: XY 11.034 nm resamples onto the 18 nm grid exactly, but
    # Z is 27.586 nm against a 24 nm target, so `auto` refuses to upsample and
    # leaves it native at **+14.9%**. Between the 400 nm/32x class (+4.2%) and
    # the 600 nm/18x class (+38.9%).
    #
    # THESE ARE THE LARGEST PREPARED VOLUMES IN THE PROJECT, because XY only
    # shrinks 1.63x and Z not at all: 1.7-2.3 Gvoxel against ExPID108's 0.26.
    # That puts them near ABISS's uint32 watershed index cap of 2,147,483,648
    # voxels, which is a hard limit in the `ws` binary, not a memory question:
    #
    #   _04  1,716,613,584   79.9% of cap   whole-volume decode fine
    #   _02  1,931,937,936   90.0%          fine
    #   _06  2,053,556,320   95.6%          tight; the IST val decode ran at 98.3%
    #   _05  2,286,824,368  106.5%          EXCEEDS THE CAP
    #
    # `ExPID107_14.5x_05` was held back here until 2026-09-22, when the cap turned
    # out to be an assert rather than a property of uint32 -- it is registered at
    # the end of this table; see the note there.
    "ExPID107_14.5x_02": {"auto": True},
    "ExPID107_14.5x_04": {"auto": True},
    "ExPID107_14.5x_06": {"auto": True},
    # Added 2026-09-22, third wave: the rest of the ExPID108 drop at
    # `gs://donglai_public/liconn/moe/expid108/image/`. Same 32x acquisition
    # grid as L1_01/L1_02 ([12.5, 5.078125, 5.078125] nm, XY 2304), so `auto`
    # lands on [25, 18, 18] -- Z +4.2%, exact x2 block average, no upsample:
    #
    #   L3_00          (1182, 2304, 2304) -> (591, 650, 650) = 250 Mvoxel  11.6% of cap
    #   L3_01          (1008, 2304, 2304) -> (504, 650, 650) = 213 Mvoxel   9.9%
    #   Hippocampus_01 (1137, 2304, 2304) -> (569, 650, 650) = 240 Mvoxel  11.2%
    #
    # L1_02 (255 Mvoxel) ran STAGES=all on g2-standard-16, so these do too.
    # Hippocampus is out of domain on region as well as sample, like ExPID71.
    "ExPID108_32x_Cortex_L3_00": {"auto": True},
    "ExPID108_32x_Cortex_L3_01": {"auto": True},
    "ExPID108_32x_Hippocampus_01": {"auto": True},
    # Added 2026-09-22. The fourth ExPID107 14.5x volume, (1147, 1412, 1412) =
    # 2,286,824,368 voxels -- 106.5% of the old uint32 cap, which is why it was
    # held back above. That cap was an assert on the VOXEL count. uint32 only
    # bounds the watershed's SEGMENT count; voxels are indexed with ptrdiff_t.
    # ABISS f6881cd (applied by gcloud/Dockerfile from image liconn-moe-v2)
    # replaces the assert with a segment-count check, so this volume gets ONE
    # full whole-volume decode with the same ws / max / percentile pick as its
    # three siblings, at uint32 memory (~165 GB at 71 GB/Gvoxel: run its CPU
    # stage on n2-highmem-32). On the first such decode the sweep re-decodes
    # the pick with ws64 and refuses to publish unless the labels are identical
    # (sweep_merge_threshold.py::crosscheck_ws64).
    "ExPID107_14.5x_05": {"auto": True},
    # Added 2026-09-22, fourth wave: the next ExPID108 drop (upload still in
    # progress when these were added; each was registered only once its image
    # group was complete). Same 32x grid as every ExPID108 row, so `auto` lands
    # on [25, 18, 18] -- Z +4.2%, exact x2 block average:
    #
    #   Hippocampus_03  (1220, 2304, 2304) -> (610, 650, 650) = 258 Mvoxel  12.0% of cap
    #   Hypothalamus_00 (1103, 2304, 2304) -> (552, 650, 650) = 233 Mvoxel  10.9%
    #   Hypothalamus_02 (1067, 2304, 2304) -> (534, 650, 650) = 225 Mvoxel  10.5%
    #   Piriform_03     (1218, 2304, 2304) -> (609, 650, 650) = 257 Mvoxel  12.0%
    #
    # Hippocampus, hypothalamus and piriform cortex are all out of domain on
    # region (the checkpoint saw IST cortical neuropil). Read accordingly.
    "ExPID108_32x_Hippocampus_03": {"auto": True},
    "ExPID108_32x_Hypothalamus_00": {"auto": True},
    "ExPID108_32x_Hypothalamus_02": {"auto": True},
    "ExPID108_32x_Piriform_03": {"auto": True},
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

    if rec.get("auto") and MOE_GRID == "train":
        # Per-axis choice against the training grid; see choose_axis.
        auto = auto_recipe(shape, tuple(native))
        achieved = np.asarray(auto["achieved"], dtype=np.float64)
        if auto["prepare_args"][0] == "--factor":
            factor = tuple(int(v) for v in auto["prepare_args"][1:])
            out_shape = tuple(s // f for s, f in zip(shape, factor))
            spacing = native * np.asarray(factor, dtype=np.float64)
        else:
            out_shape = tuple(
                max(1, int(round(n / (a / v))))
                for n, a, v in zip(shape, achieved, native))
            # Effective, not nominal: INTER_AREA maps the whole source extent
            # onto the whole output extent, so this is what keeps the physical
            # corners coincident and what the neuroglancer layer must declare.
            spacing = native * np.asarray(shape, dtype=np.float64) / np.asarray(out_shape)
        args = auto["prepare_args"]
    elif use_factor:
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
        # Signed per-axis deviation from the checkpoint's training grid, which
        # is the number that says how far out of domain the volume is before
        # anything is run. Reported by the table below and worth putting in any
        # write-up beside the result.
        "grid_deviation": [
            (sp - t) / t for sp, t in zip(spacing.tolist(), TRAIN_GRID_ZYX)
        ],
    }


if __name__ == "__main__":
    print(f"target grid {TRAIN_GRID_ZYX} nm ZYX (= {TRAIN_GRID_ZYX[::-1]} XYZ); "
          f"integer-factor tolerance {INTEGER_TOL:.0%}\n")
    print(f"{'volume':34s} {'-> prepared':>20s} "
          f"{'spacing ZYX nm':>26s} {'dev vs grid %':>22s}  {'Mvox':>6s}  recipe")
    for n in VOLUMES:
        # `plan` reads the source group's own attrs, so a volume staged only on
        # the other host is skipped rather than crashing the table. ExPID108's
        # source is in GCS and is present only under the cloud driver's
        # `LICONN_MOE_SRC_ZARR`.
        if not source_zarr(n).exists():
            print(f"{n:34s} {'(source not staged here: ' + str(source_zarr(n)) + ')'}")
            continue
        p = plan(n)
        mv = p["shape"][0] * p["shape"][1] * p["shape"][2] / 1e6
        tag = "PUBLISHED" if VOLUMES[n].get("published") else " ".join(p["prepare_args"])
        dev = ",".join(f"{d*100:+.1f}" for d in p["grid_deviation"])
        flag = "  <-- OFF-GRID" if max(abs(d) for d in p["grid_deviation"]) > 0.10 else ""
        print(f"{n:34s} {str(p['shape']):>20s} "
              f"{str([round(v, 3) for v in p['spacing_zyx']]):>26s} {dev:>22s}  "
              f"{mv:6.0f}  {tag}{flag}")
