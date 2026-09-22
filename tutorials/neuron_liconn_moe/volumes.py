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
PREPARED = _root("LICONN_MOE_PREPARED", MOE_ROOT / "prepared_train_grid")

TRAIN_GRID_ZYX = (24.0, 18.0, 18.0)

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
#
# On GCP the same weights are pulled from `pytc/liconn` on HuggingFace as
# `affinity_expid82_18nm_128x128x128.ckpt` -- same specimen (ExPID82_1), same
# [24,18,18] nm grid, same 200k steps. `LICONN_MOE_CKPT` must still point at a
# path with a `YYYYmmdd_HHMMSS` ancestor directory, because
# `runtime/checkpoint_dispatch.py::get_output_base_from_checkpoint` looks for
# exactly that to decide where test outputs land; without one it falls back to
# `<ckpt>/../../<stem>`, which for a checkpoint in a top-level directory is a
# path at the filesystem root. `gcloud/run_volume.sh` stages it accordingly.
REPO = _root("LICONN_MOE_REPO", "/projects/weilab/weidf/lib/pytorch_connectomics")
CKPT = _root("LICONN_MOE_CKPT",
             REPO / "outputs/liconn_final_banis_plus_tube/20260728_032436/checkpoints/step=00200000.ckpt")
# In --mode test `runtime/checkpoint_dispatch.py` overwrites `inference.save_path`
# with <ckpt run dir>/test_<ckpt stem>/, so this is where results actually land --
# the `save_path:` in the step YAMLs is ignored. The per-volume leaf is the stem
# of the image path, which is why `prepare_volume.py` writes `<vol>.h5` and not
# `<vol>.zarr/0` (every zarr volume would share the leaf "0" and overwrite).
TEST_OUT = CKPT.parent.parent / f"test_{CKPT.stem}"
OUT_ROOT = _root("LICONN_MOE_OUT_ROOT", REPO / "outputs/neuron_liconn_moe")

# Publication target: the PUBLIC bucket, which already mirrors all eight OME-Zarr
# image groups (and the first segmentation) under the same prefix, so layers land
# next to the images they overlay.
#
# The `clip_percentile_1_99` component is load-bearing and must not be dropped: a
# second clip variant with IDENTICAL dataset names exists at
# `preprocessed/zarr/`, so a flat `liconn/moe/` would collide the moment the
# other variant is published. See [liconn_expid96_moe_preprocessed].
# `GCS_PREFIX` is overridable because the layout moved on: ExPID96/99 were
# published under the clip-variant prefix, while ExPID108 arrived already
# published at `liconn/moe/expid108/image/`, and a segmentation belongs beside
# the image it overlays rather than under another volume's clip name.
GCS_BUCKET = os.environ.get("LICONN_MOE_GCS_BUCKET") or "donglai_public"
GCS_PREFIX = os.environ.get("LICONN_MOE_GCS_PREFIX") or "liconn/moe/clip_percentile_1_99"
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

    if rec.get("auto"):
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
    elif "factor" in rec:
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
        # Signed per-axis deviation from the checkpoint's training grid, which
        # is the number that says how far out of domain the volume is before
        # anything is run. Reported by the table below and worth putting in any
        # write-up beside the result.
        "grid_deviation": [
            (sp - t) / t for sp, t in zip(spacing.tolist(), TRAIN_GRID_ZYX)
        ],
    }


if __name__ == "__main__":
    print(f"target grid {TRAIN_GRID_ZYX} nm ZYX (= [18,18,24] XYZ); "
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
