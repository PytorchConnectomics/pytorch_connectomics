#!/usr/bin/env python3
"""The one eb2 segmentation this error analysis runs on, and where its parts live.

Single source of truth for the skeletonize -> analyze -> upload -> link chain.
Run it directly to print the resolved paths and check they exist.

`ExPID96_2ndgel_S1_40XW001_18x` is the first moe volume: an exact (1, 2, 2) block
average of the native 18x zarr, so its grid carries no interpolation, and the one
volume of the batch already published as a precomputed layer. Its spacing is
expansion-corrected biological nm -- do not divide by the expansion again.

There is NO GROUND TRUTH for any moe volume. Nothing downstream is a score.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py

NAME = "ExPID96_2ndgel_S1_40XW001_18x"
REPO = Path("/projects/weilab/weidf/lib/pytorch_connectomics")

# The eb2 (1 GPU, effective batch 2) IST-LICONN banis+ 200k checkpoint applied
# cross-sample, then ABISS at the percentile-matched merge threshold 0.60.
RUN = REPO / "outputs/neuron_liconn_moe/eb2" / NAME
SEG = RUN / f"{NAME}_seg_abiss_mt060.h5"
SWEEP = RUN / "mt_sweep.json"
# Per-label voxel counts, one pass over SEG. IDs are dense 1..105975.
SIZES = RUN / "label_sizes.npz"
# Source affinity, ZYX ch0=Z ch1=Y ch2=X, `scale_sigmoid` (sigmoid(0.2 * logit)).
# This volume predates the `<vol>.h5` naming fix, so its leaf is the NGFF level
# index "0" rather than the volume name.
AFFINITY = (
    REPO
    / "outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/0"
    / "raw_x1_ch0-1-2.h5"
)

SHAPE_ZYX = (585, 1152, 1152)
SPACING_NM_ZYX = (22.22222222222222, 18.055555555555557, 18.055555555555557)

# Any other volume: set LICONN_EA_RUN to its run directory (the one holding
# `<name>_seg_abiss_mt*.h5` and `mt_sweep.json`). Spacing comes from that
# volume's own sweep record and shape from its h5 -- never from the constants
# above, which are S1's: the 22x/28x/32x volumes sit near [24, 18, 18] nm.
# LICONN_EA_AFFINITY overrides the sweep's absolute affinity path (containers).
if os.environ.get("LICONN_EA_RUN"):
    RUN = Path(os.environ["LICONN_EA_RUN"]).resolve()
    NAME = RUN.name
    SWEEP = RUN / "mt_sweep.json"
    SIZES = RUN / "label_sizes.npz"
    _segs = [p for p in RUN.glob(f"{NAME}_seg_abiss_mt*.h5") if "nogt" not in p.name]
    if len(_segs) != 1:
        raise FileNotFoundError(f"expected one {NAME}_seg_abiss_mt*.h5 in {RUN}, found {_segs}")
    SEG = _segs[0]
    _sweep = json.loads(SWEEP.read_text())
    SPACING_NM_ZYX = tuple(float(v) for v in _sweep["spacing_zyx_nm"])
    AFFINITY = Path(os.environ.get("LICONN_EA_AFFINITY") or _sweep["affinity"])
    with h5py.File(SEG, "r") as _handle:
        SHAPE_ZYX = tuple(int(n) for n in _handle["main"].shape)

OUT = RUN / "error_analysis"
SKELETONS = OUT / "skeletons.npz"
CONTINUITY = OUT / "continuity.json"
ANALYSIS = OUT / "error_analysis.json"
LINKS = OUT / "split_links.json"

SPACING_UM_ZYX = tuple(v / 1000.0 for v in SPACING_NM_ZYX)
EXTENT_UM_ZYX = tuple(n * s for n, s in zip(SHAPE_ZYX, SPACING_UM_ZYX))

# The analysis cohort. 15,986 of 105,975 labels reach this size and they hold
# 94.3% of foreground voxels; ABISS dust already floors every label at 100.
MIN_VOXELS = 1000

# Skeletonization, matching the settings validated on the sibling moe volume in
# dev/astra_nogt_eval/glia_revision/build_arbors.py.
TEASAR_PARAMS = {
    "scale": 1.5,
    "const": 100.0,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_detection_threshold": 1e9,
    "soma_acceptance_threshold": 1e9,
}
SIMPLIFICATION_NM = 50.0

# Publication target, verified against the live bucket 2026-09-17. The bucket was reorganised by
# model and mip since this batch was first published: the layers are under
# `liconn/moe/expid96/<mip>_<eb>/`, NOT under `liconn/moe/clip_percentile_1_99`,
# which now holds only a stray earlier upload. The published seg layer for this
# volume is a uint32 precomputed set with 4 scales and meshes for exactly the
# 15,986 labels >= 1000 voxels -- the same ID set this analysis measures, which
# is why the catalog's IDs address it directly.
GCS_BUCKET = "donglai_public"
GCS_PREFIX = "liconn/moe/expid96/mip1_eb2"
SEG_LAYER = f"{NAME}_seg_abiss_mt060"
if os.environ.get("LICONN_EA_RUN"):
    # Bucket layout: liconn/moe/<expid>/<mip>_<eb>/<seg h5 stem>/. Run dirs sit
    # under outputs/neuron_liconn_moe/{eb2,eb8,mip0_eb8}; eb2/eb8 are mip1.
    _model = RUN.parent.name
    GCS_PREFIX = (f"liconn/moe/{NAME.split('_')[0].lower()}/"
                  f"{_model if _model.startswith('mip') else 'mip1_' + _model}")
    SEG_LAYER = SEG.stem
# Sit beside `info` inside the layer, so a viewer that has the layer has the
# records without being told a second location.
GCS_ANALYSIS_PREFIX = f"{GCS_PREFIX}/{SEG_LAYER}"
GCS_IMAGE_LAYER = f"liconn/moe/expid96/image/{NAME}.zarr"
STALE_GCS_PREFIX = "liconn/moe/clip_percentile_1_99"


def main() -> int:
    rows = [
        ("segmentation", SEG),
        ("merge-threshold sweep", SWEEP),
        ("label sizes", SIZES),
        ("affinity", AFFINITY),
        ("output dir", OUT),
    ]
    width = max(len(name) for name, _ in rows)
    for name, path in rows:
        print(f"{name:{width}s}  {'ok ' if path.exists() else 'MISSING'}  {path}")
    print()
    print(f"shape   {SHAPE_ZYX}  spacing_nm {[round(v, 4) for v in SPACING_NM_ZYX]}")
    print(f"extent  {[round(v, 3) for v in EXTENT_UM_ZYX]} um")
    print(f"cohort  labels with >= {MIN_VOXELS} voxels")
    print(f"gcs     gs://{GCS_BUCKET}/{GCS_ANALYSIS_PREFIX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
