"""Shared Z-resample plan for the 600 nm gate (card MSIDEPLOY-MODEL-002, Stage 1).

ONE source of truth for the 18 nm <-> "600 nm" Z mapping, imported by
`z600_build_degraded.py` (image, area average), `z600_oracle_floor.py` (labels,
mode) and `z600_wholeval_sweep.py` (segmentation, nearest upsample back). If the
three disagreed by a plane the gate number would be unreadable.

GEOMETRY. The val volume is [24, 18, 18] nm ZYX. The four ExPID71 600 nm volumes
sit at [33.3333, 9.0278, 9.0278] nm; after the standard XY x2 block average that
is [33.3333, 18.0556, 18.0556], i.e. Z is r = 33.3333/24 = 1.38889x the training
grid and XY is on it. So the gate degrades Z ONLY, by exactly that factor:
145 -> floor(145/r) = 104 planes.

THE THREE MAPS.

`weights(j)` -- output plane j integrates input z in [j*r, (j+1)*r), weight =
overlap length / r. That is `cv2.INTER_AREA` semantics written as a streamable
sum, and it is the project's calibrated rule for a fractional axis
(lessons/downsample_to_training_grid.md rule 3). Box, not Gaussian: a real
600 nm step integrates over the section. Gaussian-equivalent sigma = r/sqrt(12)
= 0.401 input voxels, against 0.194 for the skimage anti-alias convention, which
under-blurs relative to a physical acquisition.

`SEL` -- the categorical analogue, for instance labels, which cannot be
area-averaged. The weighted mode over the same window. With r < 2 that is always
a single source plane, so the mode downsample degenerates to plane SELECTION:
  * a 3-plane window has weights (a, 1, b) with a + b = r - 1 = 0.389 < 1, so
    the interior plane wins even if the two edge planes share a label;
  * a 2-plane window has weights summing to r, so the larger is > r/2.
`zplan()` asserts both, so the degeneracy is checked rather than assumed. The
consequence is stark and is the point of the label-damage measurement: 41 of the
145 input planes are dropped outright.

`UP` -- the inverse used to score on the 18 nm grid. Output plane j owns input
planes [j*r, (j+1)*r), so 18 nm plane z reads coarse plane floor(z/r). This is
the exact left-inverse of the area-average partition. Labels go back by nearest
neighbour because instance IDs cannot be interpolated.

NOT A 600 nm ACQUISITION. This is a software Z degradation with no PSF, exposure
or SNR model -- a lower bound on the difficulty, not a simulation of it. Card
alternative A2.
"""

from __future__ import annotations

import numpy as np

# From the ExPID71 volumes' own .zattrs (33.3333 nm Z after the 18x expansion
# fold) against the checkpoint's training grid (24 nm Z).
Z_NATIVE_NM = 33.3333
Z_TARGET_NM = 24.0
Z_FACTOR = Z_NATIVE_NM / Z_TARGET_NM  # 1.3888875


def zplan(n_in: int, factor: float = Z_FACTOR):
    """Return (n_out, weights, sel, up) for a Z-only resample by `factor`.

    weights : list of (z, w) per output plane, w summing to 1 (area average).
    sel     : (n_out,) int, the weighted-mode source plane (categorical).
    up      : (n_in,) int, coarse plane read by each original plane (nearest).
    """
    if not 1.0 < factor < 2.0:
        raise ValueError(
            f"zplan's mode->plane-selection proof assumes 1 < factor < 2, got {factor}")
    n_out = int(np.floor(n_in / factor))
    weights, sel = [], np.empty(n_out, dtype=np.int64)
    for j in range(n_out):
        lo, hi = j * factor, (j + 1) * factor
        row = []
        for z in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            ov = min(hi, z + 1) - max(lo, z)
            if ov > 1e-12:
                row.append((z, ov / factor))
        tot = sum(w for _, w in row)
        if abs(tot - 1.0) > 1e-9:
            raise AssertionError(f"output plane {j} weights sum to {tot}, not 1")
        row.sort(key=lambda t: -t[1])
        # The degeneracy proof, checked rather than assumed: the heaviest plane
        # outweighs every other plane in the window COMBINED, so no coalition of
        # planes sharing a label can outvote it and the weighted mode is a pure
        # plane selection.
        if row[0][1] <= 1.0 - row[0][1] + 1e-12:
            raise AssertionError(
                f"output plane {j}: weighted mode is not a plane selection "
                f"(top weight {row[0][1]:.6f} vs rest {1.0 - row[0][1]:.6f})")
        sel[j] = row[0][0]
        weights.append(sorted(row))
    if len(set(sel.tolist())) != n_out:
        raise AssertionError("mode plane selection is not injective")
    up = np.clip((np.arange(n_in) / factor).astype(np.int64), 0, n_out - 1)
    return n_out, weights, sel, up


def provenance(n_in: int, shape_yx, factor: float = Z_FACTOR) -> dict:
    n_out, _, sel, up = zplan(n_in, factor)
    return {
        "card": "MSIDEPLOY-MODEL-002",
        "source": "/projects/weilab/dataset/liconn/pytc/final_proofread/val",
        "source_resolution_nm_zyx": [24.0, 18.0, 18.0],
        "target_resolution_nm_zyx": [Z_NATIVE_NM, 18.0, 18.0],
        "axes_transformed": ["z"],
        "z_factor": factor,
        "source_shape_zyx": [n_in, int(shape_yx[0]), int(shape_yx[1])],
        "output_shape_zyx": [n_out, int(shape_yx[0]), int(shape_yx[1])],
        "image_method": (
            "exact area average along z (cv2.INTER_AREA semantics), streamed as an "
            "overlap-weighted sum over 2-3 input planes per output plane"),
        "antialias_kernel": (
            f"box, width {factor:.6f} input planes; gaussian-equivalent sigma "
            f"{factor / np.sqrt(12):.3f} input voxels"),
        "antialias_alternative_not_used": (
            "skimage resize anti_aliasing convention, gaussian sigma (r-1)/2 = "
            f"{(factor - 1) / 2:.3f} input voxels -- under-blurs vs a physical section"),
        "label_method": (
            "weighted mode over the same window; with r < 2 this degenerates to "
            "plane selection, verified in zplan()"),
        "dropped_input_planes": sorted(set(range(n_in)) - set(sel.tolist())),
        "n_dropped_input_planes": n_in - len(set(sel.tolist())),
        "upsample_back_method": "nearest, coarse plane floor(z/r), the partition's left inverse",
        "xy_transform": "none",
        "seg_handling": (
            "NOT degraded; the 18 nm GT is left untouched and is the scoring reference"),
        "is_a_600nm_acquisition": False,
        "note": (
            "Software Z degradation. No PSF, exposure or SNR model. A lower bound on "
            "600 nm difficulty, not a simulation of it. See card.md A2."),
        "selected_input_planes": sel.tolist(),
        "upsample_map": up.tolist(),
    }


def routeb_z_selection(n_in_18nm: int, zf: int = 3) -> tuple[np.ndarray, int]:
    """18 nm planes kept by route B's (zf, 2, 2) mode downsample of the 2x GT.

    Route B trains at [36, 18, 18] nm by block-downsampling the mip0 export at
    [12, 9, 9] by integer (3, 2, 2). The image half is an exact block average;
    the labels can only be moded. Because the mip0 `seg` is the 18 nm proofread
    segmentation nearest-upsampled 2x and bit-exact to it:

      * XY is an IDENTITY -- all four sub-voxels of a 2x2 block come from the
        same original voxel, so route B does not touch the labels in XY;
      * Z output plane j covers mip0 planes 3j..3j+2 = 18 nm planes
        floor(3j/2), floor((3j+1)/2), floor((3j+2)/2), always two copies of one
        plane and one of its neighbour, so the mode is the doubled plane,
        uniquely.

    So on the 18 nm label grid route B is exactly a 1.5x Z PLANE SELECTION:
    keep two planes in every three. Derived here rather than assumed, and the
    tie-freeness and injectivity are asserted. The bit-exactness the reduction
    rests on is checked against the real export in z600_routeb_label_damage.py.
    """
    n_mip0 = 2 * n_in_18nm
    n_out = n_mip0 // zf
    sel = np.empty(n_out, dtype=np.int64)
    for j in range(n_out):
        src = [(zf * j + k) // 2 for k in range(zf)]
        vals, counts = np.unique(np.array(src), return_counts=True)
        order = np.argsort(-counts)
        if len(order) > 1 and counts[order[0]] == counts[order[1]]:
            raise AssertionError(f"route B output plane {j} has a tied Z mode: {src}")
        sel[j] = int(vals[order[0]])
    if len(set(sel.tolist())) != n_out:
        raise AssertionError("route B plane selection is not injective")
    return sel, n_out
