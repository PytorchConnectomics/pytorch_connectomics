"""Slice-based branch merge for waterz segmentation postprocessing.

Resolves false splits by analyzing segment continuity across z-slices.  Merge
candidates are only considered at z-boundaries where a segment *ends* or
*begins* (its bbox z-extremes); interior 2D overlap between neighbours is
coincidence, and merging on it chains adjacent neurons through the union-find
(ARE 0.05 → 0.92 on SNEMI).

The algorithm runs in three stages, in this order:

1. **Cross-section completion** (:func:`complete_sections`) — absorb every
   1-slice/small fragment into the segment it most touches, so each
   cross-section is whole before any IoU is computed.

2. **Mutual IoU merge** (:func:`merge_sections`) — per z-seam, pick the
   best-IoU partner, require it to be mutual and to beat the runner-up by
   ``margin``, with the mean seam z-affinity used only as a background floor.

3. **Weak-gap bridge** (:func:`bridge_weak_gaps`) — the same operator reaching
   across *g* weak-affinity slices, scoring the projected cross-section mask.

:func:`completion_radius_link` (stage 4) is opt-in via ``prefer_length``: it
links open ends by radius and is oracle-merge-negative.

===============================================================================
CUE LADDER — what to trust when deciding "are these two pieces one neuron?"
===============================================================================
Ordered MOST → LEAST robust. Measured on MIT-LiCONN DL288B (NERL base +
oracle-merge om, merge_threshold=10); "om-safe" = the false-merge-free ceiling
did not drop. Use a lower cue only to *veto*, never to *select*.

 1. SHAPE CONTINUITY — real (unshifted) cross-section IoU.          [primary]
    fn: `merge_sections`
    The one signal that actually says "same tube". Selecting by IoU instead of
    affinity moved base 0.7495 → 0.7840. PRECONDITION: the cross-sections must
    be WHOLE (see cue 2) or the IoU is computed on half a tube and lies.

 2. CROSS-SECTION COMPLETENESS — absorb 1-slice/small fragments first.
    fn: `complete_sections`  (lateral neighbour, else best-IoU z-neighbour)
    A fragment holding half a cross-section makes a true merge look like a
    non-match (real case: IoU 0.79 pair invisible because one side was a
    645-vox fragment). base +0.009, om +0.001 — the cheapest win here.

 3. MUTUAL AGREEMENT — both endpoints pick each other.              [safety]
    fn: `merge_sections` (the `up`/`down` cross-check before union-find)
    Mutual-best is merge-SAFE; one-sided dominance is not (it chains adjacent
    neurons — the ARE 0.05 → 0.92 collapse noted above). Never merge on a
    one-sided preference alone.

 4. UNAMBIGUITY — the best partner must beat the runner-up by `margin`.
    fn: `merge_sections(margin=...)`, `bridge_weak_gaps(margin=...)`
    "No other close/confusing match." Rejecting 5 ambiguous pairs bought
    om +0.002 at zero base cost. If two candidates are close, LEAVE IT SPLIT —
    a false merge costs far more than a residual split.
    CAVEAT: this only protects you if the true partner is INSIDE the search
    window; a truncated search makes a wrong match look certain.

 5. ENDPOINT RESTRICTION — only where a segment begins/ends.        [veto]
    fn: `merge_sections` (candidates are read at each segment's bbox z0/z1)
    Interior 2D overlap between neighbours is coincidence, not continuity.

 6. Z-AFFINITY — a background FLOOR, not a selector.                [veto]
    fn: `merge_sections` (mean seam affinity vs `aff_lo`)
    Strong-affinity seams are already merged upstream, so what remains is
    mostly LOW-affinity; ranking by affinity picks the wrong partner at a
    2-branch junction. Use it to exclude background (> ~0.4), nothing more.

 7. CALIBER / AREA RATIO — weak.                                    [veto only]
    Matches thickness, not shape; happily fuses a same-size neighbour.

 8. TRAJECTORY EXTRAPOLATION — least robust; do not select on it.
    Velocity from a few slices is noisy, a false merge is *also* collinear,
    and it cannot follow a tube that drifts >1 radius across a gap. Every
    variant tried (collinear veto, low-IoU collinear rescue) was net-negative.

 9. ENDPOINT PROXIMITY — not usable on its own.
    Linking nearest open ends lost om in every configuration tested; the
    correct links were not separable from the false ones by distance.

RULE OF THUMB: cues 1–2 decide, 3–4 make it safe, 5–7 veto, 8–9 are for
diagnostics/ranking only. To go beyond this ladder you need a signal shape does
not carry (membrane/orthogonal-plane evidence, or a learned scorer).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, center_of_mass

from ....data.processing.bbox import apply_lut, seg_stats

__all__ = ["branch_merge"]


# v2_merge.py: IoU-primary completion and mutual best-buddy merge.
AFF_LO, MERGE_IOU, MIN_OV, MIN_SIZE, MERGE_ROUNDS = 0.4, 0.45, 30, 2000, 4
MARGIN = 0.15
IOU_LO, DIST_THR = 0.45, 8.0

# v3_weak.py: projected-mask bridge through weak affinity.
WEAK_MAX_GAP, CAL_RATIO, WEAK_MIN_IOU, WEAK_LO, WEAK_MIN_SIZE, WEAK_ROUNDS = (
    15,
    1.6,
    0.35,
    0.3,
    2000,
    3,
)
DIM_TOL = 3
WEAK_MARGIN = 0.15

# v4_complete.py: optional, completion-driven radius link. This stage was
# oracle-merge-negative and is therefore only run when prefer_length=True.
SPAN_FRAC, COMPLETE_MIN_SIZE, BORDER = 0.20, 20000, 2
COMPLETE_MAX_GAP, MAX_LAT, COMPLETE_CAL_RATIO, COMPLETE_MARGIN, COS_MIN = (
    25,
    45.0,
    1.8,
    8.0,
    -0.2,
)


def _find(parent: np.ndarray, label: int) -> int:
    while parent[label] != label:
        parent[label] = parent[parent[label]]
        label = int(parent[label])
    return label


def complete_sections(
    seg: np.ndarray,
    min_size: int = MIN_SIZE,
    zfrag_iou: float = 0.3,
    verbose: bool = False,
    *,
    stats: tuple[Any, Any, Any] | None = None,
) -> np.ndarray:
    """Absorb lateral and z-isolated cross-section fragments before matching."""
    if stats is None:
        bounds, sizes, _ = seg_stats(seg)
    else:
        bounds, sizes, _ = stats
    fragments = [
        label
        for label in bounds
        if label > 0 and (sizes[label] < min_size or (bounds[label][1] - bounds[label][0] + 1) <= 2)
    ]
    relabel: np.ndarray = np.arange(int(seg.max()) + 1, dtype=np.uint32)
    lateral_count = 0
    z_count = 0
    for fragment in fragments:
        z0, z1, y0, y1, x0, x1 = bounds[fragment]
        best: Counter[int] = Counter()
        for z in range(z0, z1 + 1):
            mask = (seg[z] == fragment)[y0 : y1 + 1, x0 : x1 + 1]
            if not mask.any():
                continue
            neighbors = seg[z][y0 : y1 + 1, x0 : x1 + 1][
                binary_dilation(mask, iterations=1) & ~mask
            ]
            for label in np.unique(neighbors[neighbors > 0]).tolist():
                if sizes[label] >= min_size:
                    best[label] += int((neighbors == label).sum())
        if best:
            relabel[fragment] = best.most_common(1)[0][0]
            lateral_count += 1
            continue
        zbest = None
        for target_z, neighbor_z in ((z0, z0 - 1), (z1, z1 + 1)):
            if not (0 <= neighbor_z < seg.shape[0]):
                continue
            mask = seg[target_z] == fragment
            if not mask.any():
                continue
            neighbors = seg[neighbor_z][binary_dilation(mask, iterations=1)]
            for label in np.unique(neighbors[neighbors > 0]).tolist():
                if label == fragment or sizes[label] < min_size:
                    continue
                label_mask = seg[neighbor_z] == label
                iou = int((mask & label_mask).sum()) / max(int((mask | label_mask).sum()), 1)
                if iou > zfrag_iou and (zbest is None or iou > zbest[0]):
                    zbest = (iou, int(label))
        if zbest:
            relabel[fragment] = zbest[1]
            z_count += 1
    for _ in range(3):
        relabel = relabel[relabel]
    if verbose:
        print(
            f"  completion: absorbed {lateral_count} lateral + " f"{z_count} z-isolated fragments",
            flush=True,
        )
    return apply_lut(seg, relabel)


def merge_sections(
    seg: np.ndarray,
    afz: np.ndarray,
    *,
    aff_lo: float = AFF_LO,
    merge_iou: float = MERGE_IOU,
    min_ov: int = MIN_OV,
    min_size: int = MIN_SIZE,
    iou_lo: float = IOU_LO,
    dist_thr: float = DIST_THR,
    margin: float = MARGIN,
    rounds: int = MERGE_ROUNDS,
    inplace: bool = False,
    verbose: bool = False,
    stats: tuple[Any, Any, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Complete cross-sections, then apply IoU-primary mutual merges.

    ``iou_lo``/``dist_thr`` are INERT, kept to mirror the research signature: they
    parameterised the collinear low-IoU rescue, which was measured net-negative
    (base +0.0005, om -0.0036 — it admits end-to-end collinear false merges) and is
    therefore not vendored. Setting them changes nothing.
    """
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    z_size, y_size, x_size = seg.shape
    del y_size, x_size
    seg = complete_sections(seg, min_size, verbose=verbose, stats=stats)
    total = 0
    for round_index in range(rounds):
        bounds, sizes, _ = seg_stats(seg)
        candidates = [label for label in bounds if label > 0 and sizes[label] >= min_size]
        up: dict[int, tuple[int, float, float]] = {}
        down: dict[int, tuple[int, float, float]] = {}
        for first in candidates:
            z0, z1, y0, y1, x0, x1 = bounds[first]
            for target_z, neighbor_z, store in (
                (z1, z1 + 1, up),
                (z0, z0 - 1, down),
            ):
                if not (0 <= neighbor_z < z_size):
                    continue
                first_mask = (seg[target_z] == first)[y0 : y1 + 1, x0 : x1 + 1]
                if not first_mask.any():
                    continue
                neighbors = seg[neighbor_z][y0 : y1 + 1, x0 : x1 + 1]
                seam_affinity = afz[max(target_z, neighbor_z)][y0 : y1 + 1, x0 : x1 + 1]
                dilated = binary_dilation(first_mask, iterations=1)
                region = dilated & (neighbors > 0) & (neighbors != first)
                matches = []
                for second in np.unique(neighbors[region]).tolist():
                    if second <= 0 or second == first:
                        continue
                    second_mask = neighbors == second
                    overlap = dilated & second_mask
                    if int(overlap.sum()) < min_ov:
                        continue
                    z_affinity = float(seam_affinity[overlap].mean())
                    if z_affinity <= aff_lo:
                        continue
                    intersection = int((first_mask & second_mask).sum())
                    union = int((first_mask | second_mask).sum())
                    iou = intersection / max(union, 1)
                    if iou > merge_iou:
                        matches.append((int(second), z_affinity, iou))
                if not matches:
                    continue
                matches.sort(key=lambda candidate: -candidate[2])
                if len(matches) > 1 and matches[0][2] - matches[1][2] < margin:
                    continue
                store[first] = matches[0]
        parent: np.ndarray = np.arange(int(seg.max()) + 1, dtype=np.int64)
        merge_count = 0
        for first in candidates:
            edge = up.get(first)
            if edge is None:
                continue
            second = edge[0]
            reverse_edge = down.get(second)
            if reverse_edge is not None and reverse_edge[0] == first:
                first_root, second_root = _find(parent, first), _find(parent, second)
                if first_root != second_root:
                    parent[first_root] = second_root
                    merge_count += 1
        if verbose:
            print(
                f"  round {round_index}: {merge_count} mutual merges",
                flush=True,
            )
        if merge_count == 0:
            break
        root = np.array(
            [_find(parent, index) for index in range(len(parent))],
            dtype=np.uint32,
        )
        apply_lut(seg, root)
        total += merge_count
    return seg, total


def _weak_velocity(
    seg: np.ndarray,
    label: int,
    z_end: int,
    direction: int,
    bounds: dict[int, tuple[int, int, int, int, int, int]],
    n: int = 4,
) -> np.ndarray:
    points = []
    for k in range(n):
        z = z_end - direction * k
        if bounds[label][0] <= z <= bounds[label][1] and (seg[z] == label).any():
            points.append((z, *center_of_mass(seg[z] == label)))
    if len(points) < 2:
        return np.zeros(2)
    array = np.array(points, float)
    dz = array[0, 0] - array[-1, 0]
    return (array[0, 1:] - array[-1, 1:]) / (dz if dz else 1)


def bridge_weak_gaps(
    seg: np.ndarray,
    fgmax: np.ndarray,
    *,
    max_gap: int = WEAK_MAX_GAP,
    cal_ratio: float = CAL_RATIO,
    min_iou: float = WEAK_MIN_IOU,
    weak_lo: float = WEAK_LO,
    min_size: int = WEAK_MIN_SIZE,
    dim_tol: int = DIM_TOL,
    margin: float = WEAK_MARGIN,
    rounds: int = WEAK_ROUNDS,
    inplace: bool = False,
    verbose: bool = False,
    stats: tuple[Any, Any, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Bridge weak gaps by projected-mask IoU, mutual choice, and margin."""
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    z_size, y_size, x_size = seg.shape
    del y_size, x_size
    total = 0
    for round_index in range(rounds):
        if round_index == 0 and stats is not None:
            bounds, sizes, _ = stats
        else:
            bounds, sizes, _ = seg_stats(seg)
        candidates = [label for label in bounds if label > 0 and sizes[label] >= min_size]
        up: dict[int, int] = {}
        down: dict[int, int] = {}
        for first in candidates:
            caliber = sizes[first] / (bounds[first][1] - bounds[first][0] + 1)
            for z_end, direction, store in (
                (bounds[first][1], +1, up),
                (bounds[first][0], -1, down),
            ):
                mask = seg[z_end] == first
                if not mask.any():
                    continue
                velocity = _weak_velocity(seg, first, z_end, direction, bounds)
                best = None
                second_best = 0.0
                dim_count = 0
                for gap in range(1, max_gap + 1):
                    z = z_end + direction * gap
                    if not (0 <= z < z_size):
                        break
                    dy, dx = (
                        int(round(float(velocity[0] * gap))),
                        int(round(float(velocity[1] * gap))),
                    )
                    projected = np.roll(np.roll(mask, dy, 0), dx, 1)
                    foreground_values = fgmax[z][projected]
                    if (
                        foreground_values.size
                        and float(np.percentile(foreground_values, 75)) < weak_lo
                    ):
                        dim_count += 1
                        if dim_count > dim_tol:
                            break
                    else:
                        dim_count = 0
                    neighbors = seg[z][projected]
                    for second in np.unique(neighbors[neighbors > 0]).tolist():
                        if second == first:
                            continue
                        second_mask = seg[z] == second
                        iou = int((projected & second_mask).sum()) / max(
                            int((projected | second_mask).sum()), 1
                        )
                        second_caliber = sizes[second] / (bounds[second][1] - bounds[second][0] + 1)
                        if not (1 / cal_ratio <= caliber / max(second_caliber, 1e-6) <= cal_ratio):
                            continue
                        if iou >= min_iou:
                            if best is None or iou > best[0]:
                                second_best = best[0] if best else second_best
                                best = (iou, int(second))
                            elif iou > second_best:
                                second_best = iou
                    if best is not None:
                        break
                if best is not None and best[0] - second_best >= margin:
                    store[first] = best[1]
        parent: np.ndarray = np.arange(int(seg.max()) + 1, dtype=np.int64)
        merge_count = 0
        for first in candidates:
            second = up.get(first)
            if second is not None and down.get(second) == first:
                first_root, second_root = _find(parent, first), _find(parent, second)
                if first_root != second_root:
                    parent[first_root] = second_root
                    merge_count += 1
        if verbose:
            print(
                f"  round {round_index}: {merge_count} weak-gap bridges",
                flush=True,
            )
        if merge_count == 0:
            break
        root = np.array(
            [_find(parent, index) for index in range(len(parent))],
            dtype=np.uint32,
        )
        apply_lut(seg, root)
        total += merge_count
    return seg, total


def _completion_velocity(
    seg: np.ndarray,
    label: int,
    z_end: int,
    direction: int,
    bounds: dict[int, tuple[int, int, int, int, int, int]],
    n: int = 6,
) -> np.ndarray:
    points = []
    for k in range(n):
        z = z_end - direction * k
        if bounds[label][0] <= z <= bounds[label][1] and (seg[z] == label).any():
            points.append((z, *center_of_mass(seg[z] == label)))
    if len(points) < 2:
        return np.zeros(2)
    array = np.array(points, float)
    dz = array[0, 0] - array[-1, 0]
    return (array[0, 1:] - array[-1, 1:]) / (dz if dz else 1)


def _completion_ends(
    seg: np.ndarray,
    label: int,
    bounds: dict[int, tuple[int, int, int, int, int, int]],
    z_size: int,
    y_size: int,
    x_size: int,
    border: int = BORDER,
) -> dict[int, tuple[int, np.ndarray, bool]]:
    output = {}
    for target_z, direction in (
        (bounds[label][0], -1),
        (bounds[label][1], +1),
    ):
        mask = seg[target_z] == label
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        at_border = (
            target_z <= border
            or target_z >= z_size - 1 - border
            or ys.min() <= border
            or ys.max() >= y_size - 1 - border
            or xs.min() <= border
            or xs.max() >= x_size - 1 - border
        )
        output[direction] = (
            target_z,
            np.array(center_of_mass(mask)),
            bool(at_border),
        )
    return output


def completion_radius_link(
    seg: np.ndarray,
    *,
    span_frac: float = SPAN_FRAC,
    min_size: int = COMPLETE_MIN_SIZE,
    max_gap: int = COMPLETE_MAX_GAP,
    max_lat: float = MAX_LAT,
    cal_ratio: float = COMPLETE_CAL_RATIO,
    margin: float = COMPLETE_MARGIN,
    cos_min: float = COS_MIN,
    prefer_length: bool = False,
    inplace: bool = False,
    verbose: bool = False,
    stats: tuple[Any, Any, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Optionally link incomplete open ends by radius; this may reduce OM."""
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    z_size, y_size, x_size = seg.shape
    if stats is None:
        bounds, sizes, _ = seg_stats(seg)
    else:
        bounds, sizes, _ = stats
    decent = [
        label
        for label in bounds
        if sizes[label] >= min_size
        and (bounds[label][1] - bounds[label][0] + 1) >= span_frac * z_size
    ]
    info = {label: _completion_ends(seg, label, bounds, z_size, y_size, x_size) for label in decent}
    caliber = {label: sizes[label] / (bounds[label][1] - bounds[label][0] + 1) for label in decent}
    open_count = {label: sum(1 for end in info[label].values() if not end[2]) for label in decent}
    open_ends = [
        (label, direction)
        for label in decent
        for direction in info[label]
        if not info[label][direction][2] and (prefer_length or open_count[label] >= 1)
    ]
    if verbose:
        print(
            f"  v4: {len(decent)} decent axons, " f"{len(open_ends)} open (non-border) ends",
            flush=True,
        )
    proposals: dict[tuple[int, int], tuple[int, int, float]] = {}
    for label, direction in open_ends:
        target_z, centroid, _ = info[label][direction]
        velocity = _completion_velocity(seg, label, target_z, direction, bounds)
        candidates = []
        for partner, partner_direction in open_ends:
            if partner == label or partner_direction == direction:
                continue
            partner_z, partner_centroid, _ = info[partner][partner_direction]
            gap = (partner_z - target_z) * direction
            if not (0 < gap <= max_gap):
                continue
            lateral = float(np.hypot(*(partner_centroid - (centroid + velocity * gap))))
            if lateral > max_lat:
                continue
            ratio = caliber[label] / max(caliber[partner], 1e-6)
            if not (1 / cal_ratio <= ratio <= cal_ratio):
                continue
            partner_velocity = _completion_velocity(
                seg,
                partner,
                partner_z,
                partner_direction,
                bounds,
            )
            velocity_norm = np.hypot(*velocity)
            partner_norm = np.hypot(*partner_velocity)
            if (
                velocity_norm > 0.5
                and partner_norm > 0.5
                and float(velocity @ -partner_velocity) / (velocity_norm * partner_norm) < cos_min
            ):
                continue
            candidates.append((lateral, partner, partner_direction))
        if not candidates:
            continue
        candidates.sort()
        if (
            len(candidates) > 1
            and (candidates[1][0] - candidates[0][0]) < margin
            and not prefer_length
        ):
            continue
        proposals[(label, direction)] = (
            candidates[0][1],
            candidates[0][2],
            candidates[0][0],
        )
    parent: np.ndarray = np.arange(int(seg.max()) + 1, dtype=np.int64)
    merge_count = 0
    for (label, direction), (
        partner,
        partner_direction,
        lateral,
    ) in proposals.items():
        back = proposals.get((partner, partner_direction))
        if back is None or back[0] != label:
            continue
        label_root, partner_root = _find(parent, label), _find(parent, partner)
        if label_root != partner_root:
            parent[label_root] = partner_root
            merge_count += 1
            if verbose:
                print(
                    f"    link {label} <-> {partner} (lat {lateral:.0f}px)",
                    flush=True,
                )
    if merge_count:
        root = np.array(
            [_find(parent, index) for index in range(len(parent))],
            dtype=np.uint32,
        )
        apply_lut(seg, root)
    return seg, merge_count


def branch_merge(
    affinities: np.ndarray,
    seg: np.ndarray,
    *,
    prefer_length: bool = False,
    stats: tuple[Any, Any, Any] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Run completion, mutual-IoU merge, and weak-gap bridging in fixed order.

    ``prefer_length=True`` additionally runs the completion-driven radius link.
    That optional research stage is known to reduce the oracle-merge ceiling and
    is deliberately disabled by default.
    """
    affinities = np.asarray(affinities)
    seg = np.asarray(seg)
    if affinities.ndim != 4 or affinities.shape[0] < 3:
        raise ValueError(
            "branch_merge affinities must be CZYX with at least 3 channels, "
            f"got {affinities.shape}."
        )
    if seg.ndim != 3:
        raise ValueError(f"branch_merge segmentation must be ZYX, got {seg.shape}.")
    if tuple(affinities.shape[1:]) != tuple(seg.shape):
        raise ValueError(
            "branch_merge affinity/segmentation spatial shapes differ: "
            f"{affinities.shape[1:]} vs {seg.shape}."
        )

    afz = affinities[0]
    fgmax = affinities[:3].max(0)
    merged, _ = merge_sections(
        seg,
        afz,
        inplace=inplace,
        verbose=verbose,
        stats=stats,
    )
    merged, _ = bridge_weak_gaps(
        merged,
        fgmax,
        inplace=True,
        verbose=verbose,
    )
    if prefer_length:
        merged, _ = completion_radius_link(
            merged,
            prefer_length=True,
            inplace=True,
            verbose=verbose,
        )
    return merged
