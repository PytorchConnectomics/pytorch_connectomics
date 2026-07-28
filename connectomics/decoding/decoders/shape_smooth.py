"""Shape smoothing for tube-like instance segmentations.

An affinity watershed leaves two shape defects that are cheap to fix from
geometry alone, without going back to the affinity:

1. **Thin protrusions and hairline bridges.** A one-voxel neck can fuse two
   otherwise clean tubes. A multi-label morphological opening removes the neck,
   and a connected-components relabel then separates what the neck was holding
   together.

2. **Cross-section jumps.** Where a tube absorbs a neighbour, its per-slice
   area steps up and stays up for a run of slices. The step is an outlier
   against the label's own local median area, which makes it detectable without
   ground truth.

The stage runs, in order: ``fastmorph`` label opening -> ``cc3d`` relabel ->
cross-z outlier split.

===============================================================================
THE CROSS-Z SPLIT
===============================================================================
This reuses the boundary-placement idiom validated in ``branch.split``: propagate
the ACTUAL cross-section mask rather than a disk seed, and let a distance
watershed place the boundary.

At the first slice of an area-outlier run, the previous slice is what the tube
looked like before the jump. Eroding it gives a marker for the continuation; the
part of the current cross-section that this marker does not explain is the extra
region, and it seeds the second marker. A distance-transform watershed over the
current cross-section then partitions it into "the tube" and "what joined it".
The kept part becomes the reference for the next slice, so the carve tracks the
tube through the whole run instead of drifting.

Cue-wise this is the one-sided case (see ``branch.split``: TWO-SIDED ANCHORING
is preferred, ONE-SIDED CARVE is riskier), so the run is bounded by the outlier
detection and aborts as soon as the extra region stops being substantial.
Splitting cannot lower the false-merge-free ceiling, but a badly placed boundary
still leaks voxels, hence the conservative gates.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, median_filter
from skimage.segmentation import watershed

__all__ = ["label_opening", "split_area_outliers", "shape_smooth"]


# 0 = the 3x3x3 multi-label opening; spherical radii below 2.0 erode nothing.
OPEN_RADIUS = 0.0
CONNECTIVITY = 26
# Area-outlier detection, mirroring the tube report's bump test.
BUMP_RATIO, BUMP_WINDOW, BUMP_MIN_EXTRA = 0.5, 15, 100
# Carve gates.
ERODE_ITERATIONS, SPLIT_MIN_SIZE, SPLIT_MIN_SPAN, MAX_RUN = 1, 1000, 5, 64
ANCHOR_BORDER = 2


def label_opening(
    seg: np.ndarray,
    *,
    radius: float = OPEN_RADIUS,
    plane: str = "3d",
    anisotropy: tuple[float, float, float] | None = None,
    parallel: int = 0,
) -> np.ndarray:
    """Multi-label opening; labels erode and regrow without swapping identity.

    ``radius=0`` uses ``fastmorph.opening`` -- one 3x3x3 multi-label pass, which
    is the cheapest structuring element that actually removes a one-voxel neck.
    A positive radius switches to ``spherical_open``.

    Measured on this voxel lattice: ``spherical_open(radius=1.0)`` erodes zero
    voxels, so the whole stage silently becomes a no-op. Radii in that dead zone
    are rejected rather than accepted and ignored.
    """
    try:
        import fastmorph
    except ImportError as exc:  # pragma: no cover - environment/dependency error.
        raise ImportError("shape_smooth's label opening requires the fastmorph package") from exc

    if radius and radius < 2.0:
        raise ValueError(
            f"open_radius={radius} erodes nothing (spherical radii below 2.0 are a no-op "
            "on the voxel lattice). Use open_radius=0 for the 3x3x3 opening, or >= 2.0."
        )
    if plane not in ("2d", "3d"):
        raise ValueError(f"plane must be '2d' or '3d', got {plane!r}")
    if plane == "2d":
        # Per-slice 3x3 opening: touches only in-plane shape, so a tube is never
        # eroded along z. fastmorph takes 2D arrays directly -- passing a
        # (1, Y, X) volume instead erodes every voxel away.
        if radius:
            slices = [
                fastmorph.spherical_open(seg[z], radius=radius, parallel=parallel)
                for z in range(seg.shape[0])
            ]
        else:
            slices = [
                fastmorph.opening(seg[z], background_only=True, parallel=parallel)
                for z in range(seg.shape[0])
            ]
        opened = np.stack(slices)
    elif radius:
        opened = fastmorph.spherical_open(
            seg,
            radius=radius,
            parallel=parallel,
            anisotropy=anisotropy,
        )
    else:
        opened = fastmorph.opening(seg, background_only=True, parallel=parallel)
    # The dilate half of the opening grows labels into background, so clamp the
    # result to the original support: an opening must never add voxels. The
    # input dtype is preserved -- ABISS ids start at 2**56 and do not fit uint32.
    return np.where(seg > 0, opened, 0).astype(seg.dtype, copy=False)


def _area_profile(seg: np.ndarray, label: int, z0: int, z1: int) -> np.ndarray:
    return np.array([int((seg[z] == label).sum()) for z in range(z0, z1 + 1)], np.int64)


def _area_steps(
    profile: np.ndarray,
    *,
    ratio: float,
    window: int,
    min_extra: int,
) -> list[tuple[int, int]]:
    """Return ``(index, direction)`` for each outlier *change* in the area profile.

    A whole-run test ("is this slice above the local median?") cannot see a bump
    longer than about half its window, because the run then dominates its own
    baseline -- and a sustained false merge is exactly such a run. Detecting the
    *step* instead is scale-free: it fires once where the extra region appears
    and once where it leaves, whatever the run length.

    ``direction`` says which way to walk: ``+1`` means the extra region appears
    at ``index`` (anchor on ``index - 1``), ``-1`` means ``index`` is the last
    slice still carrying it (anchor on ``index + 1``).
    """
    if len(profile) < 3:
        return []
    level = median_filter(profile.astype(np.float64), size=window, mode="nearest")
    steps: list[tuple[int, int]] = []
    for index in range(1, len(profile)):
        delta = float(profile[index] - profile[index - 1])
        base = max(float(min(level[index], level[index - 1])), 1.0)
        threshold = max(ratio * base, float(min_extra))
        if delta > threshold:
            steps.append((index, +1))
        elif -delta > threshold:
            steps.append((index - 1, -1))
    return steps


def _carve_slice(
    current: np.ndarray,
    seed: np.ndarray,
    *,
    min_extra: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split ``current`` into (kept, extra) with a 2-marker distance watershed."""
    keep_marker = current & binary_dilation(seed, iterations=1)
    extra_marker = current & ~binary_dilation(seed, iterations=3)
    keep_marker = keep_marker & ~extra_marker
    if not keep_marker.any() or int(extra_marker.sum()) < min_extra:
        return None
    markers = np.zeros(current.shape, np.int32)
    markers[extra_marker] = 2
    markers[keep_marker] = 1
    labels = watershed(-distance_transform_edt(current), markers, mask=current)
    kept = labels == 1
    extra = labels == 2
    if not kept.any() or int(extra.sum()) < min_extra:
        return None
    return kept, extra


def split_area_outliers(
    seg: np.ndarray,
    *,
    ratio: float = BUMP_RATIO,
    window: int = BUMP_WINDOW,
    min_extra: int = BUMP_MIN_EXTRA,
    erode_iterations: int = ERODE_ITERATIONS,
    min_size: int = SPLIT_MIN_SIZE,
    min_span: int = SPLIT_MIN_SPAN,
    max_run: int = MAX_RUN,
    anchor_border: int = ANCHOR_BORDER,
    inplace: bool = False,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """Carve the extra region out of every area-outlier run; return ``(seg, splits)``."""
    from connectomics.data.processing.bbox import seg_stats

    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    bounds, sizes, _ = seg_stats(seg)
    next_id = int(seg.max()) + 1
    splits = 0
    for label in sorted(bounds):
        if label <= 0 or int(sizes[label]) < min_size:
            continue
        z0, z1 = bounds[label][0], bounds[label][1]
        if z1 - z0 + 1 < min_span:
            continue
        profile = _area_profile(seg, label, z0, z1)
        carved_z: set[int] = set()
        for index, direction in _area_steps(
            profile,
            ratio=ratio,
            window=window,
            min_extra=min_extra,
        ):
            anchor = z0 + index - direction
            if not (z0 <= anchor <= z1) or profile[anchor - z0] <= 0:
                continue
            # A cross-section sitting on a volume z-face is truncated by the
            # face, so the step next to it is a boundary artifact, not a merge.
            # Measured on the APL crop: without this, 8 of 15 carves started at
            # z1 and one ran the full 63 slices -- the one-sided-carve drift.
            if anchor <= anchor_border or anchor >= seg.shape[0] - 1 - anchor_border:
                continue
            reference = seg[anchor] == label
            if not reference.any():
                continue
            new_id = next_id
            carved = 0
            truncated = False
            # Walk away from the anchor until the extra region stops being
            # substantial -- that is where the merged neighbour left.
            for step in range(max_run):
                z = z0 + index + direction * step
                if not (z0 <= z <= z1):
                    break
                if z in carved_z:
                    break
                current = seg[z] == label
                if not current.any():
                    break
                seed = binary_erosion(reference, iterations=erode_iterations)
                if not seed.any():
                    seed = reference
                result = _carve_slice(current, seed, min_extra=min_extra)
                if result is None:
                    break
                kept, extra = result
                seg[z][extra] = new_id
                carved_z.add(z)
                reference = kept
                carved += 1
            else:
                truncated = True
            if carved:
                next_id += 1
                splits += 1
                if verbose:
                    print(
                        f"  split {label} @z{z0 + index} dir{direction:+d}: "
                        f"{carved} slices carved into {new_id}"
                        f"{' (hit max_run)' if truncated else ''}",
                        flush=True,
                    )
                elif truncated:
                    print(
                        f"  split {label} @z{z0 + index}: stopped at max_run={max_run}",
                        flush=True,
                    )
    return seg, splits


def shape_smooth(
    seg: np.ndarray,
    *,
    open: bool = True,
    open_radius: float = OPEN_RADIUS,
    open_plane: str = "3d",
    anisotropy: tuple[float, float, float] | None = None,
    parallel: int = 0,
    connectivity: int = CONNECTIVITY,
    dust_size: int = 0,
    split: bool = True,
    bump_ratio: float = BUMP_RATIO,
    bump_window: int = BUMP_WINDOW,
    bump_min_extra: int = BUMP_MIN_EXTRA,
    erode_iterations: int = ERODE_ITERATIONS,
    split_min_size: int = SPLIT_MIN_SIZE,
    split_min_span: int = SPLIT_MIN_SPAN,
    max_run: int = MAX_RUN,
    anchor_border: int = ANCHOR_BORDER,
    verbose: bool = False,
) -> np.ndarray:
    """Open, relabel, and split area-outlier cross-sections.

    Args:
        open: run the label opening at all.
        open_radius: ``0`` uses the 3x3x3 multi-label opening, which is what
            removes a one-voxel neck; ``>= 2.0`` switches to a spherical
            structuring element. Radii in ``(0, 2)`` are rejected because they
            erode nothing on the voxel lattice.
        open_plane: ``3d`` opens with a 3x3x3 element; ``2d`` opens each z-slice
            independently, which never erodes a tube along z. Prefer ``2d`` when
            objects are thin and densely packed, where a 3D erosion attacks every
            label-label interface rather than just the necks.
        anisotropy: voxel size passed to the spherical opening.
        connectivity: ``cc3d`` connectivity for the relabel that follows the
            opening. This is what turns a removed neck into two instances.
        dust_size: drop components below this voxel count. ``0`` keeps all.
        split: run the cross-z outlier split.
        bump_ratio / bump_window / bump_min_extra: an area outlier is a
            slice-to-slice *step* exceeding both ``bump_ratio`` (relative to the
            local median area over ``bump_window`` slices) and
            ``bump_min_extra`` (absolute voxels).
        erode_iterations: erosion applied to the previous cross-section before
            it seeds the continuation marker.
        split_min_size / split_min_span: labels smaller or shorter than this are
            not examined.
        max_run: cap on slices carved from one step; reported when hit.
        anchor_border: refuse to carve when the anchor cross-section is this
            close to a volume z-face, where it is truncated rather than whole.

    Returns:
        The smoothed segmentation as ``uint32``.
    """
    import cc3d

    seg = np.asarray(seg)
    if seg.ndim != 3:
        raise ValueError(f"shape_smooth requires a ZYX segmentation, got {seg.shape}")
    if not np.issubdtype(seg.dtype, np.integer):
        raise TypeError(f"shape_smooth requires integer labels, got {seg.dtype}")

    # Not cast to uint32 yet: ABISS emits ids >= 2**56, which would wrap. The
    # cc3d relabel below is what makes the ids compact and small.
    out = seg
    if open:
        before = int((out > 0).sum())
        out = label_opening(
            out,
            radius=open_radius,
            plane=open_plane,
            anisotropy=anisotropy,
            parallel=parallel,
        )
        if verbose:
            removed = before - int((out > 0).sum())
            print(
                f"  opening {open_plane} r={open_radius or 'cube'}: {removed} voxels removed",
                flush=True,
            )

    out = cc3d.connected_components(out, connectivity=connectivity).astype(np.uint32, copy=False)
    if dust_size > 0:
        out = cc3d.dust(out, threshold=dust_size, connectivity=connectivity).astype(
            np.uint32, copy=False
        )
    if verbose:
        print(f"  cc3d({connectivity}): {len(np.unique(out)) - 1} labels", flush=True)

    if split:
        out, splits = split_area_outliers(
            out,
            ratio=bump_ratio,
            window=bump_window,
            min_extra=bump_min_extra,
            erode_iterations=erode_iterations,
            min_size=split_min_size,
            min_span=split_min_span,
            max_run=max_run,
            anchor_border=anchor_border,
            inplace=True,
            verbose=verbose,
        )
        if verbose:
            print(f"  area-outlier split: {splits} runs carved", flush=True)
    return out
