"""Two-sided tube extension for the staged branch decode.

``branch_merge`` is deliberately conservative: a seam is merged only when both
endpoints pick each other *and* the winner beats the runner-up by ``margin``.
That protects the false-merge-free ceiling, but on a volume where most tubes
enter and leave through the faces it leaves many decent tubes truncated with a
single border end -- the GT-free tube report calls these INCOMPLETE.

This stage takes the opposite trade for the tubes only. It selects the decent
tubes (the same "large and long" subset the tube report scores), then
propagates each one outward from *both* of its z-ends, one step at a time,
absorbing the piece that best continues the cross-section until the tube
reaches a volume face or the evidence runs out.

Two-sided means both ends of the tube are walked: ``+z`` from its top slice and
``-z`` from its bottom slice. An end that already touches a face is left alone,
so the work goes exactly where the completeness metric says it is missing.

===============================================================================
WHAT THIS TRADES AWAY
===============================================================================
Cue 3 of the ``branch_merge`` ladder (MUTUAL AGREEMENT) is the safety property
being spent here. Selecting a continuation one-sidedly is what chains adjacent
neurons, so this stage is NOT oracle-merge-safe and must not be used as a
general-purpose merge: it is a length-first operator for tube-shaped volumes,
and every join it makes is a claim that the *shape* continues.

What is kept from that ladder:

  - SHAPE CONTINUITY (cue 1) still selects: the step is chosen by real
    cross-section IoU against the propagated mask, never by affinity.
  - UNAMBIGUITY (cue 4) is available via ``margin`` but defaults to 0.0, since
    the point of the stage is to extend as far as possible.
  - ENDPOINT RESTRICTION (cue 5) is structural: only a tube's own z-extremes
    are ever walked, never its interior.
  - Z-AFFINITY (cue 6) and CALIBER (cue 7) act as vetoes only, exactly as
    upstream: ``aff_lo`` excludes background seams, ``cal_ratio`` refuses a
    partner of wildly different thickness.
  - ``require_mutual=True`` restores cue 3 and turns this into a focused,
    merge-safe completion pass at the cost of reach.

Growth is into *labelled* neighbours only. Background is not claimed: on this
data background is ~3% of the volume and is mostly membrane, so dilating into
it buys little length and leaks across boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from ....data.processing.bbox import apply_lut, seg_stats
from .merge import _find, _weak_velocity

__all__ = ["branch_extend"]


# Seed selection mirrors the tube report's "decent" subset (large and long).
MIN_SIZE, MIN_SPAN_FRAC = 5000, 0.25
# Step acceptance. MIN_IOU is below branch_merge's 0.45: the pieces being picked
# up here are the ones that pass were unable to claim.
MIN_IOU, MARGIN, MIN_OV = 0.15, 0.0, 20
AFF_LO, WEAK_LO, CAL_RATIO = 0.3, 0.3, 2.5
# Reach. MAX_GAP > 1 lets a step skip slices where the tube has no label at all.
MAX_GAP, MAX_STEPS, ROUNDS, BORDER = 5, 64, 2, 2


@dataclass(frozen=True)
class _ExtendParams:
    """Resolved knobs for one :func:`branch_extend` call."""

    min_iou: float
    margin: float
    min_ov: int
    aff_lo: float
    weak_lo: float
    cal_ratio: float
    max_gap: int
    max_steps: int
    border: int
    require_mutual: bool
    absorb_tubes: bool


def _end_at_border(mask: np.ndarray, z: int, shape: tuple[int, ...], border: int) -> bool:
    """Whether a tube end already reaches a volume face.

    Mirrors ``metrics.tube._border_end_count``: an end counts as a border end
    when its slice is at the z faces or its cross-section touches y/x faces, so
    "still open" here means exactly what INCOMPLETE means in the report.
    """
    if z <= border or z >= shape[0] - 1 - border:
        return True
    ys, xs = np.where(mask)
    if not len(ys):
        return True
    return bool(
        ys.min() <= border
        or ys.max() >= shape[1] - 1 - border
        or xs.min() <= border
        or xs.max() >= shape[2] - 1 - border
    )


def _caliber(label: int, bounds: dict, sizes: Any) -> float:
    z0, z1 = bounds[label][0], bounds[label][1]
    return float(sizes[label]) / max(z1 - z0 + 1, 1)


def _best_partner_iou(seg: np.ndarray, mask: np.ndarray, z: int) -> int:
    """Label on slice ``z`` with the highest IoU against ``mask`` (0 if none)."""
    neighbors = seg[z][mask]
    best_label, best_iou = 0, 0.0
    for label in np.unique(neighbors[neighbors > 0]).tolist():
        other = seg[z] == label
        iou = int((mask & other).sum()) / max(int((mask | other).sum()), 1)
        if iou > best_iou:
            best_label, best_iou = int(label), iou
    return best_label


def _step(
    seg: np.ndarray,
    afz: np.ndarray,
    fgmax: np.ndarray,
    *,
    current: int,
    z_end: int,
    direction: int,
    bounds: dict,
    sizes: Any,
    parent: np.ndarray,
    claimed: set[int],
    seeds: set[int],
    params: _ExtendParams,
) -> int | None:
    """Return the label continuing ``current`` past ``z_end``, or None."""
    z_size = seg.shape[0]
    mask = seg[z_end] == current
    if not mask.any():
        return None
    caliber = _caliber(current, bounds, sizes)
    velocity = _weak_velocity(seg, current, z_end, direction, bounds)
    root = _find(parent, current)

    for gap in range(1, params.max_gap + 1):
        z = z_end + direction * gap
        if not (0 <= z < z_size):
            return None
        if gap == 1:
            projected = mask
        else:
            dy = int(round(float(velocity[0] * (gap - 1))))
            dx = int(round(float(velocity[1] * (gap - 1))))
            projected = np.roll(np.roll(mask, dy, 0), dx, 1)
            # Only reach across a slice the tube plausibly passes through.
            foreground = fgmax[z][projected]
            if foreground.size and float(np.percentile(foreground, 75)) < params.weak_lo:
                continue
        dilated = binary_dilation(projected, iterations=1)
        neighbors = seg[z][dilated]
        matches: list[tuple[float, int]] = []
        for label in np.unique(neighbors[neighbors > 0]).tolist():
            label = int(label)
            if label in claimed or _find(parent, label) == root:
                continue
            if not params.absorb_tubes and label in seeds:
                continue
            other = seg[z] == label
            overlap = dilated & other
            if int(overlap.sum()) < params.min_ov:
                continue
            iou = int((projected & other).sum()) / max(int((projected | other).sum()), 1)
            if iou < params.min_iou:
                continue
            ratio = caliber / max(_caliber(label, bounds, sizes), 1e-6)
            if not 1 / params.cal_ratio <= ratio <= params.cal_ratio:
                continue
            if gap == 1:
                seam = afz[max(z_end, z)][overlap]
                if seam.size and float(seam.mean()) <= params.aff_lo:
                    continue
            matches.append((iou, label))
        if not matches:
            continue
        matches.sort(key=lambda match: -match[0])
        if len(matches) > 1 and matches[0][0] - matches[1][0] < params.margin:
            return None
        winner = matches[0][1]
        if params.require_mutual:
            back = seg[z] == winner
            if gap > 1:
                dy = int(round(float(-velocity[0] * (gap - 1))))
                dx = int(round(float(-velocity[1] * (gap - 1))))
                back = np.roll(np.roll(back, dy, 0), dx, 1)
            if _best_partner_iou(seg, back, z_end) != current:
                return None
        return winner
    return None


def _extend_end(
    seg: np.ndarray,
    afz: np.ndarray,
    fgmax: np.ndarray,
    tube: int,
    direction: int,
    *,
    bounds: dict,
    sizes: Any,
    parent: np.ndarray,
    claimed: set[int],
    seeds: set[int],
    params: _ExtendParams,
) -> int:
    """Walk one end of ``tube`` outward; return how many pieces it absorbed."""
    current = tube
    z_end = bounds[tube][1] if direction > 0 else bounds[tube][0]
    absorbed = 0
    for _ in range(params.max_steps):
        mask = seg[z_end] == current
        if not mask.any() or _end_at_border(mask, z_end, seg.shape, params.border):
            break
        winner = _step(
            seg,
            afz,
            fgmax,
            current=current,
            z_end=z_end,
            direction=direction,
            bounds=bounds,
            sizes=sizes,
            parent=parent,
            claimed=claimed,
            seeds=seeds,
            params=params,
        )
        if winner is None:
            break
        winner_root, tube_root = _find(parent, winner), _find(parent, tube)
        if winner_root != tube_root:
            parent[winner_root] = tube_root
        claimed.add(winner)
        absorbed += 1
        # Propagate: the tube now ends where the absorbed piece ends.
        current = winner
        z_end = bounds[winner][1] if direction > 0 else bounds[winner][0]
    return absorbed


def extend_tubes(
    seg: np.ndarray,
    afz: np.ndarray,
    fgmax: np.ndarray,
    *,
    min_size: int = MIN_SIZE,
    min_span_frac: float = MIN_SPAN_FRAC,
    min_iou: float = MIN_IOU,
    margin: float = MARGIN,
    min_ov: int = MIN_OV,
    aff_lo: float = AFF_LO,
    weak_lo: float = WEAK_LO,
    cal_ratio: float = CAL_RATIO,
    max_gap: int = MAX_GAP,
    max_steps: int = MAX_STEPS,
    rounds: int = ROUNDS,
    border: int = BORDER,
    require_mutual: bool = False,
    absorb_tubes: bool = True,
    inplace: bool = False,
    verbose: bool = False,
    stats: tuple[Any, Any, Any] | None = None,
) -> tuple[np.ndarray, int]:
    """Extend decent tubes from both z-ends; return ``(seg, absorbed)``."""
    if not inplace:
        seg = np.array(seg, dtype=np.uint32, copy=True)
    params = _ExtendParams(
        min_iou=min_iou,
        margin=margin,
        min_ov=min_ov,
        aff_lo=aff_lo,
        weak_lo=weak_lo,
        cal_ratio=cal_ratio,
        max_gap=max_gap,
        max_steps=max_steps,
        border=border,
        require_mutual=require_mutual,
        absorb_tubes=absorb_tubes,
    )
    z_size = seg.shape[0]
    min_span = ceil(min_span_frac * z_size)
    total = 0
    for round_index in range(rounds):
        if round_index == 0 and stats is not None:
            bounds, sizes, _ = stats
        else:
            bounds, sizes, _ = seg_stats(seg)
        seeds = [
            label
            for label in bounds
            if label > 0
            and int(sizes[label]) >= min_size
            and bounds[label][1] - bounds[label][0] + 1 >= min_span
        ]
        # Largest first: a long tube gets to claim its continuation before a
        # shorter neighbour can, which makes the outcome order-independent of
        # dict iteration order.
        seeds.sort(key=lambda label: -int(sizes[label]))
        seed_set = set(seeds)
        parent: np.ndarray = np.arange(int(seg.max()) + 1, dtype=np.int64)
        claimed: set[int] = set()
        absorbed = 0
        for tube in seeds:
            for direction in (+1, -1):
                absorbed += _extend_end(
                    seg,
                    afz,
                    fgmax,
                    tube,
                    direction,
                    bounds=bounds,
                    sizes=sizes,
                    parent=parent,
                    claimed=claimed,
                    seeds=seed_set,
                    params=params,
                )
        if verbose:
            print(
                f"  round {round_index}: {len(seeds)} tubes, {absorbed} pieces absorbed",
                flush=True,
            )
        if absorbed == 0:
            break
        root = np.array([_find(parent, index) for index in range(len(parent))], dtype=np.uint32)
        apply_lut(seg, root)
        total += absorbed
    return seg, total


def branch_extend(
    affinities: np.ndarray,
    seg: np.ndarray,
    *,
    min_size: int = MIN_SIZE,
    min_span_frac: float = MIN_SPAN_FRAC,
    min_iou: float = MIN_IOU,
    margin: float = MARGIN,
    min_ov: int = MIN_OV,
    aff_lo: float = AFF_LO,
    weak_lo: float = WEAK_LO,
    cal_ratio: float = CAL_RATIO,
    max_gap: int = MAX_GAP,
    max_steps: int = MAX_STEPS,
    rounds: int = ROUNDS,
    border: int = BORDER,
    require_mutual: bool = False,
    absorb_tubes: bool = True,
    stats: tuple[Any, Any, Any] | None = None,
    inplace: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Extend the decent tubes outward from both z-ends, as far as evidence allows.

    Args:
        min_size / min_span_frac: which labels count as tubes to extend. The
            defaults mirror the tube report's "decent" subset, so the stage
            works on exactly the instances whose completeness is scored.
        min_iou: cross-section IoU a piece must reach to continue the tube.
            This is the selector; lower it to reach further.
        margin: if set, the best continuation must beat the runner-up by this
            IoU gap or the walk stops. 0.0 (default) never declines on ambiguity.
        max_gap: how many slices a single step may skip. ``1`` restricts growth
            to touching seams; larger values cross slices where the tube was
            dropped entirely, using the velocity-projected cross-section.
        max_steps: cap on pieces absorbed per end, per round.
        border: distance from a volume face that already counts as an end, so
            ends that are complete are never walked.
        require_mutual: restore the merge-safe mutual-choice rule; the piece
            must also pick this tube back. Fewer, safer extensions.
        absorb_tubes: allow a tube to absorb another decent tube (this is how
            two halves of one axon get rejoined). Set False to grow only into
            fragments.

    Returns:
        The relabelled segmentation.
    """
    affinities = np.asarray(affinities)
    seg = np.asarray(seg)
    if affinities.ndim != 4 or affinities.shape[0] < 3:
        raise ValueError(
            "branch_extend affinities must be CZYX with at least 3 channels, "
            f"got {affinities.shape}."
        )
    if seg.ndim != 3:
        raise ValueError(f"branch_extend segmentation must be ZYX, got {seg.shape}.")
    if tuple(affinities.shape[1:]) != tuple(seg.shape):
        raise ValueError(
            "branch_extend affinity/segmentation spatial shapes differ: "
            f"{affinities.shape[1:]} vs {seg.shape}."
        )

    extended, _ = extend_tubes(
        seg,
        affinities[0],
        affinities[:3].max(0),
        min_size=min_size,
        min_span_frac=min_span_frac,
        min_iou=min_iou,
        margin=margin,
        min_ov=min_ov,
        aff_lo=aff_lo,
        weak_lo=weak_lo,
        cal_ratio=cal_ratio,
        max_gap=max_gap,
        max_steps=max_steps,
        rounds=rounds,
        border=border,
        require_mutual=require_mutual,
        absorb_tubes=absorb_tubes,
        inplace=inplace,
        verbose=verbose,
        stats=stats,
    )
    return extended
