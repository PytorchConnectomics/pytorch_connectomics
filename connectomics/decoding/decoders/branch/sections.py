"""Build volume-unique 2D sections from three-channel affinities.

This is the packaged form of the validated MIT-LiCONN strong-section seed:
``decode_axon.build_strong_sections(..., small=0)`` and the exact reachable
``decode_lib.waterz_2d_spacefill`` watershed/agglomeration path.  The cache and
filesystem handling from the research script intentionally stay outside this
pure graph op.
"""

from __future__ import annotations

from typing import Optional

import fastremap
import mahotas
import numpy as np

__all__ = ["seg_2d"]


AFF_LOW = 0.01
AFF_BG = 0.66
RG_ZERO = False
SCORE_NAME = "aff30_his256_ran255"


def _get_score_func(score_name: str) -> Optional[str]:
    """Return the legacy waterz scoring type for *score_name*."""
    config = {x[:3]: x[3:] for x in score_name.split("_")}
    if "aff" in config:
        if "his" in config and config["his"] != "0":
            return "OneMinus<HistogramQuantileAffinity<RegionGraphType, %s, " "ScoreValue, %s>>" % (
                config["aff"],
                config["his"],
            )
        return "OneMinus<QuantileAffinity<RegionGraphType, " + config["aff"] + ", ScoreValue>>"
    if "max" in config:
        return "OneMinus<MeanMaxKAffinity<RegionGraphType, " + config["max"] + ", ScoreValue>>"
    return None


def _maxima_distance_seeds(boundary: np.ndarray, next_id: int = 1) -> tuple[np.ndarray, int]:
    """Exact reachable ``get_seeds(..., method="maxima_distance")`` path."""
    distance = mahotas.distance(boundary < 0.5)
    maxima = mahotas.regmax(distance)
    seeds, num_seeds = mahotas.label(maxima)
    seeds += next_id
    seeds[seeds == next_id] = 0
    return seeds, num_seeds


def _watershed_concat(affs: np.ndarray) -> np.ndarray:
    """Exact reachable one-slice-at-a-time legacy waterz watershed path."""
    fragments = np.zeros(affs[0].shape).astype(np.uint64)
    next_id = 1
    for z in range(affs.shape[1]):
        affs_z = affs[1:, z]
        boundary = 1 - affs_z.mean(axis=0)
        seeds, num_seeds = _maxima_distance_seeds(boundary, next_id=next_id)
        fragments[z] = mahotas.cwatershed(boundary, seeds)
        next_id += num_seeds
    return fragments


def _waterz_2d_spacefill(
    aff_chunk: np.ndarray,
    mask_zyx: np.ndarray,
    thr: float,
    small_size: int,
    aff_low: float,
    *,
    rg_zero: bool = True,
    score: Optional[str] = None,
    aff_bg: Optional[float] = None,
) -> np.ndarray:
    """Exact ``decode_lib.waterz_2d_spacefill`` algorithm."""
    try:
        from waterz import agglomerate
    except ImportError as exc:  # pragma: no cover - environment/dependency error.
        raise ImportError(
            "seg_2d requires the repository's waterz package to be installed"
        ) from exc

    if score is None:
        score = _get_score_func(SCORE_NAME)
    if aff_bg is None:
        aff_bg = aff_low
    z_size = aff_chunk.shape[1]
    seg_out = np.zeros(aff_chunk.shape[1:], np.uint32)
    max_id = 0
    for z in range(z_size):
        a2: np.ndarray = aff_chunk[:, z : z + 1].astype(np.float32).copy()
        a2 *= mask_zyx[z][None, None]
        tissue = mask_zyx[z] > 0
        if not tissue.any():
            continue
        frags = np.asarray(_watershed_concat(a2)).astype(np.uint64)
        if frags.max() == 0:
            continue
        a_rg = a2.copy()
        if rg_zero:
            a_rg[a_rg < aff_bg] = 0
        seg = np.asarray(
            next(
                iter(
                    agglomerate(
                        a_rg,
                        thresholds=[thr],
                        fragments=frags.copy(),
                        scoring_function=score,
                        discretize_queue=256,
                        aff_threshold_low=float(aff_low),
                        aff_threshold_high=0.98,
                    )
                )
            )
        )[0]
        fg = tissue & (a2.max(axis=0)[0] > aff_bg)
        seg[~fg] = 0
        seg = seg.astype(np.uint32)
        sizes = np.bincount(seg.ravel())
        small = np.where(sizes < small_size)[0]
        if small.size:
            seg[np.isin(seg, small)] = 0
        seg, _ = fastremap.renumber(seg, in_place=True)
        seg = seg.astype(np.uint32)
        foreground = seg > 0
        if foreground.any():
            seg[foreground] += max_id
            max_id = int(seg.max())
        seg_out[z] = seg
    return seg_out


def seg_2d(
    aff: np.ndarray,
    *,
    thr: float = 0.3,
    aff_bg: float = AFF_BG,
    aff_low: float = AFF_LOW,
    small: int = 0,
) -> np.ndarray:
    """Decode each z-slice independently and assign volume-unique section IDs.

    The validated recipe is the default of every argument below, with the
    ``aff30_his256_ran255`` waterz score.

    Args:
        thr: waterz agglomeration threshold within a slice.
        aff_bg: foreground band -- a voxel is section material only where the
            maximum affinity exceeds this. Lowering it admits the weak
            0.30-0.66 shell, which bridges touching parallel tubes; the
            weak-gap stage of :func:`branch_merge` crosses that band locally
            instead, which is measurably safer.
        aff_low: waterz ``aff_threshold_low``.
        small: waterz internal small-segment removal. The validated value is 0
            (keep every above-``aff_bg`` supervoxel): waterz's own default of
            150 silently removed 4.2% of confident skeleton coverage, whole
            thin tubes included.
    """
    aff = np.asarray(aff)
    if aff.ndim != 4 or aff.shape[0] != 3:
        raise ValueError(f"affinity must be CZYX with 3 channels, got {aff.shape}")

    expected = aff.shape[1:]
    sections = _waterz_2d_spacefill(
        aff,
        np.ones(expected, bool),
        thr,
        small,
        aff_low,
        rg_zero=RG_ZERO,
        score=_get_score_func(SCORE_NAME),
        aff_bg=aff_bg,
    )
    return np.asarray(sections, dtype=np.uint32)
