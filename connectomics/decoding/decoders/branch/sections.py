"""Build volume-unique 2D sections from three-channel affinities.

This is the packaged form of the validated MIT-LiCONN strong-section seed:
``decode_axon.build_strong_sections(..., small=0)`` and the exact reachable
``decode_lib.waterz_2d_spacefill`` watershed/agglomeration path.  The cache and
filesystem handling from the research script intentionally stay outside this
pure graph op.
"""

from __future__ import annotations

import os
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


def _section_slice(
    aff_slice: np.ndarray,
    mask_slice: np.ndarray,
    thr: float,
    small_size: int,
    aff_low: float,
    rg_zero: bool,
    score: str,
    aff_bg: float,
) -> np.ndarray:
    """Section one z-slice, returning locally renumbered ids (0 = background).

    Slices are independent: the watershed seeds restart at 1 for every slice, so
    nothing here depends on any other z. Only the volume-unique id offset is
    sequential, and that is applied by the caller.
    """
    from waterz import agglomerate

    a2: np.ndarray = aff_slice.astype(np.float32).copy()
    a2 *= mask_slice[None, None]
    tissue = mask_slice > 0
    if not tissue.any():
        return np.zeros(mask_slice.shape, np.uint32)
    frags = np.asarray(_watershed_concat(a2)).astype(np.uint64)
    if frags.max() == 0:
        return np.zeros(mask_slice.shape, np.uint32)
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
    return seg.astype(np.uint32)


_WORKER_STATE: dict = {}


def _worker_init(aff_chunk, mask_zyx, params) -> None:  # pragma: no cover - subprocess
    _WORKER_STATE["aff"] = aff_chunk
    _WORKER_STATE["mask"] = mask_zyx
    _WORKER_STATE["params"] = params


def _worker_slices(z_range):  # pragma: no cover - subprocess
    aff = _WORKER_STATE["aff"]
    mask = _WORKER_STATE["mask"]
    params = _WORKER_STATE["params"]
    return [(z, _section_slice(aff[:, z : z + 1], mask[z], *params)) for z in z_range]


def _resolve_workers(num_workers: int) -> int:
    """``-1`` means every CPU this process may actually use (cgroup-aware)."""
    if num_workers >= 0:
        return max(1, num_workers)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


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
    num_workers: int = 1,
) -> np.ndarray:
    """Exact ``decode_lib.waterz_2d_spacefill`` algorithm, optionally over N processes."""
    try:
        import waterz  # noqa: F401
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
    params = (thr, small_size, aff_low, rg_zero, score, aff_bg)

    workers = min(_resolve_workers(num_workers), z_size)
    if workers <= 1:
        local = (
            (z, _section_slice(aff_chunk[:, z : z + 1], mask_zyx[z], *params))
            for z in range(z_size)
        )
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # fork so the (large) affinity volume is shared copy-on-write rather than
        # pickled to every worker.
        batches = [range(i, min(i + 8, z_size)) for i in range(0, z_size, 8)]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("fork"),
            initializer=_worker_init,
            initargs=(aff_chunk, mask_zyx, params),
        ) as pool:
            local = [pair for batch in pool.map(_worker_slices, batches) for pair in batch]

    # The id offset stays sequential in z, so ids match the serial run exactly.
    max_id = 0
    for z, seg in local:
        if not seg.any():
            continue
        foreground = seg > 0
        seg = seg.copy()
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
    num_workers: int = 1,
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
        num_workers: processes to section slices with. Slices are independent,
            and the volume-unique id offset stays sequential, so any worker
            count gives bit-identical output. ``-1`` uses every available CPU.
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
        num_workers=num_workers,
    )
    return np.asarray(sections, dtype=np.uint32)
