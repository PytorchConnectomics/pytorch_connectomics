"""Waterz watershed + agglomeration decoder.

Converts 3-channel affinity predictions to instance segmentation using the
waterz library (watershed followed by hierarchical region agglomeration).

Requires: ``pip install -e lib/waterz``
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..utils import cast2dtype

__all__ = ["decode_waterz"]

logger = logging.getLogger(__name__)

try:
    import waterz

    WATERZ_AVAILABLE = True
except ImportError:
    WATERZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shorthand -> C++ scoring function conversion
# ---------------------------------------------------------------------------

_RG = "RegionGraphType"
_SV = "ScoreValue"


def _merge_function_to_scoring(shorthand: str) -> str:
    """Convert a shorthand merge function name to a C++ scoring type string.

    Supported shorthands (examples):
        aff50_his256  -> OneMinus<HistogramQuantileAffinity<RG, 50, SV, 256>>
        aff85_his256  -> OneMinus<HistogramQuantileAffinity<RG, 85, SV, 256>>
        aff50_his0    -> OneMinus<QuantileAffinity<RG, 50, SV>>
        max10         -> OneMinus<MeanMaxKAffinity<RG, 10, SV>>
        *_ran255      -> One255Minus<...> instead of OneMinus<...>
    """
    parts = {tok[:3]: tok[3:] for tok in shorthand.split("_")}
    use_255 = parts.get("ran") == "255"
    wrapper = "One255Minus" if use_255 else "OneMinus"

    if "aff" in parts:
        quantile = parts["aff"]
        his_bins = parts.get("his", "0")
        if his_bins and his_bins != "0":
            inner = f"HistogramQuantileAffinity<{_RG}, {quantile}, {_SV}, {his_bins}>"
        else:
            inner = f"QuantileAffinity<{_RG}, {quantile}, {_SV}>"
        return f"{wrapper}<{inner}>"

    if "max" in parts:
        k = parts["max"]
        inner = f"MeanMaxKAffinity<{_RG}, {k}, {_SV}>"
        return f"{wrapper}<{inner}>"

    # If it already looks like a C++ type string, pass through
    if "<" in shorthand:
        return shorthand

    raise ValueError(
        f"Unknown merge_function shorthand: '{shorthand}'. "
        "Expected format like 'aff50_his256', 'aff85_his256', 'max10', etc."
    )


def _build_segment_counts(seg: np.ndarray) -> np.ndarray:
    """Build a dense counts array indexed by segment id."""
    ids, cnts = np.unique(seg, return_counts=True)
    max_id = int(ids.max()) if len(ids) else 0
    counts = np.zeros(max_id + 1, dtype=np.uint64)
    counts[ids] = cnts
    return counts


def decode_waterz(
    predictions: np.ndarray,
    thresholds: Union[float, Sequence[float]] = 0.3,
    merge_function: str = "aff50_his256",
    aff_threshold: Tuple[float, float] = (0.0001, 0.9999),
    channel_order: str = "xyz",
    use_aff_uint8: bool = False,
    use_seg_uint32: bool = False,
    compute_fragments: bool = False,
    seed_method: str = "maxima_distance",
    fragments: Optional[np.ndarray] = None,
    border_threshold: float = 0.0,
    min_instance_size: int = 0,
    dust_merge: bool = True,
    dust_merge_size: int = 0,
    dust_merge_affinity: float = 0.0,
    dust_remove_size: int = 0,
    branch_merge: bool = False,
    branch_iou_threshold: float = 0.5,
    branch_best_buddy: bool = True,
    branch_one_sided_threshold: float = 0.8,
    branch_one_sided_min_size: int = 100,
    branch_affinity_threshold: float = 0.0,
    return_all_thresholds: bool = False,
    **kwargs: Any,
) -> "np.ndarray | Dict[float, np.ndarray]":
    r"""Convert affinity predictions to instance segmentation via waterz.

    Performs watershed on the affinity graph to produce an initial
    over-segmentation, then applies hierarchical region agglomeration using
    the specified scoring function and threshold(s).

    When multiple thresholds are provided, waterz performs watershed and
    region-graph extraction **once** and incrementally merges for each
    threshold — making multi-threshold evaluation nearly as fast as single.
    This is especially useful for Optuna parameter tuning.

    Args:
        predictions: Affinity predictions of shape :math:`(C, Z, Y, X)` where
            ``C >= 3``. The first 3 channels are short-range affinities.
            Supports **float32** [0, 1] or **uint8** [0, 255].  When uint8,
            the entire pipeline runs in integer arithmetic (4x less memory).
            Parameters (thresholds, aff_threshold) can be specified in float
            [0, 1] range — they are auto-scaled to [0, 255] for uint8.
        thresholds: Agglomeration threshold(s). Regions with merge score below
            the threshold are merged. Specify in [0, 1] float range
            regardless of input dtype (auto-scaled for uint8). Default: 0.3
        merge_function: Scoring function for agglomeration. Common options:

            - ``"aff50_his256"``: Median affinity via 256-bin histogram (default, recommended)
            - ``"aff85_his256"``: 85th percentile affinity
            - ``"aff75_his256"``: 75th percentile affinity
            - ``"aff50_his0"``: Exact median affinity (slower)
            - ``"max10"``: Mean of top-10 affinities

            Also accepts raw C++ type strings (e.g.
            ``"OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"``).
        aff_threshold: Tuple ``(low, high)`` for initial watershed. Affinities
            below *low* are background; above *high* are definite connections.
            Values are in [0, 1] float range. Default: ``(0.0001, 0.9999)``
        channel_order: Order of the first 3 affinity channels in the
            predictions. Waterz C++ expects ``"zyx"`` (channel 0=z, 1=y,
            2=x). If the model outputs affinities in ``"xyz"`` order
            (e.g. offsets ``["0-0-1", "0-1-0", "1-0-0"]``), set this to
            ``"xyz"`` and the channels will be transposed automatically.
            Default: ``"xyz"``
        use_aff_uint8: Convert float affinities to uint8 before waterz.
            Saves 4x memory and runs the entire C++ pipeline in integer
            arithmetic.  Lossless for ``HistogramQuantileAffinity`` with
            256 bins.  If input is already uint8, this is a no-op.
            Default: False
        fragments: Pre-computed over-segmentation (fragment IDs). If provided,
            the watershed step is skipped and agglomeration runs directly on
            these fragments. Shape :math:`(Z, Y, X)`, dtype uint64.
        min_instance_size: Minimum instance size in voxels. Instances smaller
            than this are removed (set to background). Set to 0 to disable.
            Default: 0
        dust_merge: Enable dust postprocessing.  Reuses the agglomeration's
            region graph (with accumulated histogram statistics) via
            ``waterz.merge_segments``.  When False, the dust merge and dust
            removal thresholds below are ignored. Default: True
        dust_merge_size: Size+affinity dust merge (zwatershed-style).
            Segments with fewer voxels than this are merged into their
            highest-affinity neighbor.  Unlike *min_instance_size* which
            drops dust to background, this **merges** dust into the correct
            parent using affinity evidence.  0 to disable. Default: 0
        dust_merge_affinity: Minimum affinity for a dust merge to be
            eligible.  Only edges with affinity > this value are considered.
            Default: 0.0
        dust_remove_size: After dust merge, remove any remaining segments
            smaller than this (e.g. isolated fragments with no neighbors
            to merge into).  0 to disable. Default: 0
        branch_merge: Enable branch merge postprocessing.  Resolves false
            splits by analyzing segment continuity across z-slices using
            IOU overlap, best-buddy matching, and one-sided IOU.
            Inspired by em_pipeline branch resolution.  Default: False
        branch_iou_threshold: Stage 1 threshold for full Jaccard IOU
            merge between consecutive z-slices.  Default: 0.5
        branch_best_buddy: Enable Stage 2 mutual best-match merge.
            Default: True
        branch_one_sided_threshold: Stage 3 one-sided IOU threshold.
            Merge if ``overlap / min(size0, size1)`` exceeds this.
            Set to 0 to disable.  Default: 0.8
        branch_one_sided_min_size: Stage 3 minimum segment size in the
            slice to be considered for one-sided merge.  Default: 100
        branch_affinity_threshold: Minimum mean z-boundary affinity for
            a branch merge to be accepted.  0 to disable.  Default: 0.0
        return_all_thresholds: If True and multiple thresholds are given,
            return a dict mapping each threshold to its segmentation.
            Otherwise return only the last threshold's result. Default: False
        **kwargs: Additional keyword arguments (silently ignored for YAML
            config compatibility).

    Returns:
        np.ndarray: Instance segmentation of shape :math:`(Z, Y, X)`.
            If *return_all_thresholds* is True and multiple thresholds are
            provided, returns ``Dict[float, np.ndarray]`` instead.

    Raises:
        ImportError: If the waterz library is not installed.
        ValueError: If predictions have wrong shape.

    Examples:
        >>> # Basic usage with single threshold
        >>> seg = decode_waterz(affinities, thresholds=0.3)

        >>> # Multiple thresholds, return all (efficient — watershed runs once)
        >>> results = decode_waterz(
        ...     affinities,
        ...     thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
        ...     return_all_thresholds=True,
        ... )
        >>> seg_at_0_3 = results[0.3]

        >>> # Custom merge function
        >>> seg = decode_waterz(
        ...     affinities,
        ...     thresholds=0.5,
        ...     merge_function='aff85_his256',
        ... )
    """
    if not WATERZ_AVAILABLE:
        raise ImportError(
            "waterz is not installed. Install it with:\n" "  pip install -e lib/waterz"
        )

    predictions = np.asarray(predictions)
    if predictions.ndim != 4:
        raise ValueError(
            f"Expected affinity predictions with shape (C, Z, Y, X), "
            f"got {predictions.ndim}D array with shape {predictions.shape}."
        )
    if predictions.shape[0] < 3:
        raise ValueError(f"Expected >= 3 affinity channels, got {predictions.shape[0]}.")

    # Use first 3 channels (short-range affinities)
    affs = predictions[:3]

    # Convert float → uint8 on the fly if requested (4x memory savings).
    # If already uint8, this is a no-op.
    if use_aff_uint8 and affs.dtype != np.uint8:
        affs = np.clip(affs, 0, 1)
        affs = (affs * 255).astype(np.uint8)

    # Detect uint8 affinities — pass through to waterz uint8 path directly.
    # For float dtypes, convert to float32.  Never convert uint8 to float.
    is_uint8 = affs.dtype == np.uint8
    if not is_uint8:
        affs = affs.astype(np.float32, copy=False)

    # Transpose channels to zyx order expected by waterz C++.
    # Waterz expects: channel 0=z, 1=y, 2=x.
    channel_order = channel_order.lower()
    if channel_order == "xyz":
        # Model outputs x,y,z → reverse to z,y,x
        affs = affs[[2, 1, 0]]
    elif channel_order == "zyx":
        pass  # Already in waterz order
    else:
        raise ValueError(f"Unknown channel_order '{channel_order}'. Expected 'xyz' or 'zyx'.")

    # Ensure C-contiguous for waterz
    if not affs.flags["C_CONTIGUOUS"]:
        affs = np.ascontiguousarray(affs)

    # Scale parameters for uint8: user specifies float [0,1], we map to [0,255].
    if is_uint8:
        _to_u8 = lambda v: int(round(v * 255)) if isinstance(v, float) and v <= 1.0 else int(v)
    else:
        _to_u8 = None  # unused

    # Normalize thresholds to sorted list (waterz requires ascending order)
    if isinstance(thresholds, (int, float)):
        thresholds_list = [float(thresholds)]
    else:
        thresholds_list = sorted(float(t) for t in thresholds)
    if is_uint8:
        thresholds_list = [_to_u8(t) for t in thresholds_list]

    # Convert shorthand merge function to C++ scoring function string
    scoring_function = _merge_function_to_scoring(merge_function)

    aff_low = _to_u8(aff_threshold[0]) if is_uint8 else float(aff_threshold[0])
    aff_high = _to_u8(aff_threshold[1]) if is_uint8 else float(aff_threshold[1])

    logger.info(
        "Running waterz: %d thresholds=%s, scoring_function=%s, aff_threshold=(%s, %s)%s",
        len(thresholds_list),
        thresholds_list,
        scoring_function,
        aff_low,
        aff_high,
        " [uint8]" if is_uint8 else "",
    )

    # Build kwargs for waterz.waterz()
    waterz_kwargs: Dict[str, Any] = dict(
        scoring_function=scoring_function,
        aff_threshold_low=aff_low,
        aff_threshold_high=aff_high,
        seg_dtype="uint32" if use_seg_uint32 else "uint64",
    )
    if fragments is not None:
        waterz_kwargs["fragments"] = fragments.astype(np.uint64, copy=False)
    elif compute_fragments:
        waterz_kwargs["compute_fragments"] = True
        waterz_kwargs["seed_method"] = seed_method

    do_dust_merge = bool(dust_merge) and dust_merge_size > 0
    waterz_kwargs["return_region_graph"] = do_dust_merge

    # waterz.waterz() runs watershed + region-graph once, then incrementally
    # merges for each threshold.  Returns all segmentations (copied).
    seg_list = waterz.waterz(affs, thresholds=thresholds_list, **waterz_kwargs)

    # Post-process each result
    processed: List[np.ndarray] = []
    for waterz_result in seg_list:
        if do_dust_merge:
            seg, region_graph = waterz_result
        else:
            seg = waterz_result

        # Zero out voxels where mean xy affinity is below threshold,
        # creating gaps at weak boundaries to separate segments.
        if border_threshold > 0:
            xy_mean = affs[1:3].astype(np.float32, copy=False).mean(axis=0)
            if is_uint8:
                xy_mean /= 255.0
            border_mask = xy_mean < border_threshold
            n_removed = int(border_mask.sum())
            seg[border_mask] = 0
            print(f"border_threshold={border_threshold}: zeroed {n_removed} voxels")

        # Size+affinity dust merge reusing the agglomeration's region graph
        # (accumulated histogram statistics, properly root-mapped IDs).
        # Invert OneMinus/One255Minus scores to raw affinities.
        if do_dust_merge:
            seg = seg.astype(np.uint64, copy=False)
            n_edges = len(region_graph)
            rg_affs = np.empty(n_edges, dtype=np.float32)
            id1 = np.empty(n_edges, dtype=np.uint64)
            id2 = np.empty(n_edges, dtype=np.uint64)
            score_max = 255.0 if is_uint8 else 1.0
            for idx, edge in enumerate(region_graph):
                rg_affs[idx] = score_max - float(edge["score"])
                id1[idx] = int(edge["u"])
                id2[idx] = int(edge["v"])
            if n_edges:
                np.clip(rg_affs, 0.0, score_max, out=rg_affs)
                order = np.argsort(rg_affs)[::-1]
                rg_affs = np.ascontiguousarray(rg_affs[order])
                id1 = np.ascontiguousarray(id1[order])
                id2 = np.ascontiguousarray(id2[order])
            counts = _build_segment_counts(seg)
            waterz.merge_segments(
                seg, rg_affs, id1, id2, counts,
                size_th=dust_merge_size,
                weight_th=dust_merge_affinity,
                dust_th=dust_remove_size,
            )
        # Branch merge: resolve false splits via z-slice IOU analysis
        if branch_merge:
            from .branch_merge import branch_merge as _branch_merge

            n_before = len(np.unique(seg)) - (1 if 0 in seg else 0)
            print(f"branch_merge: starting on {n_before} segments")
            seg = _branch_merge(
                seg,
                affinities=affs,
                iou_threshold=branch_iou_threshold,
                best_buddy=branch_best_buddy,
                one_sided_threshold=branch_one_sided_threshold,
                one_sided_min_size=branch_one_sided_min_size,
                affinity_threshold=branch_affinity_threshold,
                channel_order="zyx",  # already converted above
            )
            n_after = len(np.unique(seg)) - (1 if 0 in seg else 0)
            print(f"branch_merge: {n_before} -> {n_after} segments")

        if min_instance_size > 0:
            from ..utils import remove_small_instances

            seg = remove_small_instances(seg, thres_small=min_instance_size, mode="background")
        processed.append(cast2dtype(seg))

    # Return results
    if return_all_thresholds and len(thresholds_list) > 1:
        return {round(t, 10): s for t, s in zip(thresholds_list, processed)}

    # Return last threshold result
    return processed[-1]
