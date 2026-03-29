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
    from waterz import merge_function_to_scoring, dust_merge_from_region_graph

    WATERZ_AVAILABLE = True
except ImportError:
    WATERZ_AVAILABLE = False


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
    iou_threshold: float = 0.5,
    best_buddy: bool = True,
    one_sided_threshold: float = 0.8,
    one_sided_min_size: int = 100,
    affinity_threshold: float = 0.0,
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
            Default: False
        iou_threshold: Full Jaccard IOU threshold for branch merge.
            Default: 0.5
        best_buddy: Enable mutual best-match merge.  Default: True
        one_sided_threshold: One-sided IOU threshold
            (``overlap / min(size0, size1)``).  0 to disable.  Default: 0.8
        one_sided_min_size: Minimum segment size in slice for one-sided
            merge.  Default: 100
        affinity_threshold: Minimum mean z-boundary affinity for a merge.
            0 to disable.  Default: 0.0
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
    scoring_function = merge_function_to_scoring(merge_function)

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

        # Strip weak-boundary voxels so dust merge sees true core sizes.
        if border_threshold > 0:
            n_removed = waterz.strip_border(seg, affs, threshold=border_threshold, channels="xy")
            logger.info("border_threshold=%s: zeroed %d voxels", border_threshold, n_removed)

        # Size+affinity dust merge reusing the agglomeration's region graph
        # (accumulated histogram statistics, properly root-mapped IDs).
        if do_dust_merge:
            seg = seg.astype(np.uint64, copy=False)
            dust_merge_from_region_graph(
                seg, region_graph,
                is_uint8=is_uint8,
                size_th=dust_merge_size,
                weight_th=dust_merge_affinity,
                dust_th=dust_remove_size,
            )
        # Branch merge: resolve false splits via z-slice IOU analysis
        if branch_merge:
            from .branch_merge import branch_merge as _branch_merge

            n_before = len(np.unique(seg)) - (1 if 0 in seg else 0)
            logger.info("branch_merge: starting on %d segments", n_before)
            seg = _branch_merge(
                seg,
                affinities=affs,
                iou_threshold=iou_threshold,
                best_buddy=best_buddy,
                one_sided_threshold=one_sided_threshold,
                one_sided_min_size=one_sided_min_size,
                affinity_threshold=affinity_threshold,
                channel_order="zyx",  # already converted above
            )
            n_after = len(np.unique(seg)) - (1 if 0 in seg else 0)
            logger.info("branch_merge: %d -> %d segments", n_before, n_after)

        if min_instance_size > 0:
            from ..utils import remove_small_instances

            seg = remove_small_instances(seg, thres_small=min_instance_size, mode="background")
        processed.append(cast2dtype(seg))

    # Return results
    if return_all_thresholds and len(thresholds_list) > 1:
        return {round(t, 10): s for t, s in zip(thresholds_list, processed)}

    # Return last threshold result
    return processed[-1]
