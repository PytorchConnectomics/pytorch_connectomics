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
    from waterz import dust_merge_from_region_graph, merge_function_to_scoring

    WATERZ_AVAILABLE = True
except ImportError:
    WATERZ_AVAILABLE = False


def decode_waterz(
    predictions: np.ndarray,
    thresholds: Union[float, Sequence[float]] = 0.3,
    merge_function: str = "aff50_his256",
    aff_threshold: Tuple[float, float] = (0.0001, 0.9999),
    channel_order: str = "xyz",
    edge_offset: int = 1,
    use_aff_uint8: bool = False,
    use_seg_uint32: bool = False,
    compute_fragments: bool = False,
    seed_method: str = "maxima_distance",
    fragments: Optional[np.ndarray] = None,
    boundary_threshold: float = 0.0,
    min_instance_size: int = 0,
    dust_merge: bool = True,
    dust_merge_size: int = 0,
    dust_merge_affinity: float = 0.0,
    dust_remove_size: int = 0,
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
        edge_offset: Index of edge ``v ↔ v+1`` along each axis, relative to
            source voxel ``v``. ``1`` = destination-index (waterz / zwatershed
            / abiss / ``affinity_mode=deepem``; aff stored at ``v+1``).
            ``0`` = source-index (BANIS's ``comp_affinities`` /
            ``affinity_mode=banis``; aff stored at ``v``). When ``0``, each
            channel is rolled by +1 along its corresponding spatial axis
            after channel reorder, converting BANIS source-stored affinities
            to waterz-native destination-stored layout. Default: ``1``.
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

    branch_merge_requested = bool(kwargs.pop("branch_merge", False))
    branch_merge_keys = {
        "iou_threshold",
        "best_buddy",
        "one_sided_threshold",
        "one_sided_min_size",
        "affinity_threshold",
    }
    for key in branch_merge_keys:
        kwargs.pop(key, None)
    if branch_merge_requested:
        raise ValueError(
            "decode_waterz no longer runs branch_merge. Add a separate "
            "decoding step with name: branch_merge."
        )

    from waterz._uint8 import prepare_affinities, scale_aff_threshold, scale_thresholds

    # Prepare affinities: dtype normalisation, channel reorder, contiguous
    affs, is_uint8 = prepare_affinities(
        predictions,
        channel_order=channel_order,
        use_aff_uint8=use_aff_uint8,
    )

    # Convert source-stored (BANIS) -> destination-stored (waterz native).
    # affs has shape (3, Z, Y, X); channel c maps to spatial axis c of the
    # (Z, Y, X) volume, so rolling channel c by +1 along its corresponding
    # spatial axis lands the edge value at the destination voxel v+1.
    if int(edge_offset) == 0:
        for c in range(3):
            rolled = np.roll(affs[c], shift=1, axis=c)
            boundary: List[Any] = [slice(None)] * 3
            boundary[c] = 0
            rolled[tuple(boundary)] = 0
            affs[c] = rolled
    elif int(edge_offset) != 1:
        raise ValueError(
            f"edge_offset must be 0 (source/BANIS) or 1 (destination/waterz), got {edge_offset}."
        )

    # Keep public result keys in the caller-provided threshold scale even when
    # uint8 execution requires scaled internal waterz thresholds.
    threshold_keys = scale_thresholds(thresholds, is_uint8=False)

    # Scale float [0,1] parameters to [0,255] for uint8
    thresholds_list = scale_thresholds(thresholds, is_uint8)
    aff_low, aff_high = scale_aff_threshold(aff_threshold, is_uint8)

    # Convert shorthand merge function to C++ scoring function string
    scoring_function = merge_function_to_scoring(merge_function)

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
    waterz_kwargs["rescore_region_graph"] = False  # fast: use cached scores for dust merge

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
        if boundary_threshold > 0:
            n_removed = waterz.strip_boundary(
                seg, affs, threshold=boundary_threshold, channels="xy"
            )
            logger.info("boundary_threshold=%s: zeroed %d voxels", boundary_threshold, n_removed)

        # Size+affinity dust merge reusing the agglomeration's region graph
        # (accumulated histogram statistics, properly root-mapped IDs).
        if do_dust_merge:
            seg = seg.astype(np.uint64, copy=False)
            dust_merge_from_region_graph(
                seg,
                region_graph,
                is_uint8=is_uint8,
                size_th=dust_merge_size,
                weight_th=dust_merge_affinity,
                dust_th=dust_remove_size,
            )
        if min_instance_size > 0:
            from ..utils import remove_small_instances

            seg = remove_small_instances(seg, thres_small=min_instance_size, mode="background")
        processed.append(cast2dtype(seg))

    # Return results
    if return_all_thresholds and len(thresholds_list) > 1:
        return {round(t, 10): s for t, s in zip(threshold_keys, processed)}

    # Return last threshold result
    return processed[-1]
