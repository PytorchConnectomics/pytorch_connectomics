"""Mutex Watershed decoder.

Converts affinity predictions with **both** short-range (attractive) and
long-range (repulsive) offsets into an instance segmentation without any
seeds, thresholds, or agglomeration step (Wolf et al., "The Mutex Watershed",
ECCV 2018 / TPAMI 2020).

Wraps ``affogato.segmentation.compute_mws_segmentation`` — the reference
implementation from the paper's authors — the same way :mod:`waterz.py` wraps
waterz.

Requires ``affogato`` (C++/xtensor, conda-forge only — the PyPI ``affogato``
is an unrelated project): ``conda install -c conda-forge affogato``
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...data.processing.affinity import parse_affinity_offsets
from ..utils import cast2dtype

__all__ = ["decode_mutex_watershed"]

logger = logging.getLogger(__name__)

try:
    from affogato.segmentation import compute_mws_segmentation

    AFFOGATO_AVAILABLE = True
except ImportError:
    AFFOGATO_AVAILABLE = False


# Standard 3D banis 6-channel layout: 3 short-range (attractive) nearest-
# neighbor edges followed by 3 long-range (repulsive) edges.
_DEFAULT_OFFSETS_6 = ["1-0-0", "0-1-0", "0-0-1", "10-0-0", "0-10-0", "0-0-10"]


def decode_mutex_watershed(
    predictions: np.ndarray,
    offsets: Optional[Sequence[Any]] = None,
    num_attractive: int = 3,
    strides: Optional[Sequence[int]] = None,
    randomize_strides: bool = True,
    invert_repulsive: bool = True,
    invert_attractive: bool = False,
    mask: Optional[np.ndarray] = None,
    min_instance_size: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    r"""Convert affinity predictions to instance segmentation via Mutex Watershed.

    The Mutex Watershed sorts *all* affinity edges (attractive + repulsive) by
    weight and greedily processes them with a union-find that also tracks
    mutual-exclusion (mutex) constraints: an attractive edge merges two regions
    unless a mutex forbids it; a repulsive edge adds a mutex. No threshold or
    seed is needed — segment boundaries emerge where repulsion dominates.

    Args:
        predictions: Affinity predictions of shape :math:`(C, Z, Y, X)`, float
            in ``[0, 1]``, where ``C == len(offsets)``. The first
            ``num_attractive`` channels are the short-range attractive edges;
            the remainder are long-range repulsive (mutex) edges.
        offsets: Edge offsets, one per channel, as ``"z-y-x"`` strings or
            length-3 sequences (negative components allowed as sequences, e.g.
            ``[-1, 0, 0]``). Defaults to the standard 6-channel banis layout
            (3 nearest-neighbor + 3 long-range at distance 10) when ``C == 6``.
        num_attractive: Number of leading channels treated as attractive
            (short-range) edges. Default: 3.
        strides: Optional per-axis subsampling of the *repulsive* edges (e.g.
            ``[1, 4, 4]``) — the standard speed/quality knob for MWS on dense
            long-range affinities. ``None`` uses every edge. Default: None.
        randomize_strides: Randomly sample strided repulsive edges rather than
            taking them on a fixed grid, reducing striding artifacts. Only used
            when ``strides`` is set. Default: True.
        invert_repulsive: PyTC networks predict affinity as ``P(same segment)``
            (high = merge) for *every* channel, including long-range ones. MWS
            needs repulsive edges where high = *split*, so the repulsive
            channels are inverted (``1 - aff``). Set False only if your model
            already predicts long-range repulsion/boundary directly.
            Default: True.
        invert_attractive: Invert the attractive channels too (for models that
            predict boundary probability instead of affinity). Default: False.
        mask: Optional boolean foreground mask of shape :math:`(Z, Y, X)`.
            Voxels outside the mask are left as background (label 0).
        min_instance_size: Remove instances smaller than this many voxels
            (set to background). 0 disables. Default: 0.
        **kwargs: Additional keyword arguments (silently ignored for YAML
            config compatibility).

    Returns:
        np.ndarray: Instance segmentation of shape :math:`(Z, Y, X)`.

    Raises:
        ImportError: If ``affogato`` is not installed.
        ValueError: If predictions have wrong shape or channel/offset mismatch.

    Examples:
        >>> # 6-channel banis affinities (3 attractive + 3 long-range)
        >>> seg = decode_mutex_watershed(affs)

        >>> # Anisotropic subsampling of repulsive edges for speed
        >>> seg = decode_mutex_watershed(affs, strides=[1, 4, 4])
    """
    if not AFFOGATO_AVAILABLE:
        raise ImportError(
            "affogato is not installed (needed for Mutex Watershed). It is a "
            "conda-forge package (the PyPI 'affogato' is an unrelated project):\n"
            "  conda install -c conda-forge affogato"
        )

    predictions = np.asarray(predictions)
    if predictions.ndim != 4:
        raise ValueError(
            f"Expected affinity predictions with shape (C, Z, Y, X), "
            f"got {predictions.ndim}D array with shape {predictions.shape}."
        )

    num_channels = predictions.shape[0]
    if offsets is None:
        if num_channels != 6:
            raise ValueError(
                f"offsets must be provided for {num_channels}-channel predictions; "
                "the default 6-channel banis layout only applies when C == 6."
            )
        offsets = _DEFAULT_OFFSETS_6
    parsed_offsets: List[Tuple[int, int, int]] = parse_affinity_offsets(offsets)

    if len(parsed_offsets) != num_channels:
        raise ValueError(
            f"Number of offsets ({len(parsed_offsets)}) must match affinity "
            f"channels ({num_channels})."
        )
    if not 1 <= num_attractive < num_channels:
        raise ValueError(
            f"num_attractive must be in [1, {num_channels - 1}], got {num_attractive}."
        )

    # affogato expects float32 affinities where, after inversion, HIGH = merge
    # for attractive edges and HIGH = split for repulsive edges.
    affs = predictions.astype(np.float32, copy=True)
    if invert_attractive:
        affs[:num_attractive] = 1.0 - affs[:num_attractive]
    if invert_repulsive:
        affs[num_attractive:] = 1.0 - affs[num_attractive:]

    mws_kwargs: dict[str, Any] = {}
    if strides is not None:
        mws_kwargs["strides"] = list(int(s) for s in strides)
        mws_kwargs["randomize_strides"] = bool(randomize_strides)
    if mask is not None:
        mws_kwargs["mask"] = np.asarray(mask, dtype=bool)

    logger.info(
        "Running mutex watershed: %d offsets (%d attractive), strides=%s",
        len(parsed_offsets),
        num_attractive,
        mws_kwargs.get("strides"),
    )

    seg = compute_mws_segmentation(
        affs,
        parsed_offsets,
        num_attractive,
        **mws_kwargs,
    )
    seg = np.asarray(seg)

    if min_instance_size > 0:
        from ..utils import remove_small_instances

        seg = remove_small_instances(seg, thres_small=min_instance_size, mode="background")

    return cast2dtype(seg)
