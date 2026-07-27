"""Build an ERL skeleton graph from a dense label volume, so scoring needs no side-car.

kimimaro on a downsampled copy of the labels (default ZYX stride (2, 4, 4), which
keeps physical vertex coordinates identical to full resolution because the
anisotropy handed to kimimaro is scaled by the same factor). One skeleton per
label, ``skeleton_id`` = the dense label as a string.

This is a pure function of the label volume, so a missing ``.erlgraph.npz`` is
regenerated deterministically instead of being carried around as an input.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["build_erl_graph", "ensure_erl_graph"]

DOWNSAMPLE = (2, 4, 4)


class _Skel:
    """Minimal duck-typed skeleton accepted by ``em_erl.erl.skel_to_erlgraph``."""

    vertices: np.ndarray
    edges: np.ndarray


def _erl_api():
    from ..metrics.nerl import import_em_erl

    import_em_erl()  # ensures lib/em_erl is importable, raises a clear error if not
    from em_erl.erl import ERLGraph, skel_to_erlgraph

    return ERLGraph, skel_to_erlgraph


def build_erl_graph(
    labels: np.ndarray,
    resolution: Sequence[float],
    *,
    downsample: Sequence[int] = DOWNSAMPLE,
    num_workers: int = 8,
) -> Any:
    """Skeletonize each label and return an ``ERLGraph``.

    Args:
        labels: dense ZYX instance volume.
        resolution: voxel size in nm, ZYX (array-axis order, not xyz).
        downsample: ZYX stride applied before skeletonizing. The anisotropy given
            to kimimaro is scaled by the same factor, so vertex coordinates stay
            in the full-resolution physical frame.
    """
    import kimimaro

    from ..data.processing.distance import kimimaro_config

    _, skel_to_erlgraph = _erl_api()

    res = np.asarray(resolution, dtype=float)
    down = tuple(int(d) for d in downsample)
    aniso = tuple(float(res[i] * down[i]) for i in range(3))
    small = np.ascontiguousarray(labels[:: down[0], :: down[1], :: down[2]])
    logger.info(
        "skeletonizing %s -> %s (stride %s, anisotropy %s nm)",
        labels.shape,
        small.shape,
        down,
        aniso,
    )
    skels = kimimaro.skeletonize(
        small,
        **kimimaro_config(small, aniso),
        parallel=int(num_workers),
        progress=False,
    )

    skeletons: dict[str, _Skel] = {}
    for label, sk in skels.items():
        edges = np.asarray(sk.edges, np.int64)
        if not len(edges):
            continue
        s = _Skel()
        s.vertices = np.round(np.asarray(sk.vertices, float) / res).astype(np.int64)
        s.edges = edges
        skeletons[str(int(label))] = s
    logger.info("built %d skeletons", len(skeletons))
    return skel_to_erlgraph(
        skeletons,
        skeleton_resolution=list(np.asarray(resolution).tolist()),
        length_threshold=0,
    )


def ensure_erl_graph(
    npz_path: str | Path,
    label_path: str | Path,
    resolution: Sequence[float],
    *,
    downsample: Sequence[int] = DOWNSAMPLE,
    num_workers: int = 8,
) -> Path:
    """Return ``npz_path``, building it from ``label_path`` if it does not exist.

    Writing is atomic, so an interrupted or concurrent build cannot leave a
    half-written graph behind for the next run to load.
    """
    npz_path = Path(npz_path)
    if npz_path.exists():
        return npz_path

    from ..data.io import read_volume

    logger.info("skeleton graph %s not found -- building it from %s", npz_path.name, label_path)
    labels = np.asarray(read_volume(str(label_path), dataset="main"))
    graph = build_erl_graph(labels, resolution, downsample=downsample, num_workers=num_workers)
    del labels

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.stem + ".TMP.npz")
    graph.save_npz(str(tmp))
    os.replace(tmp, npz_path)
    logger.info("wrote %s (%d skeletons)", npz_path, graph.num_skeletons)
    return npz_path
