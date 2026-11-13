"""Greedy mean-affinity agglomeration of an existing segmentation.

Repeatedly merges the two touching segments whose shared boundary has the highest mean
affinity, recomputing the mean of every boundary the merged segment inherits, until no
boundary reaches ``threshold``. Unlike single linkage (ABISS ``max`` merge with a large size
threshold), one strong edge on a small contact cannot join two segments.

Affinities are in the ABISS layout: channels x, y, z of a (C, Z, Y, X) array, each value
stored at the second voxel of its edge. Label 0 is background and is never merged.

    python -m connectomics.decoding.decoders.affinity_agglomeration \
        --seg seg.h5 --aff aff.h5 --output merged.h5 --threshold 0.4
"""

from __future__ import annotations

import argparse
import heapq

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

__all__ = ["agglomerate_mean_affinity"]


def agglomerate_mean_affinity(seg: np.ndarray, aff: np.ndarray, threshold: float) -> np.ndarray:
    """Merge touching segments of ``seg`` (Z, Y, X) down to a mean boundary affinity.

    Args:
        seg: integer labels, 0 = background.
        aff: (3, Z, Y, X) affinities, channels x, y, z, stored at the second voxel of each edge.
        threshold: stop when the highest mean boundary affinity is below this.

    Returns:
        Labels with the dtype of ``seg``; a merged segment keeps one of its members' ids.
    """
    if aff.ndim != 4 or aff.shape[0] != 3 or aff.shape[1:] != seg.shape:
        raise ValueError(f"expected aff (3, *{seg.shape}), got {aff.shape}")
    ids, lab = np.unique(seg, return_inverse=True)
    lab = lab.reshape(seg.shape)
    n = len(ids)

    # Boundary graph: sum and count of the affinities between each pair of touching segments.
    us, vs, ws = [], [], []
    for channel, axis in ((0, 2), (1, 1), (2, 0)):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        u, v = lab[tuple(lo)], lab[tuple(hi)]
        d = (u != v) & (ids[u] != 0) & (ids[v] != 0)
        us.append(u[d])
        vs.append(v[d])
        ws.append(aff[channel][tuple(hi)][d])
    u, v = np.concatenate(us), np.concatenate(vs)
    w = np.concatenate(ws).astype(np.float64)
    key, inv = np.unique(np.minimum(u, v) * n + np.maximum(u, v), return_inverse=True)
    e_sum, e_cnt = np.bincount(inv, weights=w), np.bincount(inv)
    del u, v, w, us, vs, ws, inv

    nb: list[dict[int, list[float]]] = [{} for _ in range(n)]
    heap = []
    for a, b, s, c in zip((key // n).tolist(), (key % n).tolist(), e_sum.tolist(), e_cnt.tolist()):
        nb[a][b] = nb[b][a] = [s, c]
        if s / c >= threshold:
            heap.append((-s / c, a, b))
    heapq.heapify(heap)
    kept, removed = [], []
    while heap:
        neg_mean, a, b = heapq.heappop(heap)
        e = nb[a].get(b)
        if e is None or -neg_mean != e[0] / e[1]:
            continue  # stale entry: the boundary changed after it was pushed
        if len(nb[a]) < len(nb[b]):
            a, b = b, a
        kept.append(a)
        removed.append(b)
        del nb[a][b]
        for k, ek in nb[b].items():
            if k == a:
                continue
            del nb[k][b]
            cur = nb[a].get(k)
            if cur is None:
                cur = nb[a][k] = nb[k][a] = [ek[0], ek[1]]
            else:
                cur[0] += ek[0]
                cur[1] += ek[1]
            if cur[0] / cur[1] >= threshold:
                heapq.heappush(heap, (-cur[0] / cur[1], a, k))
        nb[b] = {}

    graph = sparse.coo_matrix((np.ones(len(kept)), (kept, removed)), shape=(n, n))
    n_comp, comp = connected_components(graph, directed=False)
    rep = np.zeros(n_comp, dtype=ids.dtype)
    rep[comp] = ids
    return rep[comp][lab]


def main() -> None:
    import h5py

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--seg", required=True, help="segmentation h5 (Z, Y, X), first dataset")
    ap.add_argument("--aff", required=True, help="affinity h5 (C>=3, Z, Y, X), first dataset")
    ap.add_argument("--output", required=True, help="output h5, dataset 'main'")
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()
    with h5py.File(args.seg, "r") as f:
        seg = f[list(f)[0]][...]
    with h5py.File(args.aff, "r") as f:
        aff = f[list(f)[0]][:3].astype(np.float32)
    out = agglomerate_mean_affinity(seg, aff, args.threshold)
    with h5py.File(args.output, "w") as f:
        f.create_dataset("main", data=out, compression="gzip")
    print(f"{len(np.unique(seg))} -> {len(np.unique(out))} labels (threshold {args.threshold})")


if __name__ == "__main__":
    main()
