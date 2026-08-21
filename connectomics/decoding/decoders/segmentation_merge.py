"""GT-free agglomeration of an over-segmentation: grow every fragment, then link.

A watershed/ABISS segmentation of neural tissue is dominated by SPLITS -- one neurite arrives as a
backbone plus a cloud of small fragments. Fixing that needs no ground truth and no skeletons: a
fragment's own voxel count says whether it can stand alone, and voxel CONTACT AREA says who it
belongs to.

Two rounds, in this order:

  round 1  GROW -- every fragment below ``anchor_size`` is absorbed into whichever anchor component
           it shares the most contact area with, repeated for ``hops`` rounds so chains of fragments
           are pulled in one shell at a time. Absorption assigns a label; it never unions two
           anchors, so it CANNOT create a false merge between two objects that were separate.
           Nothing needs to be known about which fragment belongs to which object -- every fragment
           is grown.

  round 2  LINK -- a fragment that touches exactly TWO anchor components is evidence that those two
           are one object separated by a thin break. Unioning them IS a merge and is the only round
           that can go wrong, so it is gated: the fragment must be substantial (``link_min_size``),
           its two contacts must be comparable (``link_balance``, i.e. a through-piece rather than a
           side-graze), and ``max_hub_size`` refuses any union that would put two backbone-scale
           segments in one component.

Measured on the zebrafinch/j0126 ABISS segmentation against 50 held-out skeletons (NERL,
merge_threshold 50), starting from a published GT-free skeleton join at 0.4629:

    round 1 alone (anchor 40k, 8 hops)      0.5452   0 neurons regressed
    + round 2                               see tutorials/neuron_j0126/README.md

``anchor_size`` is the one parameter worth tuning and it trades recall against safety: on that
volume 40,000 vox was the largest drop-free value (80k gained +0.007 but regressed one neuron).

The contact graph is computed from the label volume alone. For volumes too large to hold in memory,
build the graph in chunks with ``scripts/build_contact_graph.py`` and pass it as ``contact_path``.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, Iterable, Tuple

import h5py
import numpy as np

from ..utils import cast2dtype

__all__ = [
    "contact_graph",
    "grow_fragments",
    "link_through_fragments",
    "segmentation_merge",
]

logger = logging.getLogger(__name__)


def contact_graph(
    seg: np.ndarray,
    *,
    min_size: int = 0,
    sizes: Dict[int, int] | None = None,
    affinity: np.ndarray | None = None,
) -> Dict[Tuple[int, int], int] | Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]:
    """Face-adjacency between labels, as ``{(lo, hi): n_faces}``.

    Only pairs where both labels are non-zero and at least ``min_size`` voxels are kept.

    With ``affinity`` (a CZYX array whose channels are in **XYZ** order, as emitted by the
    affinity models here), also returns ``{(lo, hi): mean_affinity}`` over the shared faces.
    Note the channel mapping: array axis ``ax`` is ZYX, so it reads channel ``2 - ax``. Getting
    this backwards still "works" -- it scored 84% against 86% for the correct mapping -- so it
    will not announce itself as a failure; see dev/zebrafinch/lessons.md L89.
    """
    if seg.ndim != 3:
        raise ValueError(f"contact_graph expects a 3D label volume, got shape {seg.shape}")
    if sizes is None:
        sizes = segment_sizes(seg)
    if affinity is not None and affinity.shape[-3:] != seg.shape:
        raise ValueError(f"affinity spatial shape {affinity.shape[-3:]} != seg shape {seg.shape}")
    acc: Counter = Counter()
    aff_sum: Counter = Counter()
    for axis in range(3):
        a = np.moveaxis(seg, axis, 0)
        lo, hi = a[:-1], a[1:]
        m = (lo != hi) & (lo != 0) & (hi != 0)
        if not m.any():
            continue
        x, y = lo[m], hi[m]
        pair = np.stack([np.minimum(x, y), np.maximum(x, y)], axis=1)
        if affinity is None:
            uq, cnt = np.unique(pair, axis=0, return_counts=True)
            for (p, q), n in zip(uq.tolist(), cnt.tolist()):
                if min_size and (sizes.get(p, 0) < min_size or sizes.get(q, 0) < min_size):
                    continue
                acc[(p, q)] += n
        else:
            w = np.moveaxis(affinity[2 - axis], axis, 0)[1:][m]
            uq, inv, cnt = np.unique(pair, axis=0, return_inverse=True, return_counts=True)
            tot = np.zeros(len(uq), float)
            np.add.at(tot, inv.ravel(), w.astype(float))
            for (p, q), n, t in zip(uq.tolist(), cnt.tolist(), tot.tolist()):
                if min_size and (sizes.get(p, 0) < min_size or sizes.get(q, 0) < min_size):
                    continue
                acc[(p, q)] += n
                aff_sum[(p, q)] += t
    if affinity is None:
        return dict(acc)
    return dict(acc), {k: aff_sum[k] / max(acc[k], 1) for k in acc}


def segment_sizes(seg: np.ndarray) -> Dict[int, int]:
    """Voxel count per label, excluding 0."""
    lab, cnt = np.unique(seg, return_counts=True)
    return {
        int(label): int(count) for label, count in zip(lab.tolist(), cnt.tolist()) if label != 0
    }


def _neighbours(contacts: Dict[Tuple[int, int], int]) -> Dict[int, Dict[int, int]]:
    nb: Dict[int, Dict[int, int]] = defaultdict(dict)
    for (a, b), n in contacts.items():
        nb[a][b] = n
        nb[b][a] = n
    return nb


class _UnionFind:
    """Union-find that refuses to put two backbone-scale members in one component."""

    def __init__(self, sizes: Dict[int, int], max_hub_size: float) -> None:
        self._parent: Dict[int, int] = {}
        self._hubs: Dict[int, int] = {}
        self._sizes = sizes
        self._max_hub = max_hub_size

    def add(self, x: int) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._hubs[x] = 1 if self._sizes.get(x, 0) >= self._max_hub else 0

    def find(self, x: int) -> int:
        self.add(x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int, *, guard: bool = True) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if guard and self._hubs[ra] + self._hubs[rb] >= 2:
            return False
        self._parent[ra] = rb
        self._hubs[rb] = self._hubs[ra] + self._hubs[rb]
        return True


def link_through_fragments(
    neighbours: Dict[int, Dict[int, int]],
    sizes: Dict[int, int],
    uf: _UnionFind,
    anchors: Iterable[int],
    *,
    anchor_size: int,
    link_min_size: int,
    link_balance: float,
    link_min_contact: int,
) -> int:
    """Round 2: union anchor components bridged by a single substantial fragment."""
    anchor_set = set(anchors)
    proposals = []
    for frag, nb in neighbours.items():
        size = sizes.get(frag, 0)
        if size >= anchor_size or size < link_min_size:
            continue
        touched: Counter = Counter()
        for other, area in nb.items():
            if other in anchor_set and area >= link_min_contact:
                touched[uf.find(other)] += area
        if len(touched) != 2:
            continue
        (c_a, area_a), (c_b, area_b) = touched.most_common(2)
        if min(area_a, area_b) < link_balance * max(area_a, area_b):
            continue
        proposals.append((min(area_a, area_b), c_a, c_b))
    proposals.sort(reverse=True)  # strongest evidence first
    return sum(uf.union(c_a, c_b) for _area, c_a, c_b in proposals)


def grow_fragments(
    neighbours: Dict[int, Dict[int, int]],
    sizes: Dict[int, int],
    labels: Dict[int, int],
    *,
    anchor_size: int,
    hops: int,
    min_size: int = 0,
    min_contact: int = 0,
    margin: float | None = None,
    dominance: float | None = None,
    affinity_of: Dict[Tuple[int, int], float] | None = None,
    min_affinity: float = 0.0,
) -> int:
    """Round 1: absorb each sub-anchor fragment into its best-scoring neighbouring component.

    The bare argmax is not a good enough decision rule on a whole-volume contact graph: it
    assigns every label in the volume, and a fragment whose largest contact happens to be a
    passing neurite is assigned to it. Measured on the reference volume, ungated absorption of
    20,072,711 fragments scored 0.4405 against a 0.4629 starting point with 4 neurons regressed.
    ``margin`` and ``dominance`` refuse the ambiguous cases instead, leaving the fragment alone.

    ``labels`` is updated in place; returns the number of fragments absorbed.
    """
    total_contact = {f: sum(nb.values()) for f, nb in neighbours.items()}
    absorbed = 0
    for _hop in range(hops):
        fresh: Dict[int, int] = {}
        for frag, nb in neighbours.items():
            if frag in labels or sizes.get(frag, 0) >= anchor_size:
                continue
            if sizes.get(frag, 0) < min_size:
                continue
            votes: Counter = Counter()
            weights: Counter = Counter()
            for other, area in nb.items():
                if other in labels and area >= min_contact:
                    votes[labels[other]] += area
                    if affinity_of is not None:
                        key = (min(frag, other), max(frag, other))
                        a = affinity_of.get(key, 0.0)
                        if a < min_affinity:
                            continue
                        weights[labels[other]] = max(weights[labels[other]], a)
            if not votes:
                continue
            if affinity_of is not None:
                # rank hosts by affinity, not area: area is BELOW the random-guess baseline on
                # fragments with many candidates (16.2% vs 24.2%), while affinity is ~87% and
                # flat in difficulty. Measured 0.5036 vs 0.4510 NERL. See lessons.md L89/L90.
                top = weights.most_common(2)
            else:
                top = votes.most_common(2)
            if not top:
                # every candidate fell below min_affinity -- leave the fragment alone rather than
                # falling back to contact area, which is below chance here (lessons.md L89).
                continue
            win_label, win_area = top[0]
            if margin is not None and len(top) > 1 and top[1][1] > margin * win_area:
                continue
            if dominance is not None and win_area < dominance * max(total_contact[frag], 1):
                continue
            fresh[frag] = win_label
        if not fresh:
            break
        labels.update(fresh)
        absorbed += len(fresh)
    return absorbed


def _apply(seg: np.ndarray, labels: Dict[int, int]) -> np.ndarray:
    if not labels:
        return seg.copy()
    keys = np.array(sorted(labels), dtype=seg.dtype)
    vals = np.array([labels[int(k)] for k in keys.tolist()], dtype=seg.dtype)
    idx = np.clip(np.searchsorted(keys, seg), 0, len(keys) - 1)
    hit = keys[idx] == seg
    out = seg.copy()
    out[hit] = vals[idx[hit]]
    return out


def segmentation_merge(
    seg: np.ndarray,
    affinity: np.ndarray | None = None,
    *,
    anchor_size: int = 40000,
    hops: int = 8,
    link: bool = True,
    link_min_size: int = 5000,
    link_balance: float = 0.35,
    link_min_contact: int = 20,
    max_hub_size: float = 2e7,
    min_size: int = 200,
    grow_min_size: int = 0,
    grow_min_contact: int = 0,
    margin: float | None = None,
    dominance: float | None = None,
    use_affinity: bool = True,
    min_affinity: float = 0.02,
    affinity_path: str = "",
    affinity_dataset: str = "main",
    contact_path: str = "",
) -> np.ndarray:
    """Grow every fragment into its host, then link anchors bridged by a fragment.

    Fully ground-truth-free: the only inputs are the label volume's own voxel counts and
    contact areas. See the module docstring for what each round can and cannot break.
    """
    seg = np.ascontiguousarray(seg)
    if seg.ndim == 4 and seg.shape[0] == 1:
        seg = seg[0]
    if affinity is None and use_affinity and affinity_path:
        with h5py.File(affinity_path, "r") as f:
            key = (
                affinity_dataset
                if affinity_dataset in f
                else next(k for k in f if hasattr(f[k], "shape"))
            )
            affinity = f[key][...]
    if affinity is not None and affinity.ndim == 4 and affinity.shape[0] > 3:
        affinity = np.moveaxis(affinity, -1, 0)  # tolerate ZYXC as well as CZYX
    if use_affinity and affinity is None:
        logger.warning(
            "segmentation_merge: use_affinity is set but no affinity was given; falling back to "
            "contact area, which measured 0.4510 vs 0.5036 with affinity (lessons L89/L90)."
        )
    sizes = segment_sizes(seg)
    if contact_path:
        loaded = np.load(contact_path)
        contacts = {
            (int(a), int(b)): int(n)
            for a, b, n in zip(loaded["a"].tolist(), loaded["b"].tolist(), loaded["n"].tolist())
        }
        aff_of = None
    elif affinity is not None and use_affinity:
        contacts, aff_of = contact_graph(seg, min_size=min_size, sizes=sizes, affinity=affinity)
    else:
        contacts = contact_graph(seg, min_size=min_size, sizes=sizes)
        aff_of = None
    neighbours = _neighbours(contacts)
    anchors = {s for s, v in sizes.items() if v >= anchor_size}
    logger.info(
        "segmentation_merge: %d labels, %d anchors (>=%d vox), %d contacts",
        len(sizes),
        len(anchors),
        anchor_size,
        len(contacts),
    )

    uf = _UnionFind(sizes, max_hub_size)
    for a in anchors:
        uf.add(a)
    linked = 0
    if link:
        linked = link_through_fragments(
            neighbours,
            sizes,
            uf,
            anchors,
            anchor_size=anchor_size,
            link_min_size=link_min_size,
            link_balance=link_balance,
            link_min_contact=link_min_contact,
        )

    labels = {a: uf.find(a) for a in anchors}
    absorbed = grow_fragments(
        neighbours,
        sizes,
        labels,
        anchor_size=anchor_size,
        hops=hops,
        min_size=grow_min_size,
        min_contact=grow_min_contact,
        margin=margin,
        dominance=dominance,
        affinity_of=aff_of,
        min_affinity=min_affinity,
    )
    logger.info(
        "segmentation_merge: linked %d anchor pairs, absorbed %d fragments", linked, absorbed
    )
    return cast2dtype(_apply(seg, labels))
