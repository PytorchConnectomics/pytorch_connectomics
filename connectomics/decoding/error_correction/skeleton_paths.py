#!/usr/bin/env python3
"""Skeleton path API over the GT-free whole-volume predicted-segment skeletons.

Everything the L143 directive asks for ("do skeleton interpolation smoothness") needs one
primitive the existing feature tables never had: for a PREDICTED segment id, the actual
assembled skeleton graph, and the ability to walk it outward from a junction.

Frame (verified empirically, see ``skel_validate_paths.py``):
  ``vertices_nm`` columns are **(z, y, x) nanometres**, native voxel size (20, 9, 9) nm zyx.
  Chunk ``z{k}_y{j}_x{i}`` owns the nm box ``[k,j,i] * [256,512,512] * [20,9,9]`` (zyx),
  and every vertex written for a chunk lies strictly inside its own core box.

Assembly: ``edges`` in the HDF5 are already GLOBAL vertex indices (the aggregate stage added
``vertex_start`` per chunk), and ``stitch_edges.npz`` holds cross-chunk bridges, also global
and always between two vertices of the SAME segment id. So the whole-volume graph is
``edges ∪ stitch_edges`` and it decomposes cleanly by ``vertex_label``.

Caches (built once, ~1 min, then mmap-loaded in ~1 s):
  ``reports/skel_interp/label_index.npz``  sorted label -> vertex slice + vertex order
  ``reports/skel_interp/adjacency_csr.npz`` global CSR adjacency over the assembled graph

Nothing here reads GT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

DEV = Path(__file__).resolve().parent
ROOT = DEV / "matchguard_error_correction" / "decoder_gtfree"
GRAPH_PATH = ROOT / "segment_skeleton_graph.h5"
STITCH_PATH = ROOT / "stitch_edges.npz"
CACHE_DIR = DEV / "reports" / "skel_interp"
LABEL_INDEX_PATH = CACHE_DIR / "label_index.npz"
ADJACENCY_PATH = CACHE_DIR / "adjacency_csr.npz"

#: native voxel size in nm, in the (z, y, x) order of ``vertices_nm``
VOXEL_NM_ZYX = np.asarray([20.0, 9.0, 9.0], dtype=np.float64)
#: ABISS atomic chunk core, voxels, (z, y, x)
CHUNK_VOX_ZYX = np.asarray([256, 512, 512], dtype=np.int64)
#: target sampling spacing of the skeletonizer (arm096_skeletonize_chunks.py --skeleton-spacing-nm)
SKELETON_SPACING_NM = 250.0
#: face-gap repair budget, derived from the sampling spacing, NOT fitted on any score
REPAIR_MAX_GAP_NM = 2.0 * SKELETON_SPACING_NM
REPAIR_FACE_TOLERANCE_NM = SKELETON_SPACING_NM


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


@dataclass
class Skeleton:
    """One PREDICTED segment's assembled skeleton."""

    label: int
    vertices_nm: np.ndarray  # (n, 3) float32, (z, y, x) nm
    radii_nm: np.ndarray  # (n,) float32
    edges: np.ndarray  # (m, 2) int64, LOCAL indices into vertices_nm
    global_index: np.ndarray  # (n,) int64, index into the whole-volume vertex arrays
    n_stitch_edges: int  # how many of ``edges`` came from the port bridges
    n_repair_edges: int = 0  # how many were added by the chunk-face gap repair

    def __len__(self) -> int:
        return len(self.vertices_nm)


@dataclass
class WalkBranch:
    """One continuation of a walk outward from a junction vertex."""

    points_nm: np.ndarray  # (k, 3) float32, first point is the start vertex
    radii_nm: np.ndarray  # (k,)
    vertex_index: np.ndarray  # (k,) local indices; -1 for the interpolated final point
    arc_length_nm: float
    stop: str  # 'length' | 'leaf' | 'cap'


class SkeletonPaths:
    """Lazy, cached accessor over the whole-volume skeleton graph."""

    def __init__(
        self,
        graph_path: Path = GRAPH_PATH,
        stitch_path: Path = STITCH_PATH,
        cache_dir: Path = CACHE_DIR,
        max_cached_skeletons: int = 64,
        repair_gaps: bool = True,
        repair_max_gap_nm: float = REPAIR_MAX_GAP_NM,
        repair_face_tolerance_nm: float = REPAIR_FACE_TOLERANCE_NM,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.stitch_path = Path(stitch_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._handle: h5py.File | None = None
        self._vertices: np.ndarray | None = None
        self._radii: np.ndarray | None = None
        self._index: dict[str, np.ndarray] | None = None
        self._adjacency: dict[str, np.ndarray] | None = None
        self._skeletons: dict[int, Skeleton] = {}
        self._trees: dict[int, object] = {}
        self._walk_adjacency: dict[int, list[list[int]]] = {}
        self._max_cached = max_cached_skeletons
        self.repair_gaps = repair_gaps
        self.repair_max_gap_nm = float(repair_max_gap_nm)
        self.repair_face_tolerance_nm = float(repair_face_tolerance_nm)

    # ------------------------------------------------------------------ storage

    @property
    def handle(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.graph_path, "r")
        return self._handle

    @property
    def vertices_nm(self) -> np.ndarray:
        """All 30M vertices, (z, y, x) nm. Read once (360 MB)."""
        if self._vertices is None:
            self._vertices = np.asarray(self.handle["vertices_nm"])
        return self._vertices

    @property
    def radii_nm(self) -> np.ndarray:
        if self._radii is None:
            self._radii = np.asarray(self.handle["radii_nm"])
        return self._radii

    # ------------------------------------------------------------ label index

    def build_label_index(self, force: bool = False) -> dict[str, np.ndarray]:
        """sorted unique label -> contiguous run in ``order`` of global vertex indices."""
        path = self.cache_dir / LABEL_INDEX_PATH.name
        if not force and path.exists():
            with np.load(path) as data:
                return {key: data[key] for key in data.files}
        log("building label index over vertex_label")
        labels = np.asarray(self.handle["vertex_label"])
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        unique, starts, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
        payload = {
            "label": unique.astype(np.uint64),
            "start": starts.astype(np.int64),
            "count": counts.astype(np.int64),
            "order": order.astype(np.int32),
            "n_vertices": np.asarray(len(labels), dtype=np.int64),
        }
        np.savez(path, **payload)
        log(f"label index -> {path} labels={len(unique):,}")
        return payload

    @property
    def index(self) -> dict[str, np.ndarray]:
        if self._index is None:
            self._index = self.build_label_index()
        return self._index

    def labels(self) -> np.ndarray:
        return self.index["label"]

    def vertex_indices(self, label: int) -> np.ndarray:
        """Global vertex indices owned by ``label`` (sorted ascending)."""
        table = self.index
        position = int(np.searchsorted(table["label"], np.uint64(label)))
        if position >= len(table["label"]) or int(table["label"][position]) != int(label):
            return np.zeros(0, dtype=np.int64)
        start = int(table["start"][position])
        stop = start + int(table["count"][position])
        picked = table["order"][start:stop].astype(np.int64)
        picked.sort()
        return picked

    # -------------------------------------------------------------- adjacency

    def build_adjacency(self, force: bool = False) -> dict[str, np.ndarray]:
        """CSR adjacency over ``edges ∪ stitch_edges`` (global vertex ids)."""
        path = self.cache_dir / ADJACENCY_PATH.name
        if not force and path.exists():
            with np.load(path) as data:
                return {key: data[key] for key in data.files}
        log("building global CSR adjacency")
        edges = np.asarray(self.handle["edges"], dtype=np.int64)
        with np.load(self.stitch_path) as data:
            stitch = np.asarray(data["edges"], dtype=np.int64).reshape(-1, 2)
        n_intra = len(edges)
        both = np.concatenate([edges, stitch], axis=0)
        is_stitch = np.zeros(len(both), dtype=bool)
        is_stitch[n_intra:] = True
        n_vertices = int(self.handle["vertices_nm"].shape[0])
        src = np.concatenate([both[:, 0], both[:, 1]])
        dst = np.concatenate([both[:, 1], both[:, 0]])
        flag = np.concatenate([is_stitch, is_stitch])
        order = np.argsort(src, kind="stable")
        src, dst, flag = src[order], dst[order], flag[order]
        indptr = np.zeros(n_vertices + 1, dtype=np.int64)
        np.add.at(indptr, src + 1, 1)
        np.cumsum(indptr, out=indptr)
        payload = {
            "indptr": indptr,
            "indices": dst.astype(np.int32),
            "is_stitch": flag,
            "n_intra_edges": np.asarray(n_intra, dtype=np.int64),
            "n_stitch_edges": np.asarray(len(stitch), dtype=np.int64),
        }
        np.savez(path, **payload)
        log(f"adjacency -> {path} edges={len(both):,}")
        return payload

    @property
    def adjacency(self) -> dict[str, np.ndarray]:
        if self._adjacency is None:
            self._adjacency = self.build_adjacency()
        return self._adjacency

    def neighbors(self, global_vertex: int) -> np.ndarray:
        adj = self.adjacency
        lo = int(adj["indptr"][global_vertex])
        hi = int(adj["indptr"][global_vertex + 1])
        return adj["indices"][lo:hi].astype(np.int64)

    # -------------------------------------------------------------- skeletons

    def skeleton(self, label: int, use_stitch: bool = True, repair: bool | None = None) -> Skeleton:
        """Assemble one segment's skeleton across chunk boundaries.

        ``repair`` (default: the instance's ``repair_gaps``) additionally closes the
        chunk-face gaps the port stitch missed — 85% of the sub-300 nm inter-component
        gaps sit within 300 nm of a chunk face plane, so leaving them open turns one
        neurite into two and fakes a kink at every chunk boundary. Vertices are never
        changed, so ``closest_approach`` is identical either way; only ``walk`` and the
        component structure differ.
        """
        key = int(label)
        repair = self.repair_gaps if repair is None else bool(repair)
        cached = self._skeletons.get(key)
        if cached is not None and use_stitch and repair == self.repair_gaps:
            return cached
        globals_ = self.vertex_indices(key)
        vertices = self.vertices_nm[globals_]
        radii = self.radii_nm[globals_]
        adj = self.adjacency
        indptr, indices, is_stitch = adj["indptr"], adj["indices"], adj["is_stitch"]
        # gather every incident edge of the owned vertices, keep each undirected edge once
        lo = indptr[globals_]
        hi = indptr[globals_ + 1]
        counts = (hi - lo).astype(np.int64)
        total = int(counts.sum())
        if total:
            ends = np.cumsum(counts)
            take = (
                np.arange(total, dtype=np.int64)
                - np.repeat(ends - counts, counts)
                + np.repeat(lo.astype(np.int64), counts)
            )
            left = np.repeat(np.arange(len(globals_), dtype=np.int64), counts)
            right_global = indices[take].astype(np.int64)
            stitch_flag = is_stitch[take]
            if not use_stitch:
                keep = ~stitch_flag
                left, right_global, stitch_flag = left[keep], right_global[keep], stitch_flag[keep]
            right = np.searchsorted(globals_, right_global)
            valid = (right < len(globals_)) & (
                globals_[np.minimum(right, len(globals_) - 1)] == right_global
            )
            left, right, stitch_flag = left[valid], right[valid], stitch_flag[valid]
            keep = left < right
            edges = np.stack([left[keep], right[keep]], axis=1)
            n_stitch = int(np.count_nonzero(stitch_flag[keep]))
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
            n_stitch = 0
        skeleton = Skeleton(key, vertices, radii, edges, globals_, n_stitch, 0)
        if repair and use_stitch:
            skeleton = self._repair_face_gaps(skeleton)
        if use_stitch and repair == self.repair_gaps:
            if len(self._skeletons) >= self._max_cached:
                self._skeletons.pop(next(iter(self._skeletons)))
            self._skeletons[key] = skeleton
        return skeleton

    def _repair_face_gaps(self, skeleton: Skeleton) -> Skeleton:
        """Kruskal over cross-component pairs that straddle a chunk face plane."""
        from scipy.spatial import cKDTree

        points = skeleton.vertices_nm.astype(np.float64)
        n = len(points)
        member, count = _components(n, skeleton.edges)
        if count < 2 or n < 2:
            return skeleton
        tree = cKDTree(points)
        k = int(min(16, n))
        distance, neighbor = tree.query(points, k=k, workers=-1)
        core = CHUNK_VOX_ZYX * VOXEL_NM_ZYX
        left_list, right_list, cost_list = [], [], []
        for row in range(n):
            for column in range(1, k):
                other = int(neighbor[row][column])
                if member[other] == member[row]:
                    continue
                gap = float(distance[row][column])
                if gap > self.repair_max_gap_nm:
                    break
                midpoint = 0.5 * (points[row] + points[other])
                modulo = np.mod(midpoint, core)
                if float(np.min(np.minimum(modulo, core - modulo))) > self.repair_face_tolerance_nm:
                    continue
                left_list.append(row)
                right_list.append(other)
                cost_list.append(gap)
        if not left_list:
            return skeleton
        order = np.argsort(np.asarray(cost_list))
        parent = np.arange(count, dtype=np.int64)

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return node

        added = []
        for position in order:
            a, b = find(int(member[left_list[position]])), find(int(member[right_list[position]]))
            if a == b:
                continue
            parent[a] = b
            added.append((left_list[position], right_list[position]))
        if not added:
            return skeleton
        extra = np.asarray(added, dtype=np.int64).reshape(-1, 2)
        return Skeleton(
            skeleton.label,
            skeleton.vertices_nm,
            skeleton.radii_nm,
            np.concatenate([skeleton.edges, extra], axis=0),
            skeleton.global_index,
            skeleton.n_stitch_edges,
            len(extra),
        )

    def n_components(self, label: int, use_stitch: bool = True, repair: bool | None = None) -> int:
        skeleton = self.skeleton(label, use_stitch=use_stitch, repair=repair)
        return int(_components(len(skeleton.vertices_nm), skeleton.edges)[1])

    def components(
        self, label: int, use_stitch: bool = True, repair: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """(component id per vertex, component sizes sorted descending)."""
        skeleton = self.skeleton(label, use_stitch=use_stitch, repair=repair)
        member, count = _components(len(skeleton.vertices_nm), skeleton.edges)
        sizes = np.bincount(member, minlength=count)
        return member, np.sort(sizes)[::-1]

    # ------------------------------------------------------- closest approach

    def _tree(self, label: int):
        from scipy.spatial import cKDTree

        key = int(label)
        tree = self._trees.get(key)
        if tree is None:
            tree = cKDTree(self.skeleton(key).vertices_nm.astype(np.float64))
            if len(self._trees) >= self._max_cached:
                self._trees.pop(next(iter(self._trees)))
            self._trees[key] = tree
        return tree

    def closest_approach(self, label_a: int, label_b: int) -> dict:
        """The two skeleton points that come nearest. This defines the junction."""
        skeleton_a = self.skeleton(label_a)
        skeleton_b = self.skeleton(label_b)
        if not len(skeleton_a) or not len(skeleton_b):
            return {
                "distance_nm": float("inf"),
                "pA": None,
                "pB": None,
                "vA": -1,
                "vB": -1,
                "gA": -1,
                "gB": -1,
            }
        # query the larger set against the smaller tree
        if len(skeleton_a) <= len(skeleton_b):
            tree, query, swap = self._tree(label_a), skeleton_b.vertices_nm, True
        else:
            tree, query, swap = self._tree(label_b), skeleton_a.vertices_nm, False
        distance, nearest = tree.query(query.astype(np.float64), k=1, workers=-1)
        best = int(np.argmin(distance))
        if swap:
            index_a, index_b = int(nearest[best]), best
        else:
            index_a, index_b = best, int(nearest[best])
        return {
            "distance_nm": float(distance[best]),
            "pA": skeleton_a.vertices_nm[index_a].astype(np.float64),
            "pB": skeleton_b.vertices_nm[index_b].astype(np.float64),
            "vA": index_a,
            "vB": index_b,
            "gA": int(skeleton_a.global_index[index_a]),
            "gB": int(skeleton_b.global_index[index_b]),
        }

    # -------------------------------------------------------------------- walk

    def walk(
        self,
        label: int,
        from_vertex: int,
        length_nm: float,
        away_from: Sequence[float] | int | None = None,
        away_cos_max: float = 0.5,
        max_branches: int = 64,
        max_visited: int = 200_000,
    ) -> list[WalkBranch]:
        """All continuations of arc length ``length_nm`` outward from ``from_vertex``.

        ``from_vertex`` is a LOCAL index into ``skeleton(label)``. ``away_from`` is a point
        (or a local vertex index) that the first step must move away from — the partner
        segment's junction point. Because ``from_vertex`` is by construction the closest
        vertex to that point, the measured dot product with the partner direction is
        centred on −0.05 (p25 −0.27, p75 +0.14): a shaft junction is near-perpendicular, so
        a strict ``dot <= 0`` rule empties 11% of real junctions outright and randomly kills
        one of the two shaft arms on the rest. The rule is therefore ``dot <= away_cos_max``
        (default 0.5 = a 60 deg forbidden cone, which drops 2% of directions), and if that
        still empties the set the single least-aligned neighbour is kept.

        Every branch reaching ``length_nm`` is returned, never one arbitrary continuation;
        the last point is interpolated to land exactly at ``length_nm``.
        """
        skeleton = self.skeleton(label)
        vertices = skeleton.vertices_nm.astype(np.float64)
        n = len(vertices)
        if not (0 <= from_vertex < n):
            raise IndexError(f"vertex {from_vertex} out of range for label {label} (n={n})")
        neighbor_list = self._walk_adjacency.get(int(label))
        if neighbor_list is None or len(neighbor_list) != n:
            neighbor_list = _adjacency_lists(n, skeleton.edges)
            if len(self._walk_adjacency) >= self._max_cached:
                self._walk_adjacency.pop(next(iter(self._walk_adjacency)))
            self._walk_adjacency[int(label)] = neighbor_list
        start_point = vertices[from_vertex]
        first = list(neighbor_list[from_vertex])
        if away_from is not None:
            anchor = (
                vertices[int(away_from)]
                if np.isscalar(away_from)
                else np.asarray(away_from, dtype=np.float64)
            )
            direction = anchor - start_point
            norm = float(np.linalg.norm(direction))
            if norm > 0:
                direction = direction / norm
                aligned = [
                    float(np.dot(_unit(vertices[node] - start_point), direction)) for node in first
                ]
                kept = [node for node, value in zip(first, aligned) if value <= away_cos_max]
                if not kept and first:
                    kept = [first[int(np.argmin(aligned))]]
                first = kept
        branches: list[WalkBranch] = []
        visited_total = 0
        stack: list[tuple[list[int], float, set[int]]] = []
        for node in first:
            stack.append(
                ([from_vertex, node], _norm(vertices[node] - start_point), {from_vertex, node})
            )
        if not first:
            return branches
        while stack:
            path, arc, seen = stack.pop()
            visited_total += 1
            if visited_total > max_visited or len(branches) >= max_branches:
                # the last entry carries stop='cap' so a caller can see the walk was truncated
                branches.append(_finish(skeleton, path, arc, length_nm, "cap"))
                return branches
            if arc >= length_nm:
                branches.append(_finish(skeleton, path, arc, length_nm, "length"))
                continue
            current = path[-1]
            forward = [node for node in neighbor_list[current] if node not in seen]
            if not forward:
                branches.append(_finish(skeleton, path, arc, length_nm, "leaf"))
                continue
            for node in forward:
                step = _norm(vertices[node] - vertices[current])
                stack.append((path + [node], arc + step, seen | {node}))
        return branches

    def branch_count(self, label: int, from_vertex: int, length_nm: float, **kwargs) -> int:
        return len(self.walk(label, from_vertex, length_nm, **kwargs))


# ----------------------------------------------------------------- free helpers


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _norm(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector))


def _adjacency_lists(n: int, edges: np.ndarray) -> list[list[int]]:
    lists: list[list[int]] = [[] for _ in range(n)]
    for left, right in edges.tolist():
        lists[left].append(right)
        lists[right].append(left)
    return lists


def _finish(
    skeleton: Skeleton, path: Sequence[int], arc: float, length_nm: float, stop: str
) -> WalkBranch:
    vertices = skeleton.vertices_nm.astype(np.float64)
    points = vertices[list(path)]
    radii = skeleton.radii_nm[list(path)].astype(np.float64)
    index = np.asarray(path, dtype=np.int64)
    if stop == "length" and len(points) >= 2:
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        if cumulative[-1] > length_nm:
            last = int(np.searchsorted(cumulative, length_nm))
            excess = length_nm - cumulative[last - 1]
            fraction = excess / max(steps[last - 1], 1e-9)
            point = points[last - 1] + fraction * (points[last] - points[last - 1])
            radius = radii[last - 1] + fraction * (radii[last] - radii[last - 1])
            points = np.concatenate([points[:last], point[None, :]], axis=0)
            radii = np.concatenate([radii[:last], [radius]])
            index = np.concatenate([index[:last], [-1]])
            arc = float(length_nm)
    return WalkBranch(points.astype(np.float32), radii.astype(np.float32), index, float(arc), stop)


def resample(
    polyline: np.ndarray, step_nm: float, radii: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Arc-length-uniform resample. kimimaro spacing is NOT uniform; any curvature
    estimate taken on the raw spacing is garbage, so every geometric feature must
    start here."""
    points = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        return points.astype(np.float32), (None if radii is None else np.asarray(radii, np.float32))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], steps > 0])
    points = points[keep]
    if radii is not None:
        radii = np.asarray(radii, dtype=np.float64)[keep]
    if len(points) < 2:
        return points.astype(np.float32), (None if radii is None else radii.astype(np.float32))
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    n_steps = max(int(np.floor(total / step_nm)), 1)
    targets = np.arange(n_steps + 1, dtype=np.float64) * step_nm
    if total - targets[-1] > 0.05 * step_nm:
        targets = np.concatenate([targets, [total]])
    out = np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in range(3)], axis=1)
    out_radii = None
    if radii is not None:
        out_radii = np.interp(targets, cumulative, radii).astype(np.float32)
    return out.astype(np.float32), out_radii


def _components(n: int, edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected components over a single segment's local graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    if not len(edges):
        return np.arange(n, dtype=np.int64), n
    data = np.ones(len(edges), dtype=np.uint8)
    graph = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    count, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64), int(count)


def voxel_zyx_to_nm(voxel_zyx: Iterable[float]) -> np.ndarray:
    return np.asarray(list(voxel_zyx), dtype=np.float64) * VOXEL_NM_ZYX


def nm_to_voxel_zyx(point_nm: Iterable[float]) -> np.ndarray:
    return np.asarray(list(point_nm), dtype=np.float64) / VOXEL_NM_ZYX
