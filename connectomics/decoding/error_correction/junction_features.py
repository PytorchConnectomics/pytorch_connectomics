#!/usr/bin/env python3
"""GT-free junction geometry measured on predicted-segment skeletons.

The decoder uses three principles:

1. THE JUNCTION IS THE CLOSEST APPROACH, NOT A CHOSEN ENDPOINT PAIR.
   ``best_endpoint_profile_pair`` ranks endpoint pairs by a profile score and never by
   proximity to the contact, so on the anchor pair it selected endpoints 10.4 um and
   14.4 um away and published ``gap_nm = 5316`` for two skeletons that pass within
   **125 nm**.  Here the junction is ``SkeletonPaths.closest_approach`` and nothing else.

2. THE TURN ANGLE IS A SKELETON TANGENT, NOT A VOXEL-FACE NORMAL.
   ``normal_alignment`` dotted the summed voxel-face normals of the agglomerated contact
   patch with an endpoint chord.  That surface is drawn by watershed at supervoxel scale;
   L143 measured its AUC at 0.219/0.386, i.e. anti-correlated.  Here both directions are
   quadratic fits in arc length on the RESAMPLED skeleton (kimimaro spacing is not
   uniform), taken outward from the junction at five scales, and the convention is
   **180 deg = a perfectly collinear continuation**, 90 deg = perpendicular.

3. THE SPINE TEST TESTS WHAT A SPINE IS.
   ``spine_veto = roles[left] | roles[right]`` read a per-ENDPOINT role flag at those
   far-away endpoints.  A spine is a SHORT, THIN protrusion meeting a backbone at close
   to a RIGHT ANGLE, so all three of those are measured at the junction and all three
   must hold for a side to be called a spine. The frozen physical cut points are recorded
   below and do not depend on evaluation data.

Usage
-----
  python -m connectomics.decoding.error_correction.junction_features decoder-scope ...
  python -m connectomics.decoding.error_correction.junction_features compute ...
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .artifacts import reject_evaluation_path
from .skeleton_paths import SkeletonPaths, resample

DEV = Path(__file__).resolve().parent
CANDIDATES = DEV / "matchguard_error_correction" / "decoder_gtfree" / "contact_merge_candidates.npz"
OUT_DIR = DEV / "reports" / "junction"
SCOPE_PATH = OUT_DIR / "junction_scope.npz"
DECODER_DIR = DEV / "matchguard_error_correction" / "decoder_gtfree_junction_v1"
DECODER_SCOPE_PATH = DECODER_DIR / "junction_scope_aff05.npz"

#: arc lengths (nm) at which each side's tangent is fitted
SCALES = (500.0, 1000.0, 2000.0, 3000.0, 5000.0)
WALK_MAX_NM = float(max(SCALES))
#: a branch shorter than 60% of L is a MISSING measurement at L, not a short one
MIN_FIT_FRACTION = 0.6
#: skel_paths.walk semantics; a shaft junction is near-perpendicular so a strict dot<=0
#: rule empties 11% of real junctions (measured, see skel_paths.walk docstring)
AWAY_COS_MAX = 0.5
MAX_BRANCHES = 12
MAX_VISITED = 20_000
#: local radius window (arc length along the skeleton, both sides of the junction)
LOCAL_MEDIAN_NM = 2000.0
#: geodesic search radius for "where does this branch actually end"
LEAF_CAP_NM = 20_000.0
#: a degree-2 chain longer than this is a backbone by any definition; stop walking it
CHAIN_CAP_NM = 50_000.0
#: affinity gate defining the primary scope (a superset of every operator built so far)
AFFINITY_GATE = 0.05

#: spine cut points, DERIVED FROM THE BACKGROUND SAMPLE by ``finalize`` and recorded with
#: their full derivation in reports/junction/spine_cuts.json.  No GT set was used.
#:   angle  97.4 deg = the lower half-maximum edge of the continuation mode in the
#:          background turn-angle histogram (below it the histogram is a low flat
#:          isotropic shelf, above it the mode rises steeply to its peak at 131 deg)
#:   2512 nm / 0.45 = the largest stub length and radius ratio up to which perpendicular
#:          sides stay >=1.25x enriched over continuation sides in that same sample
SPINE_ANGLE_DEG = 97.4
SPINE_STUB_NM = 2512.0
SPINE_RADIUS_REL = 0.45

NAN = float("nan")


def log(*values: object) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *values, flush=True)


# --------------------------------------------------------------------- geometry


def truncate(points: np.ndarray, length: float, radii: np.ndarray | None = None):
    """First ``length`` nm of a polyline, with the final point interpolated exactly.

    Returns ``None`` when the polyline is shorter than ``MIN_FIT_FRACTION * length``:
    silently using a truncated branch would make every scale agree by construction.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < 2:
        return None
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if arc[-1] < MIN_FIT_FRACTION * length:
        return None
    if arc[-1] <= length:
        return p if radii is None else (p, np.asarray(radii, dtype=np.float64))
    j = int(np.searchsorted(arc, length))
    fraction = (length - arc[j - 1]) / max(step[j - 1], 1e-9)
    tip = p[j - 1] + fraction * (p[j] - p[j - 1])
    out = np.concatenate([p[:j], tip[None, :]], axis=0)
    if radii is None:
        return out
    r = np.asarray(radii, dtype=np.float64)
    tip_r = r[j - 1] + fraction * (r[j] - r[j - 1])
    return out, np.concatenate([r[:j], [tip_r]])


def tangent(points: np.ndarray, length: float, mode: str = "q10"):
    """Unit OUTWARD direction at the junction, fitted over arc length ``length``.

    ``q10``: arc-length-uniform resample at L/10, quadratic least squares per coordinate,
    derivative at s=0.  ``chord``: the straight chord, kept as an independent estimator so
    that estimator sensitivity is checkable downstream.
    """
    p = truncate(points, length)
    if p is None:
        return None
    if mode == "chord":
        v = p[-1] - p[0]
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else None
    q, _ = resample(p, length / 10.0)
    q = np.asarray(q, dtype=np.float64)
    if len(q) < 4:
        return None
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))])
    basis = np.stack([np.ones_like(s), s, s * s], axis=1)
    coefficients, *_ = np.linalg.lstsq(basis, q, rcond=None)
    v = coefficients[1]
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        v = q[-1] - q[0]
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
    return v / n


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """180 deg = perfectly collinear continuation, 90 deg = perpendicular."""
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u, v)), -1.0, 1.0))))


# --------------------------------------------------------------------- per-label


class LabelLocal:
    """Adjacency-derived quantities for one segment's repaired skeleton."""

    __slots__ = ("adj", "degree", "graph", "points", "radii")

    def __init__(self, skeleton):
        from scipy.sparse import csr_matrix

        n = len(skeleton.vertices_nm)
        self.points = skeleton.vertices_nm.astype(np.float64)
        self.radii = skeleton.radii_nm.astype(np.float64)
        edges = skeleton.edges
        adj: list[list[int]] = [[] for _ in range(n)]
        for left, right in edges.tolist():
            adj[left].append(right)
            adj[right].append(left)
        self.adj = adj
        self.degree = np.asarray([len(a) for a in adj], dtype=np.int32)
        if len(edges):
            weight = np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1)
            weight = np.maximum(weight, 1e-6)
            rows = np.concatenate([edges[:, 0], edges[:, 1]])
            cols = np.concatenate([edges[:, 1], edges[:, 0]])
            data = np.concatenate([weight, weight])
            self.graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        else:
            self.graph = csr_matrix((n, n))

    def leaf_distance_nm(self, vertex: int) -> float:
        """Geodesic to the nearest degree-1 vertex, capped at ``LEAF_CAP_NM``.

        Distinguishes 'the branch ENDS here' (a cut the merge would repair) from 'the
        skeleton passes by' (a side contact).  ``inf`` means no leaf within the cap.
        """
        from scipy.sparse.csgraph import dijkstra

        if int(self.degree[vertex]) <= 1:
            return 0.0
        distance = dijkstra(self.graph, directed=False, indices=int(vertex), limit=LEAF_CAP_NM)
        leaves = np.flatnonzero(self.degree == 1)
        if not len(leaves):
            return float("inf")
        best = float(np.min(distance[leaves]))
        return best if np.isfinite(best) else float("inf")

    def _chain(self, start: int, first: int) -> tuple[float, bool]:
        """Walk a degree-2 chain from ``start`` through ``first``.

        Returns (arc length nm, ended at a leaf).  Stops at the first vertex whose degree
        is not 2, or at ``CHAIN_CAP_NM``.
        """
        previous, current = start, first
        total = float(np.linalg.norm(self.points[first] - self.points[start]))
        while total < CHAIN_CAP_NM and int(self.degree[current]) == 2:
            step = [w for w in self.adj[current] if w != previous]
            if not step:
                break
            previous, current = current, step[0]
            total += float(np.linalg.norm(self.points[current] - self.points[previous]))
        return total, bool(int(self.degree[current]) == 1)

    def stub(self, vertex: int) -> tuple[float, int]:
        """(length of the branch this vertex sits on, number of leaf ends).

        degree 2 -> the maximal degree-2 chain through it, both directions.
        degree 1 -> the chain from the leaf to the first branch point.
        degree>=3 -> the vertex IS a branch point; the shortest incident chain is taken,
                     because that is the one a spine would be.
        """
        degree = int(self.degree[vertex])
        if degree == 0:
            return 0.0, 0
        if degree >= 3:
            options = [self._chain(vertex, w) for w in self.adj[vertex]]
            length, leaf = min(options, key=lambda item: item[0])
            return float(length), int(leaf)
        if degree == 1:
            length, leaf = self._chain(vertex, self.adj[vertex][0])
            return float(length), 1 + int(leaf)
        left = self._chain(vertex, self.adj[vertex][0])
        right = self._chain(vertex, self.adj[vertex][1])
        return float(left[0] + right[0]), int(left[1]) + int(right[1])


# --------------------------------------------------------------------- engine


class JunctionEngine:
    def __init__(self, paths: SkeletonPaths | None = None, cache: int = 256):
        if paths is None:
            paths = SkeletonPaths(max_cached_skeletons=cache)
        else:
            # a forked worker must not share the parent's LRU dicts by identity
            paths._skeletons, paths._trees, paths._walk_adjacency = {}, {}, {}
            paths._max_cached = cache
        self.sp = paths
        self._local: dict[int, LabelLocal] = {}
        self._cache = cache

    def local(self, label: int) -> LabelLocal:
        key = int(label)
        hit = self._local.get(key)
        if hit is None:
            hit = LabelLocal(self.sp.skeleton(key))
            if len(self._local) >= self._cache:
                self._local.pop(next(iter(self._local)))
            self._local[key] = hit
        return hit

    # ------------------------------------------------------------------ walks

    def _branches(self, label: int, vertex: int, away_from):
        try:
            return self.sp.walk(
                label,
                int(vertex),
                WALK_MAX_NM,
                away_from=away_from,
                away_cos_max=AWAY_COS_MAX,
                max_branches=MAX_BRANCHES,
                max_visited=MAX_VISITED,
            )
        except IndexError:
            return []

    @staticmethod
    def _tangents(branches, length: float, mode: str = "q10"):
        """Distinct outward directions at scale ``length``.

        Two DFS branches that only diverge beyond ``length`` share their first ``length``
        nm exactly, so they are one direction, not two: they are deduped on the truncated
        vertex-index prefix.  Without this the branch count is a DFS artefact.
        """
        seen: set[tuple] = set()
        out = []
        for branch in branches:
            points = np.asarray(branch.points_nm, dtype=np.float64)
            arc = (
                np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
                if len(points) > 1
                else np.zeros(1)
            )
            if arc[-1] < MIN_FIT_FRACTION * length:
                continue
            keep = int(np.searchsorted(arc, length, side="right"))
            key = tuple(int(v) for v in np.asarray(branch.vertex_index)[:keep])
            if key in seen:
                continue
            seen.add(key)
            direction = tangent(points, length, mode=mode)
            if direction is None:
                continue
            out.append((direction, float(arc[-1])))
        return out

    # --------------------------------------------------------------- features

    def features(self, label_a: int, label_b: int) -> dict | None:
        approach = self.sp.closest_approach(int(label_a), int(label_b))
        if approach["vA"] < 0 or approach["vB"] < 0 or not np.isfinite(approach["distance_nm"]):
            return None
        pa = np.asarray(approach["pA"], dtype=np.float64)
        pb = np.asarray(approach["pB"], dtype=np.float64)
        va, vb = int(approach["vA"]), int(approach["vB"])
        local_a, local_b = self.local(label_a), self.local(label_b)

        out: dict[str, float | int] = {
            "gap_junction_nm": float(approach["distance_nm"]),
            "a_vertex": va,
            "b_vertex": vb,
            "a_global": int(approach["gA"]),
            "b_global": int(approach["gB"]),
            "a_n_vertices": len(local_a.points),
            "b_n_vertices": len(local_b.points),
            "a_degree": int(local_a.degree[va]),
            "b_degree": int(local_b.degree[vb]),
            "a_radius_nm": float(local_a.radii[va]),
            "b_radius_nm": float(local_b.radii[vb]),
        }
        out["junction_zyx_nm"] = 0.5 * (pa + pb)
        out["a_leaf_dist_nm"] = local_a.leaf_distance_nm(va)
        out["b_leaf_dist_nm"] = local_b.leaf_distance_nm(vb)
        stub_a = local_a.stub(va)
        stub_b = local_b.stub(vb)
        out["a_stub_len_nm"], out["a_stub_leaf_ends"] = stub_a[0], stub_a[1]
        out["b_stub_len_nm"], out["b_stub_leaf_ends"] = stub_b[0], stub_b[1]

        branches_a = self._branches(label_a, va, pb)
        branches_b = self._branches(label_b, vb, pa)
        out["a_walk_branches"] = len(branches_a)
        out["b_walk_branches"] = len(branches_b)

        # local radius: median over every branch point within LOCAL_MEDIAN_NM of arc,
        # both sides of the junction, plus the junction vertex itself.
        for tag, branches, local, vertex in (
            ("a", branches_a, local_a, va),
            ("b", branches_b, local_b, vb),
        ):
            pool = [float(local.radii[vertex])]
            for branch in branches:
                cut = truncate(branch.points_nm, LOCAL_MEDIAN_NM, np.asarray(branch.radii_nm))
                if cut is None:
                    pool.extend(np.asarray(branch.radii_nm, dtype=np.float64).tolist())
                else:
                    pool.extend(cut[1].tolist())
            out[f"{tag}_radius_med2um_nm"] = float(np.median(pool)) if pool else NAN
            out[f"{tag}_radius_n"] = len(pool)

        # ------------------------------------------------- multi-scale turn angle
        best_by_scale: list[float] = []
        for length in SCALES:
            ta = self._tangents(branches_a, length)
            tb = self._tangents(branches_b, length)
            tag = int(length)
            out[f"n_branch_a_{tag}"] = len(ta)
            out[f"n_branch_b_{tag}"] = len(tb)
            if not ta or not tb:
                out[f"turn_best_{tag}"] = NAN
                out[f"turn_second_{tag}"] = NAN
                out[f"turn_margin_{tag}"] = NAN
                out[f"turn_primary_{tag}"] = NAN
                best_by_scale.append(NAN)
                continue
            matrix = np.asarray([[angle_deg(u[0], v[0]) for v in tb] for u in ta], dtype=np.float64)
            flat = np.sort(matrix.ravel())[::-1]
            best = float(flat[0])
            second = float(flat[1]) if len(flat) > 1 else NAN
            out[f"turn_best_{tag}"] = best
            out[f"turn_second_{tag}"] = second
            out[f"turn_margin_{tag}"] = best - second if len(flat) > 1 else NAN
            # the primary arm = the longest branch on each side, i.e. no max-picking
            ia = int(np.argmax([item[1] for item in ta]))
            ib = int(np.argmax([item[1] for item in tb]))
            out[f"turn_primary_{tag}"] = float(matrix[ia, ib])
            best_by_scale.append(best)

        values = np.asarray(best_by_scale, dtype=np.float64)
        finite = np.isfinite(values)
        out["turn_n_scales"] = int(finite.sum())
        out["turn_best_mean"] = float(values[finite].mean()) if finite.any() else NAN
        out["turn_best_min"] = float(values[finite].min()) if finite.any() else NAN
        out["turn_best_spread"] = (
            float(values[finite].max() - values[finite].min()) if finite.sum() >= 2 else NAN
        )
        out["turn_best_std"] = float(values[finite].std()) if finite.sum() >= 2 else NAN

        # independent estimator, one scale, as an estimator-sensitivity check
        ca = self._tangents(branches_a, 2000.0, mode="chord")
        cb = self._tangents(branches_b, 2000.0, mode="chord")
        out["turn_chord_2000"] = (
            float(max(angle_deg(u[0], v[0]) for u in ca for v in cb)) if ca and cb else NAN
        )

        # ------------------------------------------------------- spine quantities
        ra = out["a_radius_med2um_nm"]
        rb = out["b_radius_med2um_nm"]
        out["radius_ratio"] = (
            float(min(ra, rb) / max(ra, rb))
            if np.isfinite(ra) and np.isfinite(rb) and max(ra, rb) > 0
            else NAN
        )
        out["a_radius_rel"] = (
            float(ra / rb) if np.isfinite(ra) and np.isfinite(rb) and rb > 0 else NAN
        )
        out["b_radius_rel"] = (
            float(rb / ra) if np.isfinite(ra) and np.isfinite(rb) and ra > 0 else NAN
        )
        out["a_radius_rel_own"] = (
            float(local_a.radii[va] / ra) if np.isfinite(ra) and ra > 0 else NAN
        )
        out["b_radius_rel_own"] = (
            float(local_b.radii[vb] / rb) if np.isfinite(rb) and rb > 0 else NAN
        )
        turn_short = out.get("turn_best_1000", NAN)
        if not np.isfinite(turn_short):
            turn_short = out.get("turn_best_2000", NAN)
        out["turn_short_deg"] = float(turn_short)
        out["perp_dev_deg"] = float(abs(turn_short - 90.0)) if np.isfinite(turn_short) else NAN

        # a spine is SHORT and THIN and meets the backbone at close to a RIGHT ANGLE;
        # all three, or it is not a spine (the shipped column tested none of them).
        right_angle = bool(np.isfinite(turn_short) and turn_short <= SPINE_ANGLE_DEG)
        for tag in ("a", "b"):
            out[f"{tag}_is_spine"] = float(
                right_angle
                and out[f"{tag}_stub_len_nm"] <= SPINE_STUB_NM
                and np.isfinite(out[f"{tag}_radius_rel"])
                and out[f"{tag}_radius_rel"] <= SPINE_RADIUS_REL
            )
        out["spine_veto"] = float(bool(out["a_is_spine"]) or bool(out["b_is_spine"]))
        return out


# ----------------------------------------------------------------------- scope


def build_decoder_scope(
    candidates_path: Path = CANDIDATES,
    output_path: Path = DECODER_SCOPE_PATH,
    min_affinity: float = AFFINITY_GATE,
) -> dict:
    """Build the exhaustive affinity-selected scope used by the autonomous decoder.

    The scope contains no random or evaluation-selected rows. It is exactly reproducible
    from the GT-free candidate table.
    """
    reject_evaluation_path(candidates_path)
    reject_evaluation_path(output_path)
    with np.load(candidates_path) as data:
        if "gt_free" not in data or not bool(data["gt_free"].item()):
            raise ValueError(f"{candidates_path} is not explicitly marked GT-free")
        affinity = np.asarray(data["affinity_ge08_fraction"])
        left = np.asarray(data["left"])
        right = np.asarray(data["right"])
    rows = np.flatnonzero(affinity >= min_affinity)
    payload = {
        "row": rows.astype(np.int64),
        "source": np.ones(len(rows), dtype=np.uint8),
        "left": left[rows],
        "right": right[rows],
        "affinity_ge08_fraction": affinity[rows],
        "min_affinity_ge08_fraction": np.asarray(min_affinity),
        "selection": np.asarray("exhaustive GT-free affinity threshold"),
        "gt_free": np.asarray(True),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **payload)
    log(f"decoder scope -> {output_path} rows={len(rows):,} affinity>={min_affinity}")
    return payload


# --------------------------------------------------------------------- compute

_ENGINE: JunctionEngine | None = None
#: set in the parent BEFORE forking so every worker inherits the 360 MB vertex array and
#: the CSR adjacency copy-on-write instead of re-reading them from /projects eight times
_SHARED: SkeletonPaths | None = None
_ROWS = _LEFT = _RIGHT = None

VECTOR_KEYS = ("junction_zyx_nm",)


def _init_worker(rows, left, right):
    global _ENGINE, _ROWS, _LEFT, _RIGHT
    _ENGINE = JunctionEngine(paths=_SHARED)
    _ROWS, _LEFT, _RIGHT = rows, left, right


def _run_chunk(bounds):
    start, stop = bounds
    assert _ENGINE is not None
    records = []
    for position in range(start, stop):
        try:
            feature = _ENGINE.features(int(_LEFT[position]), int(_RIGHT[position]))
        except Exception as error:  # a single bad segment must not kill the shard
            feature = None
            if position % 1000 == 0:
                log(f"row {int(_ROWS[position])} failed: {type(error).__name__}: {error}")
        records.append((int(_ROWS[position]), feature))
    return records


def compute(
    workers: int,
    limit: int | None,
    out_path: Path,
    scope_path: Path = SCOPE_PATH,
    *,
    graph_path: Path | None = None,
    stitch_path: Path | None = None,
    cache_dir: Path | None = None,
) -> None:
    for path in (out_path, scope_path, graph_path, stitch_path, cache_dir):
        if path is not None:
            reject_evaluation_path(path)
    with np.load(scope_path) as data:
        if "gt_free" not in data or not bool(data["gt_free"].item()):
            raise ValueError(f"{scope_path} is not explicitly marked GT-free")
        rows = np.asarray(data["row"])
        left = np.asarray(data["left"])
        right = np.asarray(data["right"])
        source = np.asarray(data["source"])
    if limit:
        rows, left, right, source = rows[:limit], left[:limit], right[:limit], source[:limit]
    # order by the smaller label so consecutive rows reuse a cached skeleton
    order = np.argsort(np.minimum(left, right), kind="stable")
    rows, left, right, source = rows[order], left[order], right[order], source[order]
    log(f"compute rows={len(rows):,} workers={workers}")

    paths_kwargs = {}
    if graph_path is not None:
        paths_kwargs["graph_path"] = graph_path
    if stitch_path is not None:
        paths_kwargs["stitch_path"] = stitch_path
    if cache_dir is not None:
        paths_kwargs["cache_dir"] = cache_dir
    paths = SkeletonPaths(**paths_kwargs)
    engine_probe = JunctionEngine(paths=paths)
    template = None
    for position in range(len(rows)):
        template = engine_probe.features(int(left[position]), int(right[position]))
        if template is not None:
            break
    if template is None:
        raise RuntimeError("no row produced features")
    scalar_keys = [key for key in template if key not in VECTOR_KEYS]
    del engine_probe

    # touch the shared arrays before forking so workers inherit them copy-on-write
    global _SHARED
    _SHARED = SkeletonPaths(max_cached_skeletons=8, **paths_kwargs)
    _ = _SHARED.vertices_nm, _SHARED.radii_nm, _SHARED.index, _SHARED.adjacency

    chunk = 64
    bounds = [(i, min(i + chunk, len(rows))) for i in range(0, len(rows), chunk)]
    store = {key: np.full(len(rows), NAN, dtype=np.float64) for key in scalar_keys}
    junction = np.full((len(rows), 3), NAN, dtype=np.float64)
    ok = np.zeros(len(rows), dtype=bool)
    index_of_row = {int(value): position for position, value in enumerate(rows)}

    start_time = time.time()
    done = 0

    def absorb(records):
        nonlocal done
        for row_id, feature in records:
            done += 1
            if feature is None:
                continue
            position = index_of_row[row_id]
            ok[position] = True
            junction[position] = feature["junction_zyx_nm"]
            for key in scalar_keys:
                store[key][position] = feature[key]

    if workers <= 1:
        _init_worker(rows, left, right)
        for pair in bounds:
            absorb(_run_chunk(pair))
            if done % 2048 < chunk:
                rate = done / max(time.time() - start_time, 1e-9)
                log(f"{done:,}/{len(rows):,} {rate:.1f} rows/s")
    else:
        import multiprocessing as mp

        context = mp.get_context("fork")
        with context.Pool(workers, initializer=_init_worker, initargs=(rows, left, right)) as pool:
            for records in pool.imap_unordered(_run_chunk, bounds, chunksize=1):
                absorb(records)
                if done % 2048 < chunk:
                    rate = done / max(time.time() - start_time, 1e-9)
                    remaining = (len(rows) - done) / max(rate, 1e-9)
                    log(f"{done:,}/{len(rows):,} {rate:.1f} rows/s eta {remaining/60:.1f} min")

    payload = {key: value for key, value in store.items()}
    payload["row"] = rows.astype(np.int64)
    payload["left"] = left
    payload["right"] = right
    payload["source"] = source
    payload["ok"] = ok
    payload["junction_zyx_nm"] = junction
    payload["wall_clock_s"] = np.asarray(time.time() - start_time)
    payload["gt_free"] = np.asarray(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    log(
        f"features -> {out_path} rows={len(rows):,} ok={int(ok.sum()):,} "
        f"{time.time()-start_time:.0f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("decoder-scope", "compute"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "junction_features_raw.npz")
    parser.add_argument("--scope-path", type=Path, default=SCOPE_PATH)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--stitch", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--min-affinity", type=float, default=AFFINITY_GATE)
    arguments = parser.parse_args()
    if arguments.command == "decoder-scope":
        build_decoder_scope(arguments.candidates, arguments.out, arguments.min_affinity)
    else:
        compute(
            arguments.workers,
            arguments.limit or None,
            arguments.out,
            arguments.scope_path,
            graph_path=arguments.graph,
            stitch_path=arguments.stitch,
            cache_dir=arguments.cache_dir,
        )


if __name__ == "__main__":
    main()
