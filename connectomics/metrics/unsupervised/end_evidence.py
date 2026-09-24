"""Segmentation-based evidence about skeleton free ends.

A skeleton terminal is a weak end detector on its own: surface roughness leaves
hair-thin spurs, a tip can stop short of a crop face the object plainly reaches,
and a bouton's widest point lies past any short tip-side radius window. These
measurements read the label volume instead:

``terminal_branch``
    Length from a tip to its first junction and the radius there. A branch
    shorter than a few junction radii is a surface spur, not a process end.
``face_reach_um``
    Distance from a tip to the label's *own* voxels on any crop face. Unlike a
    tip-to-plane distance, it does not credit a tip that is merely near a face.
``ball_max_radius``
    Largest inscribed radius (voxel EDT) of the label within a ball around the
    tip. Compared with the same measurement at mid-shaft points, it says whether
    the tip is swollen beyond what the shaft reads -- an axon terminal.

Coordinates are ZYX micrometers with voxel centres at ``(index + 0.5) * spacing``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ball_max_radius", "face_planes", "face_reach_um", "terminal_branch"]


def terminal_branch(
    vertices: np.ndarray, edges: np.ndarray, radii: np.ndarray, tip: int
) -> tuple[float, float, bool]:
    """``(length to first junction, radius at the stop, stopped at a junction)``."""
    adjacency: list[list[int]] = [[] for _ in range(len(vertices))]
    for a, b in np.asarray(edges).tolist():
        if a != b:
            adjacency[a].append(b)
            adjacency[b].append(a)
    previous, current, length = -1, int(tip), 0.0
    while True:
        onward = [n for n in adjacency[current] if n != previous]
        if len(onward) != 1:
            return length, float(radii[current]), len(onward) > 1
        length += float(np.linalg.norm(vertices[onward[0]] - vertices[current]))
        previous, current = current, onward[0]


def face_planes(seg: np.ndarray) -> list[tuple[int, float, np.ndarray]]:
    """The six crop faces as ``(axis, index, plane)``; slice once, reuse per end."""
    return [
        (axis, index, np.ascontiguousarray(np.take(seg, index, axis=axis)))
        for axis in range(3)
        for index in (0, seg.shape[axis] - 1)
    ]


def face_reach_um(
    planes: list[tuple[int, float, np.ndarray]],
    shape: tuple[int, int, int],
    spacing: np.ndarray,
    label: int,
    tip: np.ndarray,
    window_um: float = 2.0,
) -> float:
    """Distance from ``tip`` to the nearest voxel of ``label`` on a crop face.

    ``planes`` comes from :func:`face_planes`. Only face voxels within
    ``window_um`` in-plane of the tip are searched; ``inf`` means the label
    reaches no face near the tip.
    """
    best = np.inf
    for axis, index, full in planes:
        a, b = [k for k in range(3) if k != axis]
        lo_a = max(0, int((tip[a] - window_um) / spacing[a]))
        lo_b = max(0, int((tip[b] - window_um) / spacing[b]))
        hi_a = int((tip[a] + window_um) / spacing[a]) + 1
        hi_b = int((tip[b] + window_um) / spacing[b]) + 1
        plane_um = 0.0 if index == 0 else shape[axis] * spacing[axis]
        ii, jj = np.nonzero(full[lo_a:hi_a, lo_b:hi_b] == label)
        if not len(ii):
            continue
        points = np.zeros((len(ii), 3))
        points[:, a] = (ii + lo_a + 0.5) * spacing[a]
        points[:, b] = (jj + lo_b + 0.5) * spacing[b]
        points[:, axis] = plane_um
        best = min(best, float(np.min(np.linalg.norm(points - tip, axis=1))))
    return best


def ball_max_radius(
    seg: np.ndarray,
    dist: np.ndarray,
    spacing: np.ndarray,
    label: int,
    centre: np.ndarray,
    ball_um: float,
) -> float:
    """Largest ``dist`` value of ``label`` voxels within ``ball_um`` of ``centre``."""
    reach = np.ceil(ball_um / spacing).astype(int)
    middle = np.round(centre / spacing - 0.5).astype(int)
    lo = np.maximum(middle - reach, 0)
    hi = np.minimum(middle + reach + 1, seg.shape)
    if np.any(hi <= lo):
        return 0.0
    box = tuple(slice(l, h) for l, h in zip(lo, hi))
    axes = [(np.arange(l, h) + 0.5) * s for l, h, s in zip(lo, hi, spacing)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    inside = (np.linalg.norm(grid - centre, axis=-1) <= ball_um) & (seg[box] == label)
    return float(dist[box][inside].max()) if inside.any() else 0.0
