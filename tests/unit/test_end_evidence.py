"""Contracts for segmentation-based evidence about skeleton free ends."""

import numpy as np
import pytest

from connectomics.metrics.unsupervised.end_evidence import (
    ball_max_radius,
    face_planes,
    face_reach_um,
    terminal_branch,
)

SPACING = np.array([1.0, 1.0, 1.0])


def test_terminal_branch_length_to_first_junction():
    # 0-1-2-3 chain, with 2 also joined to 4: tip 0 reaches the junction at 2.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]], float)
    edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]])
    radii = np.array([0.1, 0.1, 0.5, 0.1, 0.1])
    length, junction_radius, at_junction = terminal_branch(vertices, edges, radii, 0)
    assert (length, junction_radius, at_junction) == (2.0, 0.5, True)


def test_terminal_branch_of_a_bare_chain_has_no_junction():
    vertices = np.array([[0, 0, 0], [1, 0, 0]], float)
    assert terminal_branch(vertices, np.array([[0, 1]]), np.array([0.1, 0.1]), 0)[2] is False


def test_face_reach_uses_the_labels_own_face_voxels():
    seg = np.zeros((4, 6, 6), np.uint32)
    seg[:, 1, 1] = 7  # label 7 runs to both z faces at (y, x) = (1, 1)
    seg[1:3, 4, 4] = 9  # label 9 stays inside
    tip = np.array([0.5, 1.5, 3.5])  # 2 um in-plane from label 7's z0 voxel, 0.5 um above the face
    assert face_reach_um(face_planes(seg), seg.shape, SPACING, 7, tip) == pytest.approx(np.hypot(0.5, 2.0))
    assert face_reach_um(face_planes(seg), seg.shape, SPACING, 9, np.array([1.5, 4.5, 4.5])) == np.inf


def test_ball_max_radius_is_masked_to_the_label():
    seg = np.zeros((9, 9, 9), np.uint32)
    seg[2:7, 2:7, 2:7] = 3
    dist = np.zeros(seg.shape, np.float32)
    dist[4, 4, 4] = 2.5  # centre of label 3
    dist[0, 0, 0] = 9.0  # outside the label: must be ignored
    assert ball_max_radius(seg, dist, SPACING, 3, np.array([4.5, 4.5, 4.5]), 1.0) == 2.5
    assert ball_max_radius(seg, dist, SPACING, 3, np.array([0.5, 0.5, 0.5]), 1.0) == 0.0
