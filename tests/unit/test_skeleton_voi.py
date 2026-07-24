"""Skeleton-based VOI must match funlib.evaluate.rand_voi exactly.

``connectomics.metrics.nerl.skeleton_voi`` is a numpy port of the vendored
``lib/funlib.evaluate`` (``impl/rand_voi.hpp``) — the implementation BANIS uses.
These vectors are copied from funlib's own ``test_rand_voi.py`` so the port stays
bit-faithful. funlib's ``rand_voi(labels_a=gt, labels_b=pred)`` maps to
``skeleton_voi(node_pred_ids=pred, node_gt_ids=gt)``.
"""
from __future__ import annotations

import numpy as np
import pytest

from connectomics.metrics.nerl import skeleton_voi


@pytest.mark.parametrize(
    "gt, pred, exp_split, exp_merge",
    [
        # perfect 1:1 mapping -> no split, no merge
        ([1, 2, 3], [4, 5, 6], 0.0, 0.0),
        # two GT skeletons collapsed into one prediction -> pure merge
        ([1, 1, 2, 2], [2, 2, 2, 2], 0.0, 1.0),
        # one prediction split relative to GT -> pure split (args swapped)
        ([2, 2, 2, 2], [1, 1, 2, 2], 1.0, 0.0),
        # funlib's non-trivial reference case
        (
            [1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
            [3, 3, 3, 3, 4, 5, 6, 6, 5, 5, 5],
            0.6140806820148608,
            0.7272727272727271,
        ),
    ],
)
def test_skeleton_voi_matches_funlib_rand_voi(gt, pred, exp_split, exp_merge):
    split, merge, total = skeleton_voi(pred, gt)
    assert split == pytest.approx(exp_split)
    assert merge == pytest.approx(exp_merge)
    assert total == pytest.approx(exp_split + exp_merge)


def test_skeleton_voi_ignores_gt_zero_like_funlib():
    # funlib counts only nodes with GT != 0; a predicted background id (0) is kept.
    with_bg = skeleton_voi([9, 4, 5, 6], [0, 1, 2, 3])  # first node has gt==0
    without = skeleton_voi([4, 5, 6], [1, 2, 3])
    assert with_bg == pytest.approx(without)


def test_skeleton_voi_empty_and_shape_guard():
    assert skeleton_voi([], []) == (0.0, 0.0, 0.0)
    # node_gt_ids all 0 -> funlib counts nothing (args are pred, gt)
    assert skeleton_voi([1, 2], [0, 0]) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        skeleton_voi([1, 2, 3], [1, 2])
