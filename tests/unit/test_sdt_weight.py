import numpy as np
import pytest
import torch

from connectomics.data.processing import distance as distance_mod
from connectomics.data.processing.build import count_stacked_label_transform_channels
from connectomics.data.processing.transforms import MultiTaskLabelTransformd

THICK_CENTER = (4, 4, 3)
THIN_NECK = (4, 4, 9)


def _make_dumbbell_label() -> np.ndarray:
    label = np.zeros((9, 9, 19), dtype=np.int32)
    label[2:7, 2:7, 1:6] = 1
    label[2:7, 2:7, 13:18] = 1
    label[4, 4, 6:13] = 1
    return label


def _make_skeleton_vertices() -> np.ndarray:
    return np.asarray([[4, 4, x] for x in range(1, 18)], dtype=np.int64)


def _make_skeleton_volume(label: np.ndarray) -> np.ndarray:
    skel = np.zeros_like(label)
    verts = _make_skeleton_vertices()
    skel[verts[:, 0], verts[:, 1], verts[:, 2]] = 1
    return skel


def _patch_skeletonize(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_batch_skeletonize(label, resolution, max_parallel=1):
        return {1: _make_skeleton_vertices()}

    monkeypatch.setattr(distance_mod, "_batch_skeletonize", _fake_batch_skeletonize)


def _assert_weight_bounds(
    label: np.ndarray, weight: np.ndarray, w_base: float, boost: float
) -> None:
    np.testing.assert_allclose(weight[label == 0], w_base)
    assert float(weight.max()) <= w_base + boost + 1e-5
    assert np.all(weight[label > 0] >= w_base)


def test_skeleton_aware_distance_transform_weight_map_thin_neck(monkeypatch):
    _patch_skeletonize(monkeypatch)
    label = _make_dumbbell_label()
    w_base = 1.0
    weight_param = 5.0

    unweighted = distance_mod.skeleton_aware_distance_transform(
        label,
        resolution=(1.0, 1.0, 1.0),
        weight_param=0.0,
        max_parallel=0,
    )
    weighted = distance_mod.skeleton_aware_distance_transform(
        label,
        resolution=(1.0, 1.0, 1.0),
        weight_param=weight_param,
        w_base=w_base,
        max_parallel=0,
    )

    assert weighted.shape == (2, *label.shape)
    np.testing.assert_array_equal(weighted[0], unweighted)
    _assert_weight_bounds(label, weighted[1], w_base, weight_param)
    assert weighted[1][THIN_NECK] > weighted[1][THICK_CENTER]


def test_skeleton_aware_edt_from_skeleton_vol_weight_map_thin_neck():
    label = _make_dumbbell_label()
    skeleton_vol = _make_skeleton_volume(label)
    w_base = 1.0
    weight_param = 5.0

    unweighted = distance_mod.skeleton_aware_edt_from_skeleton_vol(
        label,
        skeleton_vol,
        resolution=(1.0, 1.0, 1.0),
        weight_param=0.0,
    )
    weighted = distance_mod.skeleton_aware_edt_from_skeleton_vol(
        label,
        skeleton_vol,
        resolution=(1.0, 1.0, 1.0),
        weight_param=weight_param,
        w_base=w_base,
    )

    assert weighted.shape == (2, *label.shape)
    np.testing.assert_array_equal(weighted[0], unweighted)
    _assert_weight_bounds(label, weighted[1], w_base, weight_param)
    assert weighted[1][THIN_NECK] > weighted[1][THICK_CENTER]


def test_weight_param_zero_keeps_single_channel_output(monkeypatch):
    _patch_skeletonize(monkeypatch)
    label = _make_dumbbell_label()
    skeleton_vol = _make_skeleton_volume(label)

    direct_default = distance_mod.skeleton_aware_distance_transform(
        label,
        resolution=(1.0, 1.0, 1.0),
        max_parallel=0,
    )
    direct_zero = distance_mod.skeleton_aware_distance_transform(
        label,
        resolution=(1.0, 1.0, 1.0),
        weight_param=0.0,
        w_base=2.0,
        max_parallel=0,
    )
    precomputed_default = distance_mod.skeleton_aware_edt_from_skeleton_vol(
        label,
        skeleton_vol,
        resolution=(1.0, 1.0, 1.0),
    )
    precomputed_zero = distance_mod.skeleton_aware_edt_from_skeleton_vol(
        label,
        skeleton_vol,
        resolution=(1.0, 1.0, 1.0),
        weight_param=0.0,
        w_base=2.0,
    )

    assert direct_default.shape == label.shape
    assert precomputed_default.shape == label.shape
    np.testing.assert_array_equal(direct_default, direct_zero)
    np.testing.assert_array_equal(precomputed_default, precomputed_zero)


def test_multitask_transform_weighted_label_aux_skeleton_path_emits_matching_mask():
    label = _make_dumbbell_label()
    skeleton_vol = _make_skeleton_volume(label)
    transform = MultiTaskLabelTransformd(
        keys=["label"],
        tasks=[
            {
                "name": "affinity",
                "kwargs": {
                    "offsets": [
                        "1-0-0",
                        "0-1-0",
                        "0-0-1",
                        "2-0-0",
                        "0-2-0",
                        "0-0-2",
                    ],
                    "affinity_mode": "banis",
                },
            },
            {
                "name": "skeleton_aware_edt",
                "kwargs": {
                    "resolution": (1.0, 1.0, 1.0),
                    "weight_param": 5.0,
                    "w_base": 1.0,
                },
            },
        ],
    )

    out = transform({"label": label, "label_aux": skeleton_vol})

    assert isinstance(out["label"], torch.Tensor)
    assert tuple(out["label"].shape) == (8, *label.shape)
    assert tuple(out["label_mask"].shape) == tuple(out["label"].shape)
    assert out["label_mask"].dtype == torch.bool
    assert torch.all(out["label_mask"][6:8])
    assert out["label"][7][THIN_NECK] > out["label"][7][THICK_CENTER]


def test_weighted_skeleton_aware_edt_counts_extra_target_channel():
    channels = count_stacked_label_transform_channels(
        {
            "targets": [
                {
                    "name": "affinity",
                    "kwargs": {
                        "offsets": [
                            "1-0-0",
                            "0-1-0",
                            "0-0-1",
                            "2-0-0",
                            "0-2-0",
                            "0-0-2",
                        ],
                        "affinity_mode": "banis",
                    },
                },
                {
                    "name": "skeleton_aware_edt",
                    "kwargs": {"weight_param": 5.0},
                },
            ]
        }
    )

    assert channels == 8


def test_weighted_full_sdt_label_aux_raises_clear_error():
    label = _make_dumbbell_label()
    aux = np.full(label.shape, -1.0, dtype=np.float32)
    aux[label > 0] = 0.5
    transform = MultiTaskLabelTransformd(
        keys=["label"],
        tasks=[
            {
                "name": "skeleton_aware_edt",
                "kwargs": {"weight_param": 1.0},
            }
        ],
    )

    with pytest.raises(ValueError, match="full-SDT label_aux cache"):
        transform({"label": label, "label_aux": aux})
