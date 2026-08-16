import copy
from typing import Optional

import numpy as np
import pytest
import torch

import connectomics.inference.tta as tta_module
from connectomics.config import Config
from connectomics.config.schema.model import ModelHeadConfig
from connectomics.data.processing.affinity import (
    compute_affinity_valid_mask,
    resolve_affinity_channel_groups_from_cfg,
    resolve_stacked_label_channel_count,
    seg_to_affinity,
)
from connectomics.inference.tta import TTAPredictor
from connectomics.inference.tta_affinity import (
    ViewValidity,
    build_affinity_tta_plan,
    invert_view,
    transform_offset,
)
from connectomics.inference.tta_combinations import resolve_tta_augmentation_combinations
from connectomics.inference.tta_ensemble import TTAEnsembleAccumulator
from connectomics.inference.window import build_sliding_inferer


def _segmentation(shape=(13, 14, 15)) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.integers(1, 8, size=shape, dtype=np.int32)


def _affinity_cfg(
    offsets,
    mode: str,
    *,
    extra_targets=None,
    out_channels: Optional[int] = None,
) -> Config:
    cfg = Config()
    cfg.data.label_transform.targets = [
        {
            "name": "affinity",
            "kwargs": {"offsets": [list(offset) for offset in offsets], "affinity_mode": mode},
        },
        *(extra_targets or []),
    ]
    cfg.model.out_channels = len(offsets) if out_channels is None else out_channels
    cfg.inference.model.channel_activations = []
    cfg.inference.test_time_augmentation.apply_mask = False
    return cfg


def _augment_spatial(
    tensor: torch.Tensor,
    flip_axes,
    rotation_plane: Optional[tuple[int, int]],
    k: int,
) -> torch.Tensor:
    if flip_axes:
        tensor = torch.flip(tensor, dims=list(flip_axes))
    if rotation_plane is not None and k:
        tensor = torch.rot90(tensor, k=k, dims=rotation_plane)
    return tensor


def _validity_mask(validity: ViewValidity, shape, channel: int) -> torch.Tensor:
    entry = validity.channels[channel]
    if entry is None:
        return torch.ones(shape, dtype=torch.bool)
    if isinstance(entry, torch.Tensor):
        return entry.squeeze(0).to(torch.bool)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[entry] = True
    return mask


def _assert_exact_geometry(offsets, mode, transform, shape=(13, 14, 15)):
    flip_axes, rotation_plane, k = transform
    cfg = _affinity_cfg(offsets, mode)
    plan = build_affinity_tta_plan(
        cfg,
        augmentation_combinations=[transform],
        num_raw=len(offsets),
        requested_head=None,
    )
    assert plan is not None

    segmentation = _segmentation(shape)
    augmented = _augment_spatial(
        torch.from_numpy(segmentation), flip_axes, rotation_plane, k
    ).numpy()
    augmented_affinity = seg_to_affinity(
        augmented,
        offsets=offsets,
        affinity_mode=mode,
    )
    prediction = torch.from_numpy(augmented_affinity.values.astype(np.float32)).unsqueeze(0)
    corrected, validity = invert_view(
        prediction,
        flip_axes=flip_axes,
        rotation_plane_spatial=rotation_plane,
        k=k,
        view_plan=plan.views[0],
        tta_plan=plan,
    )
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode=mode)

    for channel in range(len(offsets)):
        valid = torch.from_numpy(canonical.mask[channel])
        valid &= _validity_mask(validity, shape, channel)
        assert valid.any()
        actual = corrected[0, channel]
        expected = torch.from_numpy(canonical.values[channel]).to(dtype=actual.dtype)
        assert torch.equal(actual[valid], expected[valid])


def test_transform_offset_matches_coordinate_field_for_every_enumerated_transform():
    cfg = Config()
    tta_cfg = cfg.inference.test_time_augmentation
    tta_cfg.flip_axes = "all"
    tta_cfg.rotation90_axes = "all"
    combinations = resolve_tta_augmentation_combinations(tta_cfg, spatial_dims=3)
    spatial_shape = (7, 7, 7)
    coordinates = torch.stack(
        torch.meshgrid(
            *(torch.arange(size) for size in spatial_shape),
            indexing="ij",
        )
    )
    offset = (1, 0, 0)

    for flip_axes, rotation_plane, k in combinations:
        augmented = coordinates
        if flip_axes:
            augmented = torch.flip(augmented, dims=[axis + 1 for axis in flip_axes])
        if rotation_plane is not None and k:
            augmented = torch.rot90(
                augmented,
                k=k,
                dims=(rotation_plane[0] + 1, rotation_plane[1] + 1),
            )
        origin = (3, 3, 3)
        partner = list(origin)
        partner[0] += 1
        measured = tuple(
            int(value)
            for value in (augmented[(slice(None), *partner)] - augmented[(slice(None), *origin)])
        )
        assert measured == transform_offset(
            offset,
            flip_axes=flip_axes,
            rotation_plane_spatial=rotation_plane,
            k=k,
        )


@pytest.mark.parametrize("mode", ["banis", "deepem"])
@pytest.mark.parametrize(
    "transform",
    [
        ([0], None, 0),
        ([1], None, 0),
        ([2], None, 0),
        ([0, 1], None, 0),
        ([0, 1, 2], None, 0),
        ([], (0, 1), 1),
        ([], (0, 1), 2),
        ([], (0, 1), 3),
        ([], (0, 2), 1),
        ([], (0, 2), 2),
        ([], (0, 2), 3),
        ([], (1, 2), 1),
        ([], (1, 2), 2),
        ([], (1, 2), 3),
        ([0, 2], (0, 1), 1),
    ],
)
def test_exact_r1_geometry_for_flips_rotations_and_combinations(mode, transform):
    _assert_exact_geometry([(1, 0, 0), (0, 1, 0), (0, 0, 1)], mode, transform)


@pytest.mark.parametrize("mode", ["banis", "deepem"])
def test_exact_r1_r10_geometry(mode):
    offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (10, 0, 0),
        (0, 10, 0),
        (0, 0, 10),
    ]
    _assert_exact_geometry(offsets, mode, ([0, 2], (0, 1), 3), shape=(23, 24, 25))


@pytest.mark.parametrize("mode", ["banis", "deepem"])
def test_exact_signed_multi_axis_offset_geometry(mode):
    _assert_exact_geometry([(1, 1, 0)], mode, ([0, 1], None, 0))


def test_banis_and_deepem_invalidate_opposite_r10_faces():
    offsets = [(10, 0, 0)]
    for mode, expected in [
        ("banis", slice(0, 13)),
        ("deepem", slice(10, 23)),
    ]:
        cfg = _affinity_cfg(offsets, mode)
        plan = build_affinity_tta_plan(
            cfg,
            augmentation_combinations=[([0], None, 0)],
            num_raw=1,
            requested_head=None,
        )
        assert plan is not None
        prediction = torch.ones((1, 1, 23, 12, 11))
        _corrected, validity = invert_view(
            prediction,
            flip_axes=[0],
            rotation_plane_spatial=None,
            k=0,
            view_plan=plan.views[0],
            tta_plan=plan,
        )
        assert validity.channels[0][0] == expected


def test_plan_rejects_duplicate_and_missing_transformed_offsets():
    duplicate = _affinity_cfg([(1, 0, 0), (1, 0, 0)], "banis")
    with pytest.raises(ValueError, match="duplicate offsets"):
        build_affinity_tta_plan(
            duplicate,
            augmentation_combinations=[([], None, 0)],
            num_raw=2,
            requested_head=None,
        )

    missing = _affinity_cfg([(1, 0, 0), (0, 0, 1)], "banis")
    with pytest.raises(ValueError, match="counterpart"):
        build_affinity_tta_plan(
            missing,
            augmentation_combinations=[([], (0, 1), 1)],
            num_raw=2,
            requested_head=None,
        )


def test_antipodal_offsets_prefer_exact_bijective_mapping_without_roll():
    cfg = _affinity_cfg([(1, 0, 0), (-1, 0, 0)], "banis")
    plan = build_affinity_tta_plan(
        cfg,
        augmentation_combinations=[([], None, 0), ([0], None, 0)],
        num_raw=2,
        requested_head=None,
    )
    assert plan is not None
    assert plan.partial_channels == frozenset()
    assert all(move.shift is None for view in plan.views for move in view.moves)


def test_affinity_output_rank_mismatch_names_both_ranks():
    cfg = _affinity_cfg([(1, 0, 0)], "banis")
    plan = build_affinity_tta_plan(
        cfg,
        augmentation_combinations=[([], None, 0)],
        num_raw=1,
        requested_head=None,
    )
    assert plan is not None
    with pytest.raises(ValueError, match="offset rank 3.*spatial rank 2"):
        invert_view(
            torch.ones((1, 1, 5, 5)),
            flip_axes=[],
            rotation_plane_spatial=None,
            k=0,
            view_plan=plan.views[0],
            tta_plan=plan,
        )


def test_validity_aware_accumulator_supports_per_channel_mean_min_max():
    accumulator = TTAEnsembleAccumulator(
        (1, 4, 1, 1, 4),
        dtype=torch.float32,
        device=torch.device("cpu"),
        mode_map=["mean", "min", "max", "mean"],
        partial_channels=[0, 1, 2],
        distributed_sharding=False,
        max_views=2,
    )
    first = torch.tensor([1.0, 4.0, 2.0, 10.0]).reshape(1, 1, 1, 1, 4).repeat(1, 4, 1, 1, 1)
    second = torch.tensor([3.0, 2.0, 5.0, 14.0]).reshape(1, 1, 1, 1, 4).repeat(1, 4, 1, 1, 1)
    first_validity = ViewValidity(
        (
            (slice(0, 1), slice(0, 1), slice(0, 3)),
            (slice(0, 1), slice(0, 1), slice(1, 4)),
            None,
            None,
        )
    )
    second_validity = ViewValidity(
        (
            (slice(0, 1), slice(0, 1), slice(1, 4)),
            None,
            (slice(0, 1), slice(0, 1), slice(0, 3)),
            None,
        )
    )
    accumulator.add(first, first_validity)
    accumulator.add(second, second_validity)
    result = accumulator.finalize()

    assert torch.equal(result[0, 0, 0, 0], torch.tensor([1.0, 3.0, 3.5, 14.0]))
    assert torch.equal(result[0, 1, 0, 0], torch.tensor([3.0, 2.0, 2.0, 10.0]))
    assert torch.equal(result[0, 2, 0, 0], torch.tensor([3.0, 4.0, 5.0, 10.0]))
    assert torch.equal(result[0, 3, 0, 0], torch.tensor([2.0, 3.0, 3.5, 12.0]))


def test_validity_aware_accumulator_rejects_zero_coverage():
    accumulator = TTAEnsembleAccumulator(
        (1, 1, 2, 2, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        mode_map=["mean"],
        partial_channels=[0],
        distributed_sharding=False,
        max_views=1,
    )
    accumulator.add(
        torch.ones((1, 1, 2, 2, 2)),
        ViewValidity(((slice(0, 0), slice(0, 2), slice(0, 2)),)),
    )
    with pytest.raises(RuntimeError, match="channel 0.*voxel index"):
        accumulator.finalize()


def _configured_oracle(offsets, mode, segmentation):
    def forward(x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for sample in x:
            target = seg_to_affinity(
                sample[0].round().to(torch.int32).cpu().numpy(),
                offsets=offsets,
                affinity_mode=mode,
            )
            outputs.append(torch.from_numpy(target.values.astype(np.float32)))
        return torch.stack(outputs).to(device=x.device, dtype=x.dtype)

    return forward


@pytest.mark.parametrize("mode", ["banis", "deepem"])
@pytest.mark.parametrize("selector,expected_channels", [("0:3", [0, 1, 2]), ("3:6", [3, 4, 5])])
def test_standard_predict_corrects_raw_r1_r10_before_channel_selection(
    mode, selector, expected_channels
):
    offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
    ]
    segmentation = _segmentation((13, 14, 15))
    cfg = _affinity_cfg(offsets, mode)
    cfg.inference.model.select_channel = selector
    tta_cfg = cfg.inference.test_time_augmentation
    tta_cfg.enabled = True
    tta_cfg.patch_first_local = False
    tta_cfg.flip_axes = "all"
    tta_cfg.rotation90_axes = [[1, 2]]
    tta_cfg.rotate90_k = [0, 1]
    predictor = TTAPredictor(
        cfg=cfg,
        sliding_inferer=None,
        forward_fn=_configured_oracle(offsets, mode, segmentation),
    )
    result = predictor.predict(torch.from_numpy(segmentation).float()[None, None])
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode=mode)
    expected = torch.from_numpy(canonical.values[expected_channels].astype(np.float32))[None]
    assert torch.equal(result, expected)


def test_noncontiguous_reordered_selection_keeps_values_and_validity_aligned():
    offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
    ]
    segmentation = _segmentation((13, 14, 15))
    cfg = _affinity_cfg(offsets, "banis")
    cfg.inference.model.select_channel = [5, 0, 3]
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = False
    cfg.inference.test_time_augmentation.flip_axes = "all"
    cfg.inference.test_time_augmentation.rotation90_axes = None
    predictor = TTAPredictor(
        cfg,
        sliding_inferer=None,
        forward_fn=_configured_oracle(offsets, "banis", segmentation),
    )
    result = predictor.predict(torch.from_numpy(segmentation).float()[None, None])
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode="banis")
    expected = torch.from_numpy(canonical.values[[5, 0, 3]].astype(np.float32))[None]
    assert torch.equal(result, expected)


def _coordinate_oracle(offsets, mode, segmentation):
    global_seg = torch.from_numpy(segmentation).to(torch.int64)

    def forward(coordinates: torch.Tensor) -> torch.Tensor:
        outputs = []
        for sample in coordinates:
            coord = sample.round().to(torch.int64)
            spatial_rank = coord.shape[0]
            origin = [0] * spatial_rank
            basis = []
            for axis in range(spatial_rank):
                partner = list(origin)
                partner[axis] = 1
                basis.append(coord[(slice(None), *partner)] - coord[(slice(None), *origin)])
            channels = []
            for offset in offsets:
                displacement = sum(int(offset[axis]) * basis[axis] for axis in range(spatial_rank))
                source = coord if mode == "banis" else coord - displacement[:, None, None, None]
                destination = (
                    coord + displacement[:, None, None, None] if mode == "banis" else coord
                )
                valid = torch.ones(coord.shape[1:], dtype=torch.bool, device=coord.device)
                for axis, size in enumerate(segmentation.shape):
                    valid &= (source[axis] >= 0) & (source[axis] < size)
                    valid &= (destination[axis] >= 0) & (destination[axis] < size)
                values = torch.zeros(coord.shape[1:], dtype=torch.float32, device=coord.device)
                if torch.any(valid):
                    src = tuple(source[axis][valid].cpu() for axis in range(spatial_rank))
                    dst = tuple(destination[axis][valid].cpu() for axis in range(spatial_rank))
                    values[valid] = (global_seg[src] == global_seg[dst]).to(
                        device=coord.device, dtype=torch.float32
                    )
                channels.append(values)
            outputs.append(torch.stack(channels))
        return torch.stack(outputs).to(device=coordinates.device, dtype=coordinates.dtype)

    return forward


@pytest.mark.parametrize("mode", ["banis", "deepem"])
@pytest.mark.parametrize("blending", ["constant", "bump"])
@pytest.mark.parametrize("output_dtype", ["float32", "float16"])
def test_patch_first_affinity_tta_matches_standard_without_internal_seams(
    mode, blending, output_dtype
):
    offsets = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    segmentation = _segmentation((9, 9, 9))
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(size) for size in segmentation.shape), indexing="ij")
    ).float()[None]
    cfg = _affinity_cfg(offsets, mode)
    cfg.inference.model.output_dtype = output_dtype
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = "all"
    cfg.inference.test_time_augmentation.rotation90_axes = None
    cfg.inference.sliding_window.window_size = [5, 5, 5]
    cfg.inference.sliding_window.sw_batch_size = 2
    cfg.inference.sliding_window.overlap = 0.6
    cfg.inference.sliding_window.blending = blending
    oracle = _coordinate_oracle(offsets, mode, segmentation)

    standard_cfg = copy.deepcopy(cfg)
    standard_cfg.inference.test_time_augmentation.patch_first_local = False
    standard = TTAPredictor(
        standard_cfg,
        build_sliding_inferer(standard_cfg),
        oracle,
    ).predict(coordinates)

    patch_cfg = copy.deepcopy(cfg)
    patch_cfg.inference.test_time_augmentation.patch_first_local = True
    patch = TTAPredictor(
        patch_cfg,
        build_sliding_inferer(patch_cfg),
        oracle,
    ).predict(coordinates)

    assert torch.equal(patch, standard)
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode=mode)
    canonical_valid = compute_affinity_valid_mask(
        offsets,
        segmentation.shape,
        affinity_mode=mode,
    )
    expected = torch.from_numpy(canonical.values).to(patch.dtype)
    assert torch.equal(patch[0][canonical_valid], expected[canonical_valid])


@pytest.mark.parametrize("mode", ["banis", "deepem"])
def test_patch_first_affinity_tta_matches_standard_for_all_quarter_turns(mode):
    offsets = [(2, 0, 0), (0, 2, 0), (0, 0, 2)]
    segmentation = _segmentation((9, 9, 9))
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(size) for size in segmentation.shape), indexing="ij")
    ).float()[None]
    cfg = _affinity_cfg(offsets, mode)
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = [[1, 2]]
    cfg.inference.test_time_augmentation.rotate90_k = [0, 1, 2, 3]
    cfg.inference.sliding_window.window_size = [5, 5, 5]
    cfg.inference.sliding_window.sw_batch_size = 2
    cfg.inference.sliding_window.overlap = 0.6
    cfg.inference.sliding_window.blending = "constant"
    oracle = _coordinate_oracle(offsets, mode, segmentation)

    standard_cfg = copy.deepcopy(cfg)
    standard_cfg.inference.test_time_augmentation.patch_first_local = False
    standard = TTAPredictor(
        standard_cfg,
        build_sliding_inferer(standard_cfg),
        oracle,
    ).predict(coordinates)

    patch_cfg = copy.deepcopy(cfg)
    patch_cfg.inference.test_time_augmentation.patch_first_local = True
    patch = TTAPredictor(
        patch_cfg,
        build_sliding_inferer(patch_cfg),
        oracle,
    ).predict(coordinates)

    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode=mode)
    expected = torch.from_numpy(canonical.values.astype(np.float32))[None]
    assert torch.equal(patch, standard)
    assert torch.equal(patch, expected)


@pytest.mark.parametrize("patch_first", [False, True])
def test_end_to_end_affinity_tta_rejects_zero_coverage(monkeypatch, patch_first):
    offsets = [(1, 0, 0)]
    segmentation = _segmentation((7, 7, 7))
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(size) for size in segmentation.shape), indexing="ij")
    ).float()[None]
    cfg = _affinity_cfg(offsets, "banis")
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = patch_first
    cfg.inference.test_time_augmentation.flip_axes = [0]
    cfg.inference.test_time_augmentation.rotation90_axes = None
    cfg.inference.sliding_window.window_size = [5, 5, 5]
    cfg.inference.sliding_window.sw_batch_size = 2
    cfg.inference.sliding_window.overlap = 0.6
    cfg.inference.sliding_window.blending = "constant"
    monkeypatch.setattr(
        TTAPredictor,
        "_build_augmentation_combinations",
        lambda self, tta_cfg, ndim: [([0], None, 0)],
    )
    predictor = TTAPredictor(
        cfg,
        build_sliding_inferer(cfg),
        _coordinate_oracle(offsets, "banis", segmentation),
    )

    with pytest.raises(RuntimeError, match="channel 0.*voxel index"):
        predictor.predict(coordinates)


def test_named_head_target_slices_map_affinity_and_leave_scalar_head_spatial_only():
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cfg = _affinity_cfg(offsets, "banis", extra_targets=[{"name": "binary"}], out_channels=4)
    cfg.model.heads = {
        "aff": ModelHeadConfig(out_channels=3, target_slice="0:3"),
        "scalar": ModelHeadConfig(out_channels=1, target_slice="3:4"),
    }
    affinity_plan = build_affinity_tta_plan(
        cfg,
        augmentation_combinations=[([0], None, 0)],
        num_raw=3,
        requested_head="aff",
    )
    scalar_plan = build_affinity_tta_plan(
        cfg,
        augmentation_combinations=[([0], None, 0)],
        num_raw=1,
        requested_head="scalar",
    )
    assert affinity_plan is not None and affinity_plan.partial_channels == frozenset({0})
    assert scalar_plan is not None
    assert scalar_plan.views[0].moves == ()


def test_named_affinity_and_scalar_heads_are_corrected_by_declared_target_slice():
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    segmentation = _segmentation((9, 10, 11))
    cfg = _affinity_cfg(offsets, "banis", extra_targets=[{"name": "binary"}], out_channels=4)
    cfg.model.heads = {
        "aff": ModelHeadConfig(out_channels=3, target_slice="0:3"),
        "scalar": ModelHeadConfig(out_channels=1, target_slice="3:4"),
    }
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = False
    cfg.inference.test_time_augmentation.flip_axes = [0]
    cfg.inference.test_time_augmentation.rotation90_axes = None

    affinity_forward = _configured_oracle(offsets, "banis", segmentation)

    def forward(x):
        return {"output": {"aff": affinity_forward(x), "scalar": x[:, 0:1]}}

    predictor = TTAPredictor(cfg, sliding_inferer=None, forward_fn=forward)
    image = torch.from_numpy(segmentation).float()[None, None]
    affinity = predictor.predict(image, requested_head="aff")
    scalar = predictor.predict(image, requested_head="scalar")
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode="banis")
    assert torch.equal(
        affinity,
        torch.from_numpy(canonical.values.astype(np.float32))[None],
    )
    assert torch.equal(scalar, image)


def test_mixed_patch_output_keeps_scalar_on_legacy_normalization_path(monkeypatch):
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    segmentation = _segmentation((7, 7, 7))
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(size) for size in segmentation.shape), indexing="ij")
    ).float()[None]
    cfg = _affinity_cfg(offsets, "banis", extra_targets=[{"name": "binary"}], out_channels=4)
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = True
    cfg.inference.test_time_augmentation.flip_axes = [0]
    cfg.inference.test_time_augmentation.rotation90_axes = None
    cfg.inference.sliding_window.window_size = [5, 5, 5]
    cfg.inference.sliding_window.sw_batch_size = 2
    cfg.inference.sliding_window.overlap = 0.6
    cfg.inference.sliding_window.blending = "constant"
    affinity_forward = _coordinate_oracle(offsets, "banis", segmentation)

    def forward(x):
        return torch.cat([affinity_forward(x), x[:, 0:1]], dim=1)

    normalization_calls = []
    original_normalize = tta_module.normalize_weighted_accumulator

    def tracking_normalize(values, weights):
        normalization_calls.append((values.shape, weights.shape))
        return original_normalize(values, weights)

    monkeypatch.setattr(tta_module, "normalize_weighted_accumulator", tracking_normalize)
    result = TTAPredictor(
        cfg,
        build_sliding_inferer(cfg),
        forward,
    ).predict(coordinates)

    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode="banis")
    valid = compute_affinity_valid_mask(offsets, segmentation.shape, affinity_mode="banis")
    expected_affinity = torch.from_numpy(canonical.values).to(result.dtype)
    assert torch.equal(result[0, :3][valid], expected_affinity[valid])
    assert torch.equal(result[:, 3:4], coordinates[:, 0:1])
    assert normalization_calls


def test_identity_only_affinity_tta_retains_the_fully_valid_fast_path():
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    segmentation = _segmentation((7, 8, 9))
    cfg = _affinity_cfg(offsets, "banis")
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.flip_axes = None
    cfg.inference.test_time_augmentation.rotation90_axes = None
    predictor = TTAPredictor(
        cfg,
        sliding_inferer=None,
        forward_fn=_configured_oracle(offsets, "banis", segmentation),
    )
    combinations = predictor._build_augmentation_combinations(
        cfg.inference.test_time_augmentation,
        ndim=5,
    )
    plan, lazy = predictor._prepare_affinity_plan(combinations)
    assert lazy is False
    assert plan is not None and plan.partial_channels == frozenset()
    result = predictor.predict(torch.from_numpy(segmentation).float()[None, None])
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode="banis")
    assert torch.equal(result, torch.from_numpy(canonical.values.astype(np.float32))[None])


def test_permutation_only_affinity_view_has_no_partial_channels():
    offsets = [(1, 0, 0), (-1, 0, 0)]
    segmentation = _segmentation((7, 8, 9))
    cfg = _affinity_cfg(offsets, "banis")
    cfg.inference.test_time_augmentation.enabled = True
    cfg.inference.test_time_augmentation.patch_first_local = False
    cfg.inference.test_time_augmentation.flip_axes = [0]
    cfg.inference.test_time_augmentation.rotation90_axes = None
    predictor = TTAPredictor(
        cfg,
        sliding_inferer=None,
        forward_fn=_configured_oracle(offsets, "banis", segmentation),
    )
    combinations = predictor._build_augmentation_combinations(
        cfg.inference.test_time_augmentation,
        ndim=5,
    )
    plan, lazy = predictor._prepare_affinity_plan(combinations)
    assert lazy is False
    assert plan is not None and plan.partial_channels == frozenset()
    result = predictor.predict(torch.from_numpy(segmentation).float()[None, None])
    canonical = seg_to_affinity(segmentation, offsets=offsets, affinity_mode="banis")
    assert torch.equal(result, torch.from_numpy(canonical.values.astype(np.float32))[None])


def test_ambiguous_head_and_channel_mappings_raise_actionable_errors():
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    cfg = _affinity_cfg(offsets, "banis", extra_targets=[{"name": "binary"}], out_channels=4)
    cfg.model.heads = {"aff": ModelHeadConfig(out_channels=3)}
    with pytest.raises(ValueError, match=r"model\.heads\.aff\.target_slice"):
        build_affinity_tta_plan(
            cfg,
            augmentation_combinations=[([], None, 0)],
            num_raw=3,
            requested_head="aff",
        )

    cfg.model.heads["aff"].target_slice = "0:2"
    with pytest.raises(ValueError, match="width 2"):
        build_affinity_tta_plan(
            cfg,
            augmentation_combinations=[([], None, 0)],
            num_raw=3,
            requested_head="aff",
        )

    # A head selecting a contiguous SUB-RANGE of a group is legal: offsets are
    # positional, so the sub-range's offsets are unambiguous. What must still fail is a
    # sub-range that is not closed under the requested transform group -- and only the
    # per-view bijection check can see that, not the shape of the config.
    cfg.model.heads["aff"].out_channels = 2
    cfg.model.heads["aff"].target_slice = "1:3"
    plan = build_affinity_tta_plan(
        cfg,
        augmentation_combinations=[([], None, 0)],
        num_raw=2,
        requested_head="aff",
    )
    assert plan is not None and len(plan.views) == 1

    cfg.model.heads["aff"].out_channels = 1
    cfg.model.heads["aff"].target_slice = "2:3"
    with pytest.raises(ValueError, match="counterpart"):
        build_affinity_tta_plan(
            cfg,
            # an XY quarter-turn sends (0,0,1) to (0,-1,0), which a one-offset
            # sub-range cannot supply
            augmentation_combinations=[([], (1, 2), 1)],
            num_raw=1,
            requested_head="aff",
        )

    no_head = _affinity_cfg(offsets, "banis", out_channels=4)
    with pytest.raises(ValueError, match="all three must match"):
        build_affinity_tta_plan(
            no_head,
            augmentation_combinations=[([], None, 0)],
            num_raw=4,
            requested_head=None,
        )


@pytest.mark.parametrize("head,expected", [
    ("aff_r1", [(0, 0, 1), (0, 1, 0), (1, 0, 0)]),
    ("aff_r5", [(0, 0, 5), (0, 5, 0), (5, 0, 0)]),
    ("aff_r9", [(0, 0, 9), (0, 9, 0), (9, 0, 0)]),
])
def test_multi_radius_group_split_across_heads_by_target_slice(head, expected):
    """One affinity target, many radii, split across heads -- the SNEMI/BANIS shape.

    `resolve_affinity_channel_groups_from_cfg` returns ONE group spanning label
    channels [0,9) for a single 9-offset affinity target, while the model exposes it
    as three 3-channel heads. Selecting one head must restrict the group to that
    head's positional sub-range rather than refusing it.
    """
    offsets = [
        (0, 0, 1), (0, 1, 0), (1, 0, 0),
        (0, 0, 5), (0, 5, 0), (5, 0, 0),
        (0, 0, 9), (0, 9, 0), (9, 0, 0),
    ]
    cfg = _affinity_cfg(offsets, "deepem", extra_targets=[{"name": "binary"}], out_channels=3)
    cfg.model.heads = {
        "aff_r1": ModelHeadConfig(out_channels=3, target_slice="0:3"),
        "aff_r5": ModelHeadConfig(out_channels=3, target_slice="3:6"),
        "aff_r9": ModelHeadConfig(out_channels=3, target_slice="6:9"),
    }
    # Kisuk's 16-view group: XY dihedral (4 rotations x XY flip) x z-flip.
    combos = [
        (flips, (1, 2), k)
        for flips in ([], [1], [0], [0, 1])
        for k in (0, 1, 2, 3)
    ]
    plan = build_affinity_tta_plan(cfg, augmentation_combinations=combos,
                                   num_raw=3, requested_head=head)
    assert plan is not None
    assert len(plan.views) == 16
    assert plan.num_channels == 3
    # every view is a bijection over exactly this head's 3 raw channels
    for view in plan.views:
        assert sorted(m.dst for m in view.moves) == [0, 1, 2]
        assert sorted(m.src for m in view.moves) == [0, 1, 2]
    # the z offset only ever re-anchors along z; XY offsets never shift along z
    radius = max(max(abs(v) for v in off) for off in expected)
    for shift in plan.shifts:
        assert max(abs(s) for s in shift) == radius


def test_stacked_label_channel_count_shares_affinity_target_traversal():
    cfg = _affinity_cfg(
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        "banis",
        extra_targets=[
            {"name": "polarity", "kwargs": {"exclusive": False}},
            {"name": "polarity", "kwargs": {"exclusive": True}},
            {"name": "binary"},
        ],
        out_channels=8,
    )
    assert resolve_stacked_label_channel_count(cfg) == 8
    assert resolve_affinity_channel_groups_from_cfg(cfg)[0][0] == (0, 3)


def test_mocked_distributed_partial_mean_reduces_sums_and_per_voxel_counts(monkeypatch):
    cfg = Config()
    predictor = TTAPredictor(cfg, sliding_inferer=None, forward_fn=lambda x: x)
    local = TTAEnsembleAccumulator(
        (1, 1, 1, 1, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        mode_map=["mean"],
        partial_channels=[0],
        distributed_sharding=True,
        max_views=2,
    )
    remote = TTAEnsembleAccumulator(
        (1, 1, 1, 1, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
        mode_map=["mean"],
        partial_channels=[0],
        distributed_sharding=True,
        max_views=2,
    )
    local.add(
        torch.tensor([[[[[2.0, 8.0]]]]]),
        ViewValidity(((slice(0, 1), slice(0, 1), slice(0, 1)),)),
    )
    remote.add(
        torch.tensor([[[[[6.0, 4.0]]]]]),
        ViewValidity(((slice(0, 1), slice(0, 1), slice(0, 2)),)),
    )
    monkeypatch.setattr(predictor, "_distributed_context", lambda: (True, 0, 2))
    monkeypatch.setattr(
        predictor,
        "_apply_distributed_reduction",
        lambda *args, **kwargs: local.legacy_result,
    )

    def reduce_tensor(tensor, *, op, reduction_device):
        del op, reduction_device
        if tensor.dtype == torch.int32:
            return tensor + remote.partial_counts.to(torch.int32)
        return tensor + remote.partial_statistics

    monkeypatch.setattr(predictor, "_reduce_cpu_tensor_to_rank_zero", reduce_tensor)
    result = predictor._apply_distributed_accumulator_reduction(
        local,
        "mean",
        torch.device("cpu"),
    )
    assert result is not None
    assert torch.equal(result, torch.tensor([[[[[4.0, 4.0]]]]]))


def test_mocked_distributed_mixed_modes_use_stable_collective_order(monkeypatch):
    cfg = Config()
    predictor = TTAPredictor(cfg, sliding_inferer=None, forward_fn=lambda x: x)
    accumulator = TTAEnsembleAccumulator(
        (1, 6, 1, 1, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
        mode_map=["mean", "mean", "mean", "min", "min", "min"],
        partial_channels=range(6),
        distributed_sharding=True,
        max_views=2,
    )
    operations = []
    monkeypatch.setattr(predictor, "_distributed_context", lambda: (True, 1, 2))
    monkeypatch.setattr(
        predictor,
        "_validate_distributed_reduction_shape",
        lambda *args, **kwargs: None,
    )

    def record_tensor_reduction(tensor, *, op, reduction_device):
        del tensor, reduction_device
        operations.append(op)
        return None

    def record_count_reduction(count, *, reduction_device):
        del count, reduction_device
        operations.append("count")
        return 0

    monkeypatch.setattr(predictor, "_reduce_cpu_tensor_to_rank_zero", record_tensor_reduction)
    monkeypatch.setattr(
        predictor,
        "_reduce_prediction_count_to_rank_zero",
        record_count_reduction,
    )

    result = predictor._apply_distributed_accumulator_reduction(
        accumulator,
        [["0:3", "mean"], ["3:6", "min"]],
        torch.device("cpu"),
    )

    assert result is None
    assert operations == [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        "count",
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.SUM,
    ]


def test_mocked_distributed_partial_min_max_ignore_locally_uncovered_voxels(monkeypatch):
    cfg = Config()
    predictor = TTAPredictor(cfg, sliding_inferer=None, forward_fn=lambda x: x)
    accumulator_kwargs = {
        "shape": (1, 2, 1, 1, 2),
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "mode_map": ["min", "max"],
        "partial_channels": [0, 1],
        "distributed_sharding": True,
        "max_views": 2,
    }
    local = TTAEnsembleAccumulator(**accumulator_kwargs)
    remote = TTAEnsembleAccumulator(**accumulator_kwargs)
    first_voxel_only = (slice(0, 1), slice(0, 1), slice(0, 1))
    local.add(
        torch.tensor([[[[[4.0, 100.0]]], [[[2.0, -100.0]]]]]),
        ViewValidity((first_voxel_only, first_voxel_only)),
    )
    remote.add(
        torch.tensor([[[[[6.0, 8.0]]], [[[1.0, 5.0]]]]]),
        ViewValidity.all_valid(2),
    )
    monkeypatch.setattr(predictor, "_distributed_context", lambda: (True, 0, 2))
    monkeypatch.setattr(
        predictor,
        "_apply_distributed_reduction",
        lambda *args, **kwargs: local.legacy_result,
    )

    def reduce_tensor(tensor, *, op, reduction_device):
        del reduction_device
        if tensor.dtype == torch.int32:
            return tensor + remote.partial_counts.to(torch.int32)
        if op == torch.distributed.ReduceOp.MIN:
            return torch.minimum(tensor, remote.partial_statistics)
        if op == torch.distributed.ReduceOp.MAX:
            return torch.maximum(tensor, remote.partial_statistics)
        raise AssertionError(f"Unexpected reduction op: {op}")

    monkeypatch.setattr(predictor, "_reduce_cpu_tensor_to_rank_zero", reduce_tensor)
    result = predictor._apply_distributed_accumulator_reduction(
        local,
        [["0", "min"], ["1", "max"]],
        torch.device("cpu"),
    )

    assert result is not None
    assert torch.equal(result[0, 0, 0, 0], torch.tensor([4.0, 8.0]))
    assert torch.equal(result[0, 1, 0, 0], torch.tensor([2.0, 5.0]))
