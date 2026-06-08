import os
import time

import numpy as np
import pytest

from connectomics.data.processing.lsd import LsdExtractor

ATOL = 1e-5
RTOL = 1e-4


def _blob_labels(shape, count, seed, radius_range=(2, 5)):
    rng = np.random.default_rng(seed)
    segmentation = np.zeros(shape, dtype=np.int32)
    grids = np.ogrid[tuple(slice(0, size) for size in shape)]

    for label in range(1, count + 1):
        radii = rng.integers(radius_range[0], radius_range[1] + 1, size=len(shape))
        center = tuple(int(rng.integers(0, size)) for size in shape)
        dist = np.zeros(shape, dtype=np.float32)
        for grid, c, r in zip(grids, center, radii):
            dist += ((grid - c) / float(r)) ** 2
        segmentation[dist <= 1.0] = label

    return segmentation


def _full_descriptors(
    segmentation,
    sigma,
    *,
    components=None,
    voxel_size=None,
    labels=None,
    mode="gaussian",
):
    extractor = LsdExtractor(sigma, mode=mode, downsample=1)
    dims = segmentation.ndim
    if dims == 2 and len(extractor.sigma) == 3:
        extractor.sigma = extractor.sigma[:2]

    voxel_size_t = (
        tuple(1 for _ in range(dims)) if voxel_size is None else tuple(int(v) for v in voxel_size)
    )
    labels_arr = np.unique(segmentation) if labels is None else np.asarray(list(labels))
    num_channels = (10 if dims == 3 else 6) if components is None else len(components)
    descriptors = np.zeros((num_channels,) + segmentation.shape, dtype=np.float32)

    df = extractor.downsample
    if any(s % df != 0 for s in segmentation.shape):
        raise ValueError(
            f"segmentation shape {segmentation.shape} is not divisible by "
            f"downsample factor {df}"
        )
    sub_voxel_size = tuple(v * df for v in voxel_size_t)
    sub_sigma_voxel = tuple(s / v for s, v in zip(extractor.sigma, sub_voxel_size))

    extractor._accumulate_full(
        descriptors,
        segmentation,
        labels_arr,
        sub_sigma_voxel,
        sub_voxel_size,
        components,
        df,
        dims,
    )

    if extractor.mode == "gaussian":
        max_distance = np.asarray(extractor.sigma, dtype=np.float32)
    else:
        max_distance = np.asarray([0.5 * s for s in extractor.sigma], dtype=np.float32)

    seg_mask = (segmentation != 0).astype(np.float32)
    if dims == 3:
        extractor._normalize_3d(descriptors, max_distance, seg_mask, components)
    else:
        extractor._normalize_2d(descriptors, max_distance, seg_mask, components)

    np.clip(descriptors, 0.0, 1.0, out=descriptors)
    return descriptors


def _bbox_descriptors(
    segmentation,
    sigma,
    *,
    components=None,
    voxel_size=None,
    labels=None,
    mode="gaussian",
):
    return LsdExtractor(sigma, mode=mode, downsample=1).get_descriptors(
        segmentation, components=components, voxel_size=voxel_size, labels=labels
    )


def _assert_matches_full(
    segmentation,
    sigma,
    *,
    components=None,
    voxel_size=None,
    labels=None,
    mode="gaussian",
):
    actual = _bbox_descriptors(
        segmentation,
        sigma,
        components=components,
        voxel_size=voxel_size,
        labels=labels,
        mode=mode,
    )
    expected = _full_descriptors(
        segmentation,
        sigma,
        components=components,
        voxel_size=voxel_size,
        labels=labels,
        mode=mode,
    )
    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=ATOL, rtol=RTOL), max_diff


def test_bbox_matches_full_for_3d_gaussian_components_none():
    segmentation = _blob_labels((48, 48, 48), count=12, seed=11)

    _assert_matches_full(segmentation, sigma=(4.0, 4.0, 4.0))


def test_bbox_matches_full_for_3d_subset_and_non_unit_voxel_size():
    segmentation = _blob_labels((64, 64, 64), count=16, seed=23)

    _assert_matches_full(
        segmentation,
        sigma=(6.0, 4.0, 4.0),
        components="0129",
        voxel_size=(2, 1, 1),
    )


def test_bbox_matches_full_for_3d_full_components_non_unit_voxel_size():
    segmentation = _blob_labels((32, 36, 40), count=6, seed=31)

    _assert_matches_full(
        segmentation,
        sigma=(6.0, 4.0, 4.0),
        voxel_size=(2, 1, 1),
    )


def test_bbox_matches_full_for_2d_gaussian():
    segmentation = _blob_labels((64, 72), count=10, seed=41)

    _assert_matches_full(segmentation, sigma=(5.0, 4.0))


def test_bbox_handles_empty_volume_and_absent_explicit_label():
    segmentation = np.zeros((24, 24, 24), dtype=np.int32)

    _assert_matches_full(segmentation, sigma=(3.0, 3.0, 3.0), labels=[99])
    actual = _bbox_descriptors(segmentation, sigma=(3.0, 3.0, 3.0))
    assert np.count_nonzero(actual) == 0


def test_bbox_matches_full_for_single_label_filling_volume():
    segmentation = np.ones((24, 24, 24), dtype=np.int32)

    _assert_matches_full(segmentation, sigma=(3.0, 3.0, 3.0))


def test_bbox_matches_full_for_border_touching_label():
    segmentation = np.zeros((32, 32, 32), dtype=np.int32)
    segmentation[:6, :8, :10] = 1
    segmentation[14:22, 15:24, 16:25] = 2

    _assert_matches_full(segmentation, sigma=(4.0, 4.0, 4.0))


def test_bbox_matches_full_for_sparse_large_label_ids():
    segmentation = np.zeros((30, 32, 34), dtype=np.int32)
    segmentation[3:8, 4:10, 5:12] = 1000

    _assert_matches_full(
        segmentation,
        sigma=(3.0, 3.0, 3.0),
        labels=[1000, 2000],
    )


def test_bbox_matches_full_for_float32_label_array():
    segmentation = _blob_labels((32, 32, 32), count=6, seed=47).astype(np.float32)

    _assert_matches_full(segmentation, sigma=(4.0, 4.0, 4.0))


def test_bbox_matches_full_for_3d_sphere():
    segmentation = _blob_labels((32, 32, 32), count=5, seed=53)

    _assert_matches_full(segmentation, sigma=(3.0, 3.0, 3.0), mode="sphere")


@pytest.mark.skipif(
    os.environ.get("LSD_BENCH") != "1",
    reason="set LSD_BENCH=1 to run the LSD bbox timing benchmark",
)
def test_bbox_timing_benchmark():
    segmentation = _blob_labels((128, 128, 128), count=30, seed=67)
    sigma = (8.0, 8.0, 8.0)

    start = time.perf_counter()
    expected = _full_descriptors(segmentation, sigma)
    full_time = time.perf_counter() - start

    start = time.perf_counter()
    actual = _bbox_descriptors(segmentation, sigma)
    bbox_time = time.perf_counter() - start

    max_diff = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    assert np.allclose(actual, expected, atol=ATOL, rtol=RTOL), max_diff

    speedup = full_time / bbox_time
    print("LSD_BENCH " f"full={full_time:.4f}s bbox={bbox_time:.4f}s speedup={speedup:.2f}x")
    assert speedup > 1.0
