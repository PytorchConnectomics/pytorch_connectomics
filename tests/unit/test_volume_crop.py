"""`data.<split>.crop`: keep only a sub-volume, identically across all keys.

Motivation (SegEM): the annotation covers a 100^3 centre of a 150x200x200 cube, so
uniform patches spend ~half of every loss mask on the -1 ignore sentinel while the
run is GPU-bound. Cropping to the annotated region plus a halo doubles the
supervised fraction at identical compute -- but only if the crop is applied to
image and label alike and a too-small crop fails loudly instead of being padded
back up with filler.
"""

import numpy as np
import pytest

from connectomics.data.datasets.dataset_volume_cached import (
    CachedVolumeDataset,
    _parse_volume_crop,
)


def _volumes():
    z, y, x = np.meshgrid(np.arange(20), np.arange(24), np.arange(28), indexing="ij")
    return {
        "image": (z * 10000 + y * 100 + x).astype(np.float32),
        "label": (z * 10000 + y * 100 + x).astype(np.float32) + 0.5,
        "mask": np.ones((20, 24, 28), dtype=np.float32),
    }


@pytest.fixture
def patched_read(monkeypatch):
    arrays = _volumes()
    monkeypatch.setattr(
        "connectomics.data.datasets.dataset_volume_cached.read_volume",
        lambda path: arrays[path].copy(),
    )
    return arrays


def test_crop_keeps_only_the_requested_subvolume(patched_read):
    dataset = CachedVolumeDataset(
        image_paths=["image"],
        label_paths=["label"],
        mask_paths=["mask"],
        patch_size=(4, 4, 4),
        volume_crop=[5, 15, 6, 18, 7, 21],
        mode="train",
    )

    assert dataset.cached_images[0].shape == (1, 10, 12, 14)
    assert dataset.cached_labels[0].shape == (1, 10, 12, 14)
    assert dataset.cached_masks[0].shape == (1, 10, 12, 14)
    # The kept block is the requested one, not an off-by-one neighbour: the
    # synthetic volume encodes its own coordinates.
    np.testing.assert_array_equal(
        dataset.cached_images[0][0], patched_read["image"][5:15, 6:18, 7:21]
    )
    # image and label must be cropped identically or every target is misaligned
    np.testing.assert_array_equal(dataset.cached_labels[0][0] - 0.5, dataset.cached_images[0][0])


def test_no_crop_keeps_the_whole_volume(patched_read):
    dataset = CachedVolumeDataset(
        image_paths=["image"],
        label_paths=["label"],
        patch_size=(4, 4, 4),
        mode="train",
    )
    assert dataset.cached_images[0].shape == (1, 20, 24, 28)


def test_crop_beyond_the_volume_raises(patched_read):
    with pytest.raises(ValueError, match="spans only"):
        CachedVolumeDataset(
            image_paths=["image"],
            patch_size=(4, 4, 4),
            volume_crop=[0, 10, 0, 10, 0, 99],
            mode="train",
        )


@pytest.mark.parametrize(
    "crop, message",
    [
        ([0, 10, 0, 10, 0], "start, stop"),  # odd length
        ([0, 10, 8, 8, 0, 10], "start < stop"),  # empty axis
        ([0, 10, 12, 4, 0, 10], "start < stop"),  # reversed axis
        ([-1, 10, 0, 10, 0, 10], "start < stop"),  # negative start
    ],
)
def test_malformed_crop_raises(crop, message):
    with pytest.raises(ValueError, match=message):
        _parse_volume_crop(crop)


def test_parse_volume_crop_roundtrip():
    assert _parse_volume_crop([13, 137, 38, 162, 38, 162]) == ((13, 137), (38, 162), (38, 162))
    assert _parse_volume_crop(None) is None


def test_crop_releases_the_parent_array(patched_read):
    """A view would keep the full-size volume alive and save no cache memory."""
    dataset = CachedVolumeDataset(
        image_paths=["image"],
        patch_size=(4, 4, 4),
        volume_crop=[5, 15, 6, 18, 7, 21],
        mode="train",
    )
    cached = dataset.cached_images[0]
    assert cached.base is None, "cropped volume is a view onto the uncropped parent"
    assert cached.flags["C_CONTIGUOUS"]
