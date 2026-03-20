import numpy as np

from connectomics.data.datasets.dataset_volume_cached import CachedVolumeDataset


def test_cached_volume_dataset_uses_configured_pad_mode_for_label_and_mask(monkeypatch):
    arrays = {
        "image": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "label": np.array([[10, 20], [30, 40]], dtype=np.float32),
        "mask": np.array([[1, 0], [0, 1]], dtype=np.float32),
    }

    monkeypatch.setattr(
        "connectomics.data.datasets.dataset_volume_cached.read_volume",
        lambda path: arrays[path].copy(),
    )

    dataset = CachedVolumeDataset(
        image_paths=["image"],
        label_paths=["label"],
        mask_paths=["mask"],
        patch_size=(6, 6),
        pad_size=(1, 1),
        pad_mode="reflect",
        mode="train",
    )

    expected_image = np.pad(
        np.pad(arrays["image"], ((1, 1), (1, 1)), mode="reflect"),
        ((1, 1), (1, 1)),
        mode="reflect",
    )
    expected_label = np.pad(
        np.pad(arrays["label"], ((1, 1), (1, 1)), mode="reflect"),
        ((1, 1), (1, 1)),
        mode="reflect",
    )
    expected_mask = np.pad(
        np.pad(arrays["mask"], ((1, 1), (1, 1)), mode="reflect"),
        ((1, 1), (1, 1)),
        mode="reflect",
    )

    np.testing.assert_array_equal(dataset.cached_images[0][0], expected_image)
    np.testing.assert_array_equal(dataset.cached_labels[0][0], expected_label)
    np.testing.assert_array_equal(dataset.cached_masks[0][0], expected_mask)
