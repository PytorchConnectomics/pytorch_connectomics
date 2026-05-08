from types import SimpleNamespace

import numpy as np

from connectomics.inference.output import apply_prediction_transform, apply_storage_dtype_transform


def test_prediction_transform_applies_independent_of_save():
    cfg = SimpleNamespace(
        inference=SimpleNamespace(
            prediction_transform=SimpleNamespace(
                enabled=True,
                intensity_scale=2.0,
                intensity_dtype="float16",
            ),
        )
    )
    data = np.array([0.25, 0.5], dtype=np.float32)

    transformed = apply_prediction_transform(cfg, data)

    assert transformed.dtype == np.float16
    np.testing.assert_allclose(transformed.astype(np.float32), [0.5, 1.0])


def test_disabled_prediction_transform_keeps_raw_predictions():
    cfg = SimpleNamespace(
        inference=SimpleNamespace(
            prediction_transform=SimpleNamespace(enabled=False),
        )
    )
    data = np.array([0.25, 0.5], dtype=np.float32)

    transformed = apply_prediction_transform(cfg, data)

    assert transformed is data
    np.testing.assert_array_equal(transformed, data)


def test_storage_dtype_transform_casts_when_dtype_specified():
    cfg = SimpleNamespace(
        inference=SimpleNamespace(
            prediction_transform=SimpleNamespace(enabled=False),
            save_dtype="float16",
        )
    )
    data = np.array([0.25, 0.5], dtype=np.float32)

    encoded = apply_storage_dtype_transform(cfg, data)

    assert encoded.dtype == np.float16
    np.testing.assert_allclose(encoded.astype(np.float32), data)


def test_storage_dtype_transform_skips_when_dtype_unspecified():
    cfg = SimpleNamespace(
        inference=SimpleNamespace(
            prediction_transform=SimpleNamespace(enabled=False),
            save_dtype=None,
        )
    )
    data = np.array([0.25, 0.5], dtype=np.float32)

    encoded = apply_storage_dtype_transform(cfg, data)

    assert encoded is data
    assert encoded.dtype == np.float32
