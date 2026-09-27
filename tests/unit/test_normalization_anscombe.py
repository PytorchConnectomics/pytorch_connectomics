import numpy as np
import pytest

from connectomics.data.normalization import (
    anscombe_to_uint8,
    anscombe_transform,
    estimate_noise_model,
    fit_anscombe_mapping,
)


def _shot_noise_planes(rng, gain, offset, n_planes=4, size=256):
    y, x = np.mgrid[:size, :size]
    planes = []
    for z in range(n_planes):
        photons = 5.0 + 60.0 * (0.5 + 0.5 * np.sin(y / 23.0) * np.cos(x / 31.0 + z))
        planes.append((gain * rng.poisson(photons) + offset).astype(np.uint16))
    return np.stack(planes)


def test_estimate_noise_model_recovers_gain_and_offset():
    volume = _shot_noise_planes(np.random.default_rng(0), gain=6.0, offset=140.0)

    gain, offset = estimate_noise_model(list(volume))

    assert gain == pytest.approx(6.0, rel=0.15)
    assert offset == pytest.approx(140.0, abs=15.0)


def test_anscombe_transform_gives_unit_noise_at_every_brightness():
    rng = np.random.default_rng(1)
    for photons in (10.0, 40.0, 160.0):
        raw = 6.0 * rng.poisson(photons, size=200_000) + 140.0
        assert anscombe_transform(raw, 6.0, 140.0).std() == pytest.approx(1.0, abs=0.05)


def test_estimate_noise_model_refuses_noise_free_input():
    ramp = np.tile(np.linspace(100, 1000, 256), (256, 1))

    with pytest.raises(ValueError, match="noise"):
        estimate_noise_model([ramp] * 4)


def test_fit_anscombe_mapping_uses_given_model_and_stretch_percentiles():
    volume = _shot_noise_planes(np.random.default_rng(2), gain=6.0, offset=140.0)

    mapping = fit_anscombe_mapping(
        volume, stats_planes=2, stretch_percentiles=(1.0, 99.9), noise_model=(6.0, 140.0)
    )

    assert mapping["noise_model_source"] == "given"
    np.testing.assert_array_equal(mapping["stats_z_indices"], [0, 3])
    sampled = anscombe_transform(volume[[0, 3]], 6.0, 140.0)
    np.testing.assert_allclose(
        mapping["stretch_range"], np.percentile(sampled, (1.0, 99.9)), rtol=1e-5
    )
    out = anscombe_to_uint8(
        volume[0], gain=6.0, offset=140.0, stretch_range=mapping["stretch_range"]
    )
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255


def test_fit_anscombe_mapping_rejects_bad_percentiles():
    volume = _shot_noise_planes(np.random.default_rng(3), gain=6.0, offset=140.0)

    with pytest.raises(ValueError, match="percentiles"):
        fit_anscombe_mapping(
            volume, stats_planes=2, stretch_percentiles=(99.0, 1.0), noise_model=(6.0, 140.0)
        )
