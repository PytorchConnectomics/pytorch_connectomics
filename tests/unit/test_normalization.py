import numpy as np
import pytest

from connectomics.data.augmentation.augment_ops import smart_normalize
from connectomics.data.normalization import (
    alignment_attrs,
    decode_aligned,
    display_from_codes,
    encode_aligned,
    estimate_noise_model,
    fit_alignment,
    flat_field_image,
    neuroglancer_shader,
    to_aligned,
)
from connectomics.data.normalization import (
    codes_to_u16,
    quantization_report,
    require_codec,
)
from connectomics.data.normalization.codec import CODEC_R_MAX, codec_attrs


def _shot_noise_volume(rng, gain=6.0, offset=140.0, shape=(4, 256, 256), vignette=2.0):
    """Poisson photons on a blob pattern under a smooth vignette, camera-scaled.

    ``vignette=2`` puts the corners at exp(-1) ~ 0.37 of the center, close to the
    ~0.45 measured on ExPID96.
    """
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    radius2 = ((y - shape[1] / 2) / shape[1]) ** 2 + ((x - shape[2] / 2) / shape[2]) ** 2
    illumination = np.exp(-vignette * radius2)
    photons = illumination * (8.0 + 40.0 * (0.5 + 0.5 * np.sin(y / 9.0) * np.cos(x / 13.0 + z)))
    return (gain * rng.poisson(photons) + offset).astype(np.uint16), illumination[0]


def test_estimate_noise_model_recovers_gain_and_offset():
    volume, _ = _shot_noise_volume(np.random.default_rng(0), vignette=0.0)

    gain, offset = estimate_noise_model(list(volume))

    assert gain == pytest.approx(6.0, rel=0.15)
    assert offset == pytest.approx(140.0, abs=15.0)


def test_estimate_noise_model_refuses_noise_free_input():
    ramp = np.tile(np.linspace(100, 1000, 256), (256, 1))

    with pytest.raises(ValueError, match="noise"):
        estimate_noise_model([ramp] * 4)


def test_fit_alignment_recovers_vignette_and_anchors_tissue_median_at_one():
    volume, illumination = _shot_noise_volume(np.random.default_rng(1))

    alignment = fit_alignment(volume, stats_planes=4, noise_model=(6.0, 140.0))

    flat = flat_field_image(alignment, volume.shape[1:])
    expected = illumination / np.median(illumination)
    assert np.abs(flat / expected - 1.0).max() < 0.1
    aligned = to_aligned(volume[1], alignment, flat)
    assert np.median(aligned) == pytest.approx(1.0, abs=0.1)  # all tissue here


def test_fit_alignment_keeps_all_tissue_of_a_dim_acquisition():
    """A dim volume (tissue ~6 gain units) must not be reduced to its brightest pixels."""
    rng = np.random.default_rng(5)
    shape = (4, 256, 256)
    z, y, x = np.meshgrid(*(np.arange(n) for n in shape), indexing="ij")
    photons = 6.0 * (0.6 + 0.4 * np.sin(y / 7.0) * np.cos(x / 11.0 + z))
    volume = (6.0 * rng.poisson(photons) + 114.0).astype(np.uint16)

    alignment = fit_alignment(volume, stats_planes=4, noise_model=(6.0, 114.0))

    assert alignment["tissue_fraction"] > 0.95
    assert alignment["flat_field"] is not None
    assert alignment["tissue_median_signal"] == pytest.approx(
        np.median(volume.astype(np.float32) - 114.0), rel=0.1
    )


def test_fit_alignment_excludes_empty_regions_from_the_anchor():
    rng = np.random.default_rng(6)
    volume, _ = _shot_noise_volume(rng, vignette=0.0)
    empty = (6.0 * rng.poisson(0.0, size=volume[:, :, :96].shape) + 140).astype(np.uint16)
    volume[:, :, :96] = empty  # resin: only camera offset

    alignment = fit_alignment(volume, stats_planes=4, noise_model=(6.0, 140.0), flat_field=False)

    assert alignment["tissue_fraction"] == pytest.approx(160 / 256, abs=0.05)
    tissue = volume[:, :, 128:].astype(np.float32) - 140.0
    assert alignment["tissue_median_signal"] == pytest.approx(np.median(tissue), rel=0.1)


def test_alignment_puts_differently_acquired_copies_on_one_profile():
    """Same specimen, different gain/offset/brightness -> the same aligned codes."""
    rng = np.random.default_rng(2)
    dim, _ = _shot_noise_volume(rng, gain=6.0, offset=140.0)
    photons = (dim.astype(np.float64) - 140.0) / 6.0
    bright = (20.0 * photons * 3.0 + 100.0).astype(np.uint16)  # 3x light, gain 20

    codes = []
    for volume, model in ((dim, (6.0, 140.0)), (bright, (60.0, 100.0))):
        alignment = fit_alignment(volume, stats_planes=4, noise_model=model)
        codes.append(encode_aligned(to_aligned(volume[0], alignment)).astype(int))
    assert np.median(np.abs(codes[0] - codes[1])) <= 1


def test_code_table_round_trips_within_a_fraction_of_shot_noise():
    rng = np.random.default_rng(3)
    for photons in (5.0, 20.0, 100.0):
        aligned = rng.poisson(photons, size=100_000) / photons
        error = decode_aligned(encode_aligned(aligned)) - aligned
        noise = np.sqrt(np.maximum(aligned, 1.0 / photons) / photons)
        assert np.sqrt(np.mean((error / noise) ** 2)) < 0.25


def test_code_table_is_fixed_monotone_and_saturates_outside_its_range():
    codes = np.arange(256)
    aligned = decode_aligned(codes)
    assert aligned[0] == pytest.approx(0.0, abs=1e-6)
    assert aligned[-1] == pytest.approx(CODEC_R_MAX)
    assert np.all(np.diff(aligned) > 0)
    np.testing.assert_array_equal(encode_aligned(aligned), codes)
    assert encode_aligned(np.array([-5.0, 1e6])).tolist() == [0, 255]


def test_display_blackens_the_noise_floor_and_is_one_shader():
    display = display_from_codes(np.arange(256, dtype=np.uint8))
    assert display[encode_aligned(np.array([0.5]))[0]] == 0
    assert display[-1] == 255
    assert np.all(np.diff(display.astype(int)) >= 0)
    shader = neuroglancer_shader()
    assert "emitGrayscale" in shader and "toNormalized" in shader
    # Overlapping black/white sliders must not divide by zero or a negative width.
    assert "max(white - black" in shader and "/ (white - black)" not in shader


def test_alignment_attrs_are_json_ready_and_self_describing():
    import json

    volume, _ = _shot_noise_volume(np.random.default_rng(4))
    attrs = alignment_attrs(fit_alignment(volume, stats_planes=4))

    json.dumps(attrs)
    assert attrs["intensity_mode"] == "aligned"
    assert attrs["codec"] == "sqrt_aligned_u8_v1"
    assert attrs["flat_field_model"] == "exp_quadratic_yx"
    assert len(attrs["flat_field_coefficients"]) == 6


def test_smart_normalize_aligned_u8_decodes_without_patch_statistics():
    codes = np.array([[0, 30, 255]], dtype=np.uint8)

    decoded = smart_normalize(codes, "aligned-u8")

    np.testing.assert_allclose(decoded, decode_aligned(codes))
    # Same codes in a different patch decode identically: no per-patch rescaling.
    np.testing.assert_allclose(smart_normalize(codes[:, :2], "aligned-u8"), decoded[:, :2])


def test_v1_clips_negative_aligned_values_to_code_zero():
    # The knee does not preserve values below the fitted offset: all become 0.
    assert encode_aligned(np.array([-0.25, -0.1, -1e-4, 0.0])).tolist() == [0, 0, 0, 0]
    report = quantization_report(np.array([-0.2, 0.0, 1.0, 40.0], dtype=np.float32))
    assert report["clipped_low"] == pytest.approx(0.25)
    assert report["clipped_high"] == pytest.approx(0.25)
    assert report["code0"] == pytest.approx(0.5)  # endpoint occupancy includes r = 0
    assert report["code255"] == pytest.approx(0.25)


def test_every_code_round_trips_exactly_and_stays_in_the_declared_range():
    codes = np.arange(256, dtype=np.uint8)
    aligned = decode_aligned(codes)
    np.testing.assert_array_equal(encode_aligned(aligned), codes)
    assert aligned.min() >= 0.0 and aligned.max() <= CODEC_R_MAX + 1e-4
    # Every code is the unique nearest code for its own decoded value.
    midpoints = decode_aligned(codes[:-1].astype(np.float32) + 0.5)
    np.testing.assert_array_equal(encode_aligned(midpoints - 1e-4), codes[:-1])


def test_optional_uint16_cache_keeps_every_code_distinct():
    codes = np.arange(256, dtype=np.uint8)
    u16 = codes_to_u16(codes)
    assert u16.dtype == np.uint16 and np.unique(u16).size == 256
    assert np.diff(u16.astype(int)).min() >= 41
    np.testing.assert_allclose(
        smart_normalize(u16.astype(np.float32), "divide-2000"), decode_aligned(codes), atol=2.5e-4
    )
    with pytest.raises(TypeError):
        codes_to_u16(codes.astype(np.int32))


def test_require_codec_rejects_unknown_versions():
    require_codec(codec_attrs())
    with pytest.raises(ValueError, match="Unsupported"):
        require_codec({"codec": "sqrt_aligned_u8_v2"})
    with pytest.raises(ValueError, match="Unsupported"):
        require_codec({})
    with pytest.raises(ValueError, match="codec_r_max"):
        require_codec({**codec_attrs(), "codec_r_max": 16.0})


def test_aligned_u8_rejects_percentile_clipping():
    from connectomics.data.augmentation.transforms import SmartNormalizeIntensityd

    codes = np.array([[0, 30, 255]], dtype=np.uint8)
    with pytest.raises(ValueError, match="clip_percentile"):
        smart_normalize(codes, "aligned-u8", clip_percentile_low=0.01)
    with pytest.raises(ValueError, match="clip_percentile"):
        SmartNormalizeIntensityd(keys=["image"], mode="aligned-u8", clip_percentile_high=0.99)
