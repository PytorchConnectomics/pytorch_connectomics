#!/usr/bin/env python3
"""The published layout is the spec; these tests hold the writer to it.

`reference/published_group.json` is the real metadata of
`gs://donglai_public/liconn/moe/expid96/image/ExPID96_2ndgel_S1_40XW001_18x.zarr`,
recorded 2026-09-21. Every assertion below compares against that recording
rather than against a number typed here, so a numcodecs or zarr default that
moves is caught at image-build time instead of after a 6 GB upload.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import zarr_writer as zw  # noqa: E402

REF = json.loads((Path(__file__).parent / "reference/published_group.json").read_text())


def test_reference_fixture_is_the_shape_we_think():
    assert REF["zgroup"] == {"zarr_format": 2}
    assert [REF["zarray"][str(l)]["shape"] for l in range(5)] == [
        [585, 2304, 2304], [292, 1152, 1152], [146, 576, 576], [73, 288, 288], [36, 144, 144]]


def test_pyramid_halves_by_floor_division():
    """585 -> 292 -> 146 -> 73 -> 36: the odd trailing plane is dropped, never padded."""
    shape = [585, 2304, 2304]
    for level in range(1, 5):
        shape = [s // 2 for s in shape]
        assert shape == REF["zarray"][str(level)]["shape"], level


def test_block_average_rounds_half_to_even():
    # mean 0.5 -> 0, mean 1.5 -> 2. np.rint, not floor and not round-half-up.
    a = np.zeros((2, 2, 2), dtype=np.uint8); a[0, 0, 0] = 4
    assert zw.block_average_uint8(a)[0, 0, 0] == 0          # 4/8 = 0.5 -> 0
    b = np.zeros((2, 2, 2), dtype=np.uint8); b[0, 0, 0] = 12
    assert zw.block_average_uint8(b)[0, 0, 0] == 2          # 12/8 = 1.5 -> 2


def test_block_average_rejects_unaligned_shape():
    try:
        zw.block_average_uint8(np.zeros((3, 2, 2), dtype=np.uint8))
    except ValueError as e:
        assert "divisible" in str(e)
    else:
        raise AssertionError("expected ValueError on an unaligned shape")


def test_ome_multiscales_matches_published_form():
    got = zw.ome_multiscales([[585, 2304, 2304]] * 5,
                             REF["zattrs"]["spacing_nm_zyx"], "vol")[0]
    ref = REF["zattrs"]["multiscales"][0]
    assert got["version"] == ref["version"] == "0.4"
    assert got["axes"] == ref["axes"]
    assert [d["path"] for d in got["datasets"]] == [d["path"] for d in ref["datasets"]]
    for g, r in zip(got["datasets"], ref["datasets"]):
        assert np.allclose(g["coordinateTransformations"][0]["scale"],
                           r["coordinateTransformations"][0]["scale"])


def test_provenance_attrs_key_set_matches_published():
    attrs = zw.provenance_attrs(
        source="/x.nd2", source_channel=0, source_shape_zyx=(585, 2304, 2304),
        spacing_nm_zyx=REF["zattrs"]["spacing_nm_zyx"], clip_mode="percentile",
        clip_values=(1.0, 99.0), clip_limit=0.03)
    expected = set(REF["zattrs"]) - {"multiscales"}
    assert set(attrs) == expected, set(attrs) ^ expected
    assert attrs["downsample_factor_zyx"] == [1, 1, 1]
    assert attrs["downsample_rounding"] == "nearest_even"


def test_fixed_intensity_swaps_only_the_clip_keys():
    attrs = zw.provenance_attrs(
        source="/x.nd2", source_channel=0, source_shape_zyx=(4, 4, 4),
        spacing_nm_zyx=(1, 1, 1), clip_mode="fixed_intensity",
        clip_values=(120.0, 350.0), clip_limit=0.03)
    assert attrs["clip_mode"] == "fixed_intensity"
    assert attrs["clip_intensity_range"] == [120.0, 350.0]
    assert "clip_percentiles" not in attrs


def test_write_pyramid_reproduces_the_published_metadata():
    """End-to-end on a volume small enough to check by hand, big enough to
    exercise more than one chunk row and a full 4-level reduction."""
    rng = np.random.default_rng(0)
    shape = (37, 40, 40)                      # odd Z on purpose: 37->18->9->4->2
    vol = rng.integers(0, 256, size=shape, dtype=np.uint8)

    with tempfile.TemporaryDirectory() as td:
        dest = Path(td) / "ExPID99_32x_1_cerebellum.zarr"
        attrs = zw.provenance_attrs(
            source="/drop/ExPID99_32x_1.nd2", source_channel=0, source_shape_zyx=shape,
            spacing_nm_zyx=(12.5, 5.078125, 5.078125), clip_mode="percentile",
            clip_values=(1.0, 99.0), clip_limit=0.03)
        out = zw.write_pyramid(dest, iter(vol), shape, (12.5, 5.078125, 5.078125),
                               attrs, z_slab=16)

        assert out["shapes"] == [[37, 40, 40], [18, 20, 20], [9, 10, 10], [4, 5, 5], [2, 2, 2]]
        meta = zw.read_group_metadata(dest)
        assert meta["zgroup"] == REF["zgroup"]
        assert sorted(meta["levels"]) == ["0", "1", "2", "3", "4"]

        ref0 = REF["zarray"]["0"]
        for level, za in meta["levels"].items():
            assert za["dtype"] == ref0["dtype"] == "|u1", level
            assert za["fill_value"] == ref0["fill_value"], level
            assert za["order"] == ref0["order"], level
            assert za["filters"] == ref0["filters"], level
            assert za["dimension_separator"] == ref0["dimension_separator"], level
            assert za["compressor"] == ref0["compressor"], (level, za["compressor"])
            assert za["zarr_format"] == ref0["zarr_format"], level
        # Chunks are min(128, extent); the published volume is larger than 128
        # in every axis, so it shows the raw 128.
        assert meta["levels"]["0"]["chunks"] == [min(128, s) for s in shape]
        assert ref0["chunks"] == [128, 128, 128]

        assert set(meta["zattrs"]) == set(REF["zattrs"])

        import zarr
        grp = zarr.open_group(str(dest), mode="r")
        # Level 0 must be the input, byte for byte -- streaming must not reorder.
        assert np.array_equal(np.asarray(grp["0"][:]), vol)
        # Level 1 must be the block average of the aligned part of level 0.
        assert np.array_equal(np.asarray(grp["1"][:]),
                              zw.block_average_uint8(vol[:36, :40, :40]))


def test_write_pyramid_rejects_a_short_plane_iterator():
    with tempfile.TemporaryDirectory() as td:
        try:
            zw.write_pyramid(Path(td) / "v.zarr",
                             iter(np.zeros((3, 8, 8), dtype=np.uint8)),
                             (5, 8, 8), (1, 1, 1), {}, z_slab=4)
        except ValueError as e:
            assert "yielded 3 planes" in str(e)
        else:
            raise AssertionError("expected ValueError for a truncated ND2 read")


def test_unknown_fold_adds_two_loud_keys_and_nothing_else():
    """ExPID71 has no expansion fold anywhere, so it is published with raw
    optics spacing on an explicit decision (2026-09-21). The caveat has to live
    in the metadata: a reader who overlays one of these on an expid96/99/107/108
    layer is wrong by the fold, and a log line does not travel with the data."""
    base = dict(source="/x.nd2", source_channel=0, source_shape_zyx=(4, 4, 4),
                spacing_nm_zyx=(500.0, 162.5, 162.5), clip_mode="percentile",
                clip_values=(1.0, 99.0), clip_limit=0.03)
    normal = zw.provenance_attrs(**base)
    flagged = zw.provenance_attrs(**base, spacing_basis="optics_uncorrected")

    assert set(flagged) - set(normal) == {"spacing_basis", "spacing_warning"}
    assert flagged["spacing_basis"] == "optics_uncorrected"
    assert "UNKNOWN" in flagged["spacing_warning"]
    assert "not spatially comparable" in flagged["spacing_warning"].lower()
    # The normal path must keep the published key set EXACTLY -- the flag is
    # additive, so expid96/99/107/108 are untouched by its existence.
    assert set(normal) == set(REF["zattrs"]) - {"multiscales"}


def test_unknown_spacing_basis_is_rejected():
    try:
        zw.provenance_attrs(source="/x.nd2", source_channel=0, source_shape_zyx=(4, 4, 4),
                            spacing_nm_zyx=(1, 1, 1), clip_mode="percentile",
                            clip_values=(1.0, 99.0), clip_limit=0.03,
                            spacing_basis="whatever")
    except ValueError as e:
        assert "spacing_basis" in str(e)
    else:
        raise AssertionError("expected ValueError on an unknown spacing basis")


def test_assumed_fold_is_biological_but_says_so():
    """Donglai's 2026-09-21 revision: assume 18x for ExPID71 rather than
    publish raw optics spacing. The spacing then IS on the same basis as every
    other group and is comparable -- but 18 was chosen, not read, and the
    metadata has to carry that distinction or the assumption disappears the
    moment this conversation does."""
    base = dict(source="/x.nd2", source_channel=0, source_shape_zyx=(4, 4, 4),
                spacing_nm_zyx=(27.77777, 9.02777, 9.02777), clip_mode="percentile",
                clip_values=(1.0, 99.0), clip_limit=0.03)
    plain = zw.provenance_attrs(**base)
    assumed = zw.provenance_attrs(**base, spacing_basis="biological_assumed_fold")

    assert set(assumed) - set(plain) == {"spacing_basis", "spacing_warning"}
    assert assumed["spacing_basis"] == "biological_assumed_fold"
    assert "ASSUMED" in assumed["spacing_warning"]
    # It must NOT claim incomparability -- that was the optics_uncorrected case,
    # and conflating the two would mislead in the opposite direction.
    assert "comparable with them" in assumed["spacing_warning"]

    optics = zw.provenance_attrs(**base, spacing_basis="optics_uncorrected")
    assert "not spatially comparable" in optics["spacing_warning"].lower()
    assert assumed["spacing_warning"] != optics["spacing_warning"]


def test_ome_multiscales_needs_no_numpy():
    """`refold` runs on the host, which has no numpy by design. If this module
    grows a top-level numpy import again, a fold correction silently becomes a
    container round trip."""
    import subprocess
    import sys as _sys
    code = (
        "import sys; sys.modules['numpy'] = None;"
        "sys.path.insert(0, %r);" % str(Path(__file__).parent) +
        "import importlib, zarr_writer as z; importlib.reload(z);"
        "m = z.ome_multiscales([[1,1,1]]*3, (24.0, 18.0, 18.0), 'v')[0];"
        "assert m['datasets'][2]['coordinateTransformations'][0]['scale'] == [96.0, 72.0, 72.0];"
        "print('ok')"
    )
    out = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-500:]
