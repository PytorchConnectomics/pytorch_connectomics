"""Unit tests for the LICONN drop filename grammar and spacing arithmetic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import naming  # noqa: E402


def test_sweep_filename():
    p = naming.parse_stem("ExPID71_120ms-30ms_600nm_40XW02.nd2")
    assert p.stem == "ExPID71_120ms-30ms_600nm_40XW02"
    assert p.z_step_nm == 600.0
    assert p.exposures_ms == frozenset({30, 120})
    # Fold is NOT in this filename; it comes from the drop folder.
    assert p.fold is None
    assert p.derived["exposures_ms"] == "filename"


def test_exposures_are_unordered():
    """I7: filename order is not acquisition order, so no order may be implied."""
    a = naming.parse_stem("ExPID71_120ms-30ms_600nm_40XW02.nd2")
    b = naming.parse_stem("ExPID71_30ms-120ms_600nm_40XW02.nd2")
    assert a.exposures_ms == b.exposures_ms
    assert not isinstance(a.exposures_ms, (list, tuple))


def test_acquirer_stripped_once():
    raw = "ExPID96_2ndgel_S4_40XW002_32x Mojtaba Tavakoli"
    assert naming.strip_acquirer(raw) == "ExPID96_2ndgel_S4_40XW002_32x"
    assert naming.strip_acquirer(naming.strip_acquirer(raw)) == "ExPID96_2ndgel_S4_40XW002_32x"


def test_folds_including_typo_and_fractional():
    assert naming.parse_stem("ExPID96_2ndgel_S1_40XW001_18x.nd2").fold == 18.0
    # `_28xx` is a real typo in the ExPID96 set and must still parse.
    assert naming.parse_stem("ExPID96_2ndgel_S3_40XW004_28xx.nd2").fold == 28.0
    # 14.5x arrived 2026-09-21, outside the canonical {18,22,28,32}.
    assert naming.parse_stem("ExPID_S1_40XW_14p5x.nd2").fold == 14.5


def test_anisotropy_is_invariant_to_fold():
    """The whole point: expansion cannot change Z:XY -- only the z-step can."""
    optics = (600.0, 162.5, 162.5)
    base = naming.anisotropy(optics)
    for fold in (14.5, 18, 22, 28, 32):
        phys = naming.physical_spacing_nm(optics, fold)
        assert abs(naming.anisotropy(phys) - base) < 1e-9


def test_anisotropy_reference_points():
    assert abs(naming.anisotropy((600.0, 162.5, 162.5)) - 3.692) < 1e-3   # this drop
    assert abs(naming.anisotropy((400.0, 162.5, 162.5)) - 2.462) < 1e-3   # data in hand
    assert abs(naming.anisotropy((12.0, 9.0, 9.0)) - 1.333) < 1e-3        # ExPID82 mip0
    # ~215 nm is the z-step that would match ExPID82 at 162.5 nm XY optics.
    assert abs(naming.anisotropy((216.67, 162.5, 162.5)) - 1.333) < 1e-3


def test_physical_spacing_matches_recorded_expid96():
    """lessons/liconn_expansion_factor.md: 0.1625 um / 32 = 5.078125 nm XY."""
    z, y, x = naming.physical_spacing_nm((400.0, 162.5, 162.5), 32)
    assert abs(x - 5.078125) < 1e-9 and abs(y - 5.078125) < 1e-9
    assert abs(z - 12.5) < 1e-9


def test_cube_id_is_the_stem():
    assert naming.cube_id("ExPID96_2ndgel_S1_40XW001_18x") == "ExPID96_2ndgel_S1_40XW001_18x"
    assert naming.cube_id("ExPID71_x", 30) == "ExPID71_x_e030"


def test_fold_must_be_positive():
    for bad in (0, -1, None):
        try:
            naming.physical_spacing_nm((400.0, 162.5, 162.5), bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for fold={bad!r}")
