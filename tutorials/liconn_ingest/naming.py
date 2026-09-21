#!/usr/bin/env python3
"""Filename grammar and spacing arithmetic for LICONN microscope drops.

Pure functions, no I/O -- this is the part that is easy to get silently wrong,
so it is separated out and unit tested (`test_naming.py`).

THE TRAP THIS FILE EXISTS FOR. A moe ND2 filename looks like

    ExPID71_120ms-30ms_600nm_40XW02.nd2

and Moe states the plane was illuminated **30 ms first, then 120 ms**. Filename
order is therefore NOT acquisition order. Nothing here may return an acquisition
order: `parse_stem` reports the exposures as an unordered set and the caller must
resolve order from ND2 per-frame metadata. See spec.md I7 in
research/projects/msi_liconn_deploy/.

SPACING. Physical (biological) spacing = optics spacing / linear expansion fold.
The optics spacing is recorded in the ND2; the fold is NOT -- it comes from the
filename or the drop folder. That asymmetry is why `spacing_source` is carried
through to provenance rather than assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# `_18x`, `_28xx` (a real typo in the ExPID96 set), `_14p5x` for fractional folds.
_FOLD = re.compile(r"_(\d+(?:p\d+)?)xx?(?:_|$|\.)")
_EXPOSURE = re.compile(r"_(\d+)ms-(\d+)ms_")
_ZSTEP = re.compile(r"_(\d+)nm_")
_ACQUIRER = re.compile(r"\s+[A-Z][a-z]+\s+[A-Z][a-z]+$")


@dataclass(frozen=True)
class Parsed:
    stem: str                       # join key across raw/preprocessed/published
    fold: float | None = None
    z_step_nm: float | None = None
    exposures_ms: frozenset[int] = field(default_factory=frozenset)
    derived: dict[str, str] = field(default_factory=dict)


def strip_acquirer(name: str) -> str:
    """`ExPID96_..._32x Mojtaba Tavakoli` -> `ExPID96_..._32x`.

    ND2 filenames carry the acquirer; every derived artifact drops it. The stem
    is the join key, so this must happen exactly once, here.
    """
    return _ACQUIRER.sub("", name).strip()


def parse_stem(filename: str) -> Parsed:
    """Parse a drop filename. Every field is a CANDIDATE for a human to confirm."""
    stem = strip_acquirer(filename.rsplit(".", 1)[0] if "." in filename else filename)
    derived: dict[str, str] = {}

    fold = None
    if m := _FOLD.search(stem + "_"):
        fold = float(m.group(1).replace("p", "."))
        derived["fold"] = "filename"

    z_step = None
    if m := _ZSTEP.search(stem + "_"):
        z_step = float(m.group(1))
        derived["z_step_nm"] = "filename"

    exposures: frozenset[int] = frozenset()
    if m := _EXPOSURE.search(stem + "_"):
        # UNORDERED on purpose. See module docstring / spec.md I7.
        exposures = frozenset({int(m.group(1)), int(m.group(2))})
        derived["exposures_ms"] = "filename"

    return Parsed(stem=stem, fold=fold, z_step_nm=z_step,
                  exposures_ms=exposures, derived=derived)


def physical_spacing_nm(optics_zyx_nm, fold: float) -> tuple[float, float, float]:
    """Biological spacing = optics spacing / linear expansion fold."""
    if fold is None or fold <= 0:
        raise ValueError(f"expansion fold must be positive, got {fold!r}")
    z, y, x = (float(v) for v in optics_zyx_nm)
    return (z / fold, y / fold, x / fold)


def anisotropy(optics_zyx_nm) -> float:
    """Z:XY ratio. Invariant to expansion fold -- only the z-step moves it.

    Reference points: ExPID82 mip0 = 1.33; ExPID96/99 at a 400 nm step = 2.46;
    a 600 nm step = 3.69. Matching ExPID82 at 162.5 nm XY needs ~215 nm.
    """
    z, _y, x = (float(v) for v in optics_zyx_nm)
    return z / x


def cube_id(stem: str, exposure_ms: int | None = None) -> str:
    """cube_id IS the stem. Suffix the exposure only when one file yields two.

    Do not invent a scheme: `volumes.py::V.layer_name()` and every published
    artifact already key on the stem.
    """
    return stem if exposure_ms is None else f"{stem}_e{int(exposure_ms):03d}"
