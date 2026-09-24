#!/usr/bin/env python3
"""Naming and destination rules -- the parts that decide where bytes land.

These are pure functions on purpose: getting `expid99` or a `_1Byqvupl` suffix
wrong writes a volume to a path nothing reads, and that failure is silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ingest  # noqa: E402


def test_expid_group_from_stem():
    assert ingest.expid_group("ExPID99_32x_1") == "expid99"
    assert ingest.expid_group("ExPID96_2ndgel_S1_40XW001_18x") == "expid96"
    assert ingest.expid_group("expid7_foo") == "expid7"       # leading zero-free


def test_expid_group_refuses_a_stem_it_cannot_place():
    try:
        ingest.expid_group("cerebellum_18x_2")
    except ValueError as e:
        assert "ExPID" in str(e)
    else:
        raise AssertionError("expected ValueError for a stem with no ExPID prefix")


def test_published_names_match_the_bucket():
    """Read back off gs://donglai_public/liconn/moe/expid99/image/."""
    assert ingest.published_name("ExPID99_32x_1", "cerebellum") == \
        "ExPID99_32x_1_cerebellum"
    assert ingest.published_name("ExPID99_18x_2", "cerebellum",
                                 "1Byqvupl8nRyrVqPR3aN9LLXkBFfMImPD") == \
        "ExPID99_18x_2_cerebellum_1Byqvupl"
    assert ingest.published_name("ExPID99_18x_2", "cerebellum",
                                 "1m2f9z4JAJTXmP5DiJvL05n099CtJIBx1") == \
        "ExPID99_18x_2_cerebellum_1m2f9z4J"
    # No drop slug: the expid96 set carries none.
    assert ingest.published_name("ExPID96_2ndgel_S1_40XW001_18x") == \
        "ExPID96_2ndgel_S1_40XW001_18x"


def test_gcs_dest_is_the_published_path():
    assert ingest.gcs_dest("donglai_public", "liconn/moe", "ExPID99_32x_1",
                           "ExPID99_32x_1_cerebellum") == \
        "gs://donglai_public/liconn/moe/expid99/image/ExPID99_32x_1_cerebellum.zarr"


def test_two_same_named_drive_files_get_different_destinations():
    """The `Cerebellum` drop holds two distinct ExPID99_18x_2.nd2. If this ever
    collapses to one path, one volume silently overwrites the other."""
    a = ingest.published_name("ExPID99_18x_2", "cerebellum", "1Byqvupl8nRy")
    b = ingest.published_name("ExPID99_18x_2", "cerebellum", "1m2f9z4JAJTX")
    assert a != b
    assert ingest.gcs_dest("b", "p", "ExPID99_18x_2", a) != \
        ingest.gcs_dest("b", "p", "ExPID99_18x_2", b)


def test_docker_argv_is_a_preprocess_call_not_a_replay_of_argv():
    """Under `run`, the real argv reads `run ... --docker`. Replaying that
    inside the container re-enters the driver and reaches for a gcloud that is
    deliberately not installed there -- so the argv is rebuilt, not forwarded."""
    import argparse
    import tempfile

    captured = []
    real_run = ingest._run
    ingest._run = lambda cmd, **kw: captured.append([str(c) for c in cmd])
    try:
        with tempfile.TemporaryDirectory() as td:
            nd2 = Path(td) / "ExPID99_32x_1.nd2"
            nd2.write_bytes(b"")
            a = argparse.Namespace(
                nd2=str(nd2), work=str(Path(td) / "work"), name=None,
                clip_variant="clip_percentile_1_99", clip_mode="percentile",
                clip_values=[1.0, 99.0], clip_limit=0.03, channel=0, levels=4,
                fold=None, fold_unknown=False, optics_spacing_nm=None,
                drop_slug="cerebellum", force=False, image="pytc:liconn-ingest")
            ingest._docker_preprocess(a)
    finally:
        ingest._run = real_run

    cmd = captured[0]
    assert cmd[0] == "docker" and "pytc:liconn-ingest" in cmd
    tail = cmd[cmd.index("/workspace/tutorials/liconn_ingest/ingest.py") + 1:]
    assert "preprocess" in tail and "run" not in tail
    assert "--docker" not in tail, "would recurse into the container forever"
    assert tail[tail.index("--name") + 1] == "ExPID99_32x_1_cerebellum"
    assert tail[tail.index("--clip-variant") + 1] == "clip_percentile_1_99"


def test_fold_unknown_and_explicit_fold_are_mutually_exclusive_in_the_docker_argv():
    """`--fold-unknown` must reach the container, and must not be paired with a
    fold: publishing optics spacing for a volume whose fold IS known would
    silently mislabel it."""
    import argparse
    import tempfile

    captured = []
    real_run = ingest._run
    ingest._run = lambda cmd, **kw: captured.append([str(c) for c in cmd])
    try:
        with tempfile.TemporaryDirectory() as td:
            nd2 = Path(td) / "ExPID71_Hippocampus_300nm_40XW01.nd2"
            nd2.write_bytes(b"")
            a = argparse.Namespace(
                nd2=str(nd2), work=str(Path(td) / "work"), name=None,
                clip_variant="clip_percentile_1_99", clip_mode="percentile",
                clip_values=[1.0, 99.0], clip_limit=0.03, channel=0, levels=4,
                fold=None, fold_unknown=True, optics_spacing_nm=None, drop_slug=None,
                force=False, image="pytc:liconn-ingest")
            ingest._docker_preprocess(a)
    finally:
        ingest._run = real_run

    tail = captured[0]
    assert "--fold-unknown" in tail
    assert "--fold" not in tail


def _plan_args(**over):
    import argparse
    base = dict(split_exposures=False, channel=0, drop_slug=None,
                image="pytc:liconn-ingest")
    base.update(over)
    return argparse.Namespace(**base)


def test_exposure_plan_is_a_single_cube_without_the_flag():
    """No --split-exposures means the ND2 is never opened to pick a name."""
    entry = {"published_name": "ExPID108_32x_Piriform_03", "stem": "ExPID108_32x_Piriform_03",
             "drive_name": "ExPID108_32x_Piriform_03.nd2"}
    assert ingest._exposure_plan(_plan_args(), Path("/nope.nd2"), entry) == \
        [("ExPID108_32x_Piriform_03", 0, None, None)]


def test_exposure_plan_names_each_channel_by_its_nd2_exposure():
    """Channel 0 of ExPID71_120ms-30ms_* is the 30 ms acquisition: the filename
    order is the REVERSE of the channel order. Verified against the real ND2
    on 2026-09-21 (`488_Low` = 30 ms is channel 0). This is spec.md I7."""
    real = ingest._inspect_via_docker
    ingest._inspect_via_docker = lambda image, nd2: {
        "channels": {"0": {"name": "488_Low", "exposure_ms": 30.0},
                     "1": {"name": "488", "exposure_ms": 120.0}}}
    try:
        entry = {"stem": "ExPID71_120ms-30ms_600nm_40XW02",
                 "drive_name": "ExPID71_120ms-30ms_600nm_40XW02.nd2",
                 "published_name": "ExPID71_120ms-30ms_600nm_40XW02"}
        plan = ingest._exposure_plan(_plan_args(split_exposures=True), Path("/x.nd2"), entry)
    finally:
        ingest._inspect_via_docker = real
    assert plan == [
        ("ExPID71_120ms-30ms_600nm_40XW02_e030", 0, 30, "488_Low"),
        ("ExPID71_120ms-30ms_600nm_40XW02_e120", 1, 120, "488"),
    ]


def test_exposure_plan_refuses_when_filename_and_nd2_disagree():
    """The filename is trusted for WHICH exposures exist, never for which
    channel. If even the set disagrees, one of the two is wrong about the file
    and neither reading should reach the bucket."""
    real = ingest._inspect_via_docker
    ingest._inspect_via_docker = lambda image, nd2: {
        "channels": {"0": {"name": "a", "exposure_ms": 50.0},
                     "1": {"name": "b", "exposure_ms": 120.0}}}
    try:
        entry = {"stem": "ExPID71_120ms-30ms_600nm_40XW02",
                 "drive_name": "ExPID71_120ms-30ms_600nm_40XW02.nd2",
                 "published_name": "x"}
        try:
            ingest._exposure_plan(_plan_args(split_exposures=True), Path("/x.nd2"), entry)
        except SystemExit as e:
            assert "refusing to publish either reading" in str(e)
        else:
            raise AssertionError("expected SystemExit on an exposure-set mismatch")
    finally:
        ingest._inspect_via_docker = real


def test_expected_object_count_from_metadata_alone():
    """The completeness check must work on a group whose local copy and raw
    are long gone, so it counts chunks from .zattrs + .zarray only."""
    zattrs = {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}]}]}
    zarrays = {"0": {"shape": [441, 2048, 2048], "chunks": [128, 128, 128]},
               "1": {"shape": [220, 1024, 1024], "chunks": [128, 128, 128]}}
    # level 0: ceil(441/128)=4 * 16 * 16 = 1024 chunks, +.zarray +.zattrs
    # level 1: ceil(220/128)=2 *  8 *  8 =  128 chunks, +.zarray +.zattrs
    # plus the group's own .zgroup and .zattrs
    assert ingest.expected_object_count(zattrs, zarrays) == 2 + (2 + 1024) + (2 + 128)


def test_expected_count_matches_a_real_published_group():
    """ExPID71_2Hippocampus_500nm_40XW really holds 1185 objects (counted in
    the bucket 2026-09-21). If the arithmetic drifts, a complete group starts
    looking truncated and gets needlessly re-uploaded."""
    shapes = [(441, 2048, 2048), (220, 1024, 1024), (110, 512, 512),
              (55, 256, 256), (27, 128, 128)]
    zattrs = {"multiscales": [{"datasets": [{"path": str(i)} for i in range(5)]}]}
    zarrays = {str(i): {"shape": list(s),
                        "chunks": [min(128, d) for d in s]}
               for i, s in enumerate(shapes)}
    assert ingest.expected_object_count(zattrs, zarrays) == 1185


def test_a_truncated_group_is_not_counted_as_published():
    """The regression that motivated all of this: 312 of 1185 objects, with a
    valid .zattrs, was skipped as 'already published'."""
    zattrs = {"multiscales": [{"datasets": [{"path": "0"}]}]}
    zarrays = {"0": {"shape": [441, 2048, 2048], "chunks": [128, 128, 128]}}
    expected = ingest.expected_object_count(zattrs, zarrays)
    assert expected > 312, "a partial upload must not satisfy the expected count"


def test_fetch_discards_a_short_file_instead_of_reusing_it():
    """A file that exists is not a file that is complete. An interrupted
    transfer left ExPID107_14.5x_04.nd2 at 7.16 of 9.15 GB on 2026-09-21; it
    was handed to the ND2 parser and died on an invalid ChunkMap."""
    import tempfile

    calls = []
    real_run = ingest._run
    ingest._run = lambda cmd, **kw: calls.append(cmd)
    try:
        with tempfile.TemporaryDirectory() as td:
            inbox = Path(td)
            short = inbox / "ExPID107_14.5x_04.nd2"
            short.write_bytes(b"x" * 10)
            entry = {"local": short.name, "drive_id": "abc", "bytes": 9145040896}
            try:
                ingest.fetch_one(inbox, entry)
            except SystemExit:
                pass          # the faked _run never actually downloads
            assert not short.exists() or short.stat().st_size == 10, \
                "the short file must be unlinked before re-downloading"
            assert calls, "a re-download must have been attempted"
            assert "copyid" in calls[0], calls[0]
    finally:
        ingest._run = real_run


def test_fetch_keeps_a_correctly_sized_file():
    import tempfile

    calls = []
    real_run = ingest._run
    ingest._run = lambda cmd, **kw: calls.append(cmd)
    try:
        with tempfile.TemporaryDirectory() as td:
            inbox = Path(td)
            good = inbox / "v.nd2"
            good.write_bytes(b"x" * 64)
            ingest.fetch_one(inbox, {"local": "v.nd2", "drive_id": "a", "bytes": 64})
            assert not calls, "a complete file must not be re-downloaded"
    finally:
        ingest._run = real_run


def _fake_gcloud(mapping):
    """Patch subprocess.run inside ingest with canned (rc, stdout, stderr)."""
    import subprocess as sp

    class R:
        def __init__(self, rc, out, err):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def run(cmd, **kw):
        key = " ".join(cmd[:3])
        for pat, val in mapping.items():
            if pat in " ".join(cmd):
                return R(*val)
        return R(1, "", "matched no objects")
    real = ingest.subprocess.run
    ingest.subprocess.run = run
    return real


def test_auth_failure_raises_instead_of_reporting_absent():
    """An expired token made every group look unpublished, and the pipeline
    re-downloaded 7.75 GB of an already verified volume (2026-09-21). 'Cannot
    look' must never collapse into 'not there'."""
    real = _fake_gcloud({".zattrs": (1, "", "ERROR: There was a problem refreshing "
                                            "your current auth tokens: Reauthentication failed.")})
    try:
        try:
            ingest.remote_group_complete("gs://b/p/x.zarr")
        except ingest.BucketUnreachable as e:
            assert "Reauthentication" in str(e)
        else:
            raise AssertionError("an auth error must raise, not return 'incomplete'")
    finally:
        ingest.subprocess.run = real


def test_genuinely_absent_group_still_reports_incomplete():
    """The other side of the same coin: a real 404 must NOT raise, or a first
    publish could never happen."""
    real = _fake_gcloud({".zattrs": (1, "", "ERROR: One or more URLs matched no objects.")})
    try:
        ok, found, expected, why = ingest.remote_group_complete("gs://b/p/x.zarr")
        assert ok is False and found == 0 and why == "no .zattrs"
    finally:
        ingest.subprocess.run = real
