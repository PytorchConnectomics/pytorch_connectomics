"""The shipped ABISS tutorial must resolve its outputs under the repo, not `/`.

`file://outputs/x` looks relative but is not: URI parsing reads `outputs` as the
network location, and `_cloudpath_to_local_path` keeps only the path component, so
the value silently resolves to `/x` at the filesystem root. The tutorial shipped
that form, and nothing caught it because the E2E runs used a different config.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from connectomics.runtime import abiss_chunk

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "tutorials" / "neuron_j0126" / "abiss.yaml"

# Every param key whose value names a local output location.
LOCAL_OUTPUT_KEYS = ("WS_PATH", "SEG_PATH", "SCRATCH_PATH", "CHUNKMAP_OUTPUT")


def _resolved(value: str) -> Path:
    return abiss_chunk._cloudpath_to_local_path(abiss_chunk._normalize_cloudpath(value))


def test_malformed_relative_file_uri_escapes_to_filesystem_root() -> None:
    """Pin the trap itself, so nobody 'fixes' a path back to this form."""
    assert _resolved("file://outputs/neuron_j0126_abiss/scratch") == Path(
        "/neuron_j0126_abiss/scratch"
    )
    # A bare relative path and a proper absolute URI both behave.
    assert _resolved("outputs/x") == (Path.cwd() / "outputs" / "x")
    assert _resolved("file:///srv/outputs/x") == Path("/srv/outputs/x")


@pytest.mark.skipif(not TUTORIAL.exists(), reason="tutorial not present")
def test_tutorial_local_outputs_stay_under_the_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    param = yaml.safe_load(TUTORIAL.read_text())["abiss_chunk"]["param"]

    for key in LOCAL_OUTPUT_KEYS:
        raw = param[key]
        resolved = _resolved(raw)
        assert resolved.is_absolute(), f"{key}={raw!r} did not resolve absolutely"
        assert resolved.is_relative_to(REPO_ROOT), (
            f"{key}={raw!r} resolves to {resolved}, outside the repo. A "
            f"'file://<dir>/...' value is the usual cause -- see the module docstring."
        )
