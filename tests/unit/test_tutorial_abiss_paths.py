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
from connectomics.utils.yaml_config import load_yaml_with_bases_and_params

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "tutorials" / "neuron_j0126" / "3_abiss.yaml"

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
def test_tutorial_local_outputs_stay_under_the_configured_output_root(tmp_path: Path) -> None:
    """Resolve through the real loader: the trap only shows after interpolation.

    Reading the YAML raw leaves `${params.paths.output_root}` in the string, which
    URI parsing then reads as the network location -- so a raw-loaded config looks
    broken even when it is fine. The invariant worth pinning is that every local
    output still lands under the output_root the user configured.
    """
    params = yaml.safe_load((REPO_ROOT / "tutorials" / "neuron_j0126" / "params.yaml").read_text())
    output_root = tmp_path / "out"
    params["params"]["paths"]["repository"] = str(REPO_ROOT)
    params["params"]["paths"]["dataset_root"] = str(tmp_path / "data")
    params["params"]["paths"]["output_root"] = str(output_root)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params))

    config = yaml.safe_load(TUTORIAL.read_text())
    config["_base_"] = str(tmp_path / "params.yaml")
    config_path = tmp_path / "3_abiss.yaml"
    config_path.write_text(yaml.safe_dump(config))

    param = load_yaml_with_bases_and_params(config_path)["abiss_chunk"]["param"]

    for key in LOCAL_OUTPUT_KEYS:
        raw = str(param[key])
        resolved = _resolved(raw)
        assert resolved.is_absolute(), f"{key}={raw!r} did not resolve absolutely"
        assert resolved.is_relative_to(output_root), (
            f"{key}={raw!r} resolves to {resolved}, outside the configured output_root "
            f"{output_root}. A 'file://<dir>/...' value is the usual cause -- see the "
            f"module docstring."
        )
