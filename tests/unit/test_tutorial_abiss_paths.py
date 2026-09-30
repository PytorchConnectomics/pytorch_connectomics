"""The ABISS tutorial config uses an explicit external installation."""

from pathlib import Path

from connectomics.config import load_config, resolve_default_profiles


def test_tutorial_abiss_has_explicit_external_installation():
    path = Path(__file__).resolve().parents[2] / "tutorials/basics/decoding_abiss.yaml"
    cfg = resolve_default_profiles(load_config(path))
    step = next(step for step in cfg.decoding.steps if step.name == "decode_abiss")
    assert step.kwargs["abiss_home"].startswith("/path/to/")
    assert "{abiss_home}" in step.kwargs["command"]
    assert "optic_nerve_affinities.h5" in str(cfg.decoding.load_prediction_path)
