"""Import ownership and runnable examples for the unsupervised metric package."""

import importlib
import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["arbor", "morphology", "classification"])
def test_unsupervised_modules_have_one_canonical_import_path(module):
    name = f"connectomics.metrics.unsupervised.{module}"
    assert importlib.import_module(name).__name__ == name
    assert importlib.util.find_spec(f"connectomics.metrics.{module}") is None


def test_unsupervised_package_does_not_add_facade_exports():
    import connectomics.metrics.unsupervised as unsupervised

    assert unsupervised.__all__ == []


def test_unsupervised_readme_examples():
    readme = Path(__file__).resolve().parents[2] / "connectomics/metrics/unsupervised/README.md"
    namespace = {}
    examples = readme.read_text().split("```python\n")[1:]
    assert len(examples) == 2
    for block in examples:
        exec(compile(block.split("```")[0], str(readme), "exec"), namespace)
    assert namespace["record"].label == 7
    assert namespace["decision"].automatic_class == "dendrite_like_candidate"
    assert len(namespace["retained_edges"]) == 2
