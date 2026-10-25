"""Optional numba failures must not affect default augmentation."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("broken", [False, True])
def test_optional_numba_import(broken):
    setup = """
import sys
sys.modules["numba"] = None
"""
    if broken:
        setup = """
import sys
import importlib.abc
import importlib.util
class Broken(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numba":
            return importlib.util.spec_from_loader(fullname, self)
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise RuntimeError("broken numba")
sys.meta_path.insert(0, Broken())
"""
    script = (
        setup
        + """
from connectomics.config.schema import AugmentationConfig
from connectomics.data.augmentation.build import _build_augmentations
from monai.transforms import RandAffined, Resample
cfg = AugmentationConfig()
cfg.affine.enabled = True
items = _build_augmentations(cfg, ["image"])
affine = next(t for t in items if isinstance(t, RandAffined))
assert type(affine.rand_affine.resampler) is Resample
cfg.affine.fast_resample = True
try:
    _build_augmentations(cfg, ["image"])
except ImportError as exc:
    assert str(exc) == "data.augmentation.affine.fast_resample requires a working numba installation"
else:
    raise AssertionError("expected ImportError")
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
