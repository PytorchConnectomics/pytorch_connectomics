"""Contracts for install.py wheel selection and scripts/check_install.py."""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


install = _load("pytc_install", REPO / "install.py")
check_install = _load("pytc_check_install", REPO / "scripts" / "check_install.py")


@pytest.mark.parametrize(
    "driver_cuda, expected",
    [
        ("13.2", "cu130"),  # never guess an untested newer index
        ("13.0", "cu130"),
        ("12.9", "cu128"),
        ("12.8", "cu128"),
        ("12.7", "cu126"),
        ("12.4", "cu118"),  # cu124 is frozen at torch 2.6; do not select it
        ("11.8", "cu118"),
        ("11.7", None),
        ("garbage", None),
    ],
)
def test_driver_cuda_maps_to_newest_compatible_wheel(driver_cuda, expected):
    assert install.cuda_to_pytorch(driver_cuda) == expected


def test_wheel_table_is_ordered_newest_first():
    versions = [v for v, _ in install.PYTORCH_CUDA_WHEELS]
    assert versions == sorted(versions, reverse=True)


@pytest.mark.parametrize(
    "arch_list, capability, expected",
    [
        (["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], (12, 0), True),
        (["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"], (12, 0), False),
        (["sm_80", "sm_90"], (8, 9), True),  # same-major SASS runs on a newer minor
        (["sm_100"], (12, 0), False),  # different major: not binary compatible
        (["sm_80", "compute_90"], (12, 0), True),  # PTX JIT-compiles forward
        (["compute_90"], (8, 6), False),  # PTX does not run on an older device
        (["sm_90a"], (9, 0), True),
    ],
)
def test_wheel_supports_device(arch_list, capability, expected):
    assert check_install.wheel_supports_device(arch_list, *capability) is expected


def test_libgl_import_error_points_to_headless_opencv():
    hint = check_install._hint_for_import_error(
        "cv2", "ImportError: libGL.so.1: cannot open shared object file"
    )
    assert "opencv-python-headless" in hint
