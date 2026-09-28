#!/usr/bin/env python3
"""Check that a PyTorch Connectomics environment is usable.

Run inside the target env:

    python scripts/check_install.py              # human-readable report
    python scripts/check_install.py --json       # machine-readable (agents, CI)
    python scripts/check_install.py --no-gpu-run # skip the tiny CUDA kernel launch

Exit code is 0 when every required check passes and 1 otherwise. Warnings
(optional extras, ``pip check`` noise, running from ``base``) never fail the
check. Each failure carries a ``hint`` that names the fix in INSTALLATION.md.

Only the standard library is imported at module level so the script can report
a broken env instead of crashing on the first missing import.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import subprocess
import sys

# Imports every working install needs. `connectomics` is last so a failure in a
# dependency is reported against that dependency, not the package.
CORE_MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "pytorch_lightning",
    "monai",
    "omegaconf",
    "h5py",
    "zarr",
    "cv2",
    "cc3d",
    "fastremap",
    "kimimaro",
    "mahotas",
    "skimage",
    "connectomics",
]
OPTIONAL_MODULES = [
    ("tifffile", "tifffile [full]"),
    ("wandb", "wandb [full]"),
    ("optuna", "optuna [full]"),
    ("neuroglancer", "neuroglancer [full]"),
    ("pytest", "pytest [dev]"),
]


def _hint_for_import_error(name: str, message: str) -> str:
    if "libGL.so" in message or "libgthread" in message:
        return (
            "opencv-python needs system GUI libraries. Replace it with the headless build: "
            "pip uninstall -y opencv-python && pip install --force-reinstall opencv-python-headless"
        )
    if name == "cc3d" or "numpy.dtype size changed" in message or "ABI" in message:
        return "numpy ABI mismatch. See INSTALLATION.md 'ABI mismatch on import cc3d'."
    if name == "connectomics":
        return "Package not installed in this env. From the repo root: pip install -e ."
    # Import names differ from pip names (cv2, skimage), so point at the
    # declared dependency set rather than guessing a distribution.
    return "Missing dependency. From the repo root: pip install -e .  (or rerun install.py)"


def check_imports() -> list[dict]:
    results = []
    for module in CORE_MODULES:
        entry = {"check": f"import {module}", "required": True}
        try:
            mod = importlib.import_module(module)
            entry.update(ok=True, detail=getattr(mod, "__version__", "installed"))
        except Exception as exc:  # noqa: BLE001 - report any import-time failure
            message = f"{type(exc).__name__}: {exc}"
            entry.update(ok=False, detail=message, hint=_hint_for_import_error(module, message))
        results.append(entry)
    for module, label in OPTIONAL_MODULES:
        try:
            importlib.import_module(module)
            results.append(
                {"check": f"import {module}", "required": False, "ok": True, "detail": "installed"}
            )
        except Exception:  # noqa: BLE001
            results.append(
                {
                    "check": f"import {module}",
                    "required": False,
                    "ok": False,
                    "detail": f"not installed ({label})",
                }
            )
    return results


def wheel_supports_device(arch_list: list[str], major: int, minor: int) -> bool:
    """Return True if a torch build with ``arch_list`` can run on ``sm_<major><minor>``.

    Compiled SASS runs on the same major at an equal or higher minor
    (``sm_86`` runs ``sm_80`` code, ``sm_120`` does not run ``sm_100`` code).
    Embedded PTX (``compute_XY``) JIT-compiles forward to any newer device.
    """
    for arch in arch_list:
        m = re.fullmatch(r"(sm|compute)_(\d+?)(\d)[a-z]?", arch)
        if not m:
            continue
        kind, a_major, a_minor = m.group(1), int(m.group(2)), int(m.group(3))
        if kind == "sm" and a_major == major and a_minor <= minor:
            return True
        if kind == "compute" and (a_major, a_minor) <= (major, minor):
            return True
    return False


def check_torch(run_kernel: bool) -> list[dict]:
    try:
        import torch
    except Exception:  # noqa: BLE001 - already reported by check_imports
        return []

    results = [
        {
            "check": "torch build",
            "required": True,
            "ok": True,
            "detail": f"torch {torch.__version__}, CUDA runtime {torch.version.cuda or 'none'}",
        }
    ]
    nvidia_gpu = _nvidia_gpu_present()
    if not torch.cuda.is_available():
        mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if mps:
            results.append(
                {
                    "check": "accelerator",
                    "required": False,
                    "ok": True,
                    "detail": "Apple MPS available",
                }
            )
        elif nvidia_gpu:
            results.append(
                {
                    "check": "accelerator",
                    "required": True,
                    "ok": False,
                    "detail": "nvidia-smi sees a GPU but torch.cuda.is_available() is False",
                    "hint": (
                        "CPU-only torch, or a CUDA wheel newer than the driver supports. "
                        "Rerun install.py (it maps driver CUDA to a wheel) or see "
                        "INSTALLATION.md 'CUDA not available'."
                    ),
                }
            )
        else:
            results.append(
                {
                    "check": "accelerator",
                    "required": False,
                    "ok": True,
                    "detail": "CPU only (no NVIDIA GPU visible)",
                }
            )
        return results

    arch_list = torch.cuda.get_arch_list()
    for idx in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(idx)
        name = torch.cuda.get_device_name(idx)
        sm = f"sm_{major}{minor}"
        supported = wheel_supports_device(arch_list, major, minor)
        entry = {
            "check": f"cuda:{idx} kernels",
            "required": True,
            "ok": supported,
            "detail": f"{name} ({sm}); wheel ships {' '.join(arch_list)}",
        }
        if not supported:
            entry["hint"] = (
                f"This torch wheel has no kernels for {sm}. Install a newer CUDA "
                "wheel (Blackwell sm_100/sm_120 needs cu128 or newer). "
                "See INSTALLATION.md 'no kernel image is available'."
            )
        results.append(entry)

    if run_kernel:
        entry = {"check": "cuda:0 kernel launch", "required": True}
        try:
            x = torch.randn(2, 1, 8, 16, 16, device="cuda")
            y = torch.nn.Conv3d(1, 4, 3).cuda()(x).sum()
            torch.cuda.synchronize()
            entry.update(ok=bool(torch.isfinite(y)), detail="Conv3d forward OK")
        except Exception as exc:  # noqa: BLE001
            entry.update(
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                hint="See INSTALLATION.md 'no kernel image is available'.",
            )
        results.append(entry)
    return results


def _nvidia_gpu_present() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        return out.returncode == 0 and "GPU" in out.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def check_environment() -> list[dict]:
    results = [
        {
            "check": "python",
            "required": True,
            "ok": (3, 8) <= sys.version_info[:2] < (3, 13),
            "detail": f"{platform.python_version()} at {sys.executable}",
        }
    ]
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env == "base":
        results.append(
            {
                "check": "conda env",
                "required": False,
                "ok": False,
                "detail": "running in conda 'base'; use a dedicated env (e.g. pytc)",
            }
        )
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    results.append(
        {
            "check": "pip check",
            "required": False,
            "ok": pip.returncode == 0,
            "detail": (pip.stdout.strip() or pip.stderr.strip())[:500],
        }
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument(
        "--no-gpu-run", action="store_true", help="do not launch a CUDA kernel (inspection only)"
    )
    args = parser.parse_args()

    results = check_environment() + check_imports() + check_torch(not args.no_gpu_run)
    failed = [r for r in results if r["required"] and not r["ok"]]

    if args.json:
        print(json.dumps({"ok": not failed, "results": results}, indent=2))
    else:
        for r in results:
            tag = "OK  " if r["ok"] else ("FAIL" if r["required"] else "warn")
            print(f"[{tag}] {r['check']}: {r['detail']}")
            if not r["ok"] and r.get("hint"):
                print(f"       -> {r['hint']}")
        print()
        print(
            "PASS: environment is ready."
            if not failed
            else f"FAIL: {len(failed)} required check(s) failed."
        )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
