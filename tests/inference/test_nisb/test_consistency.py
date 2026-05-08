"""Consistency test: whole-volume vs chunked inference on a NISB crop.

Runs `scripts/main.py --mode test` once per strategy on the same checkpoint
and the same 1024x1024x200 crop, then compares the saved raw predictions.
The two strategies should produce identical short-range affinity volumes
up to numerical noise. A divergence beyond `MAX_ABS_TOL` is the bug we
are debugging.

Manual usage::

    NISB_TEST_CKPT=path/to/checkpoint.ckpt \\
    pytest tests/inference/test_nisb/test_consistency.py -s

    # or, for ad-hoc debugging without pytest, run this file as a script:
    python tests/inference/test_nisb/test_consistency.py \\
        --ckpt path/to/checkpoint.ckpt

Skips when:
    - the crop volume is missing under `dev/nisb/data/seed0/`
    - the env var `NISB_TEST_CKPT` is unset (pytest path)

`pytest_addoption` cannot live in a subdirectory conftest, so the
pytest entry point reads the checkpoint path only from the env var.
The script entry point (`python …test_consistency.py`) still accepts
`--ckpt` because argparse runs in-process there.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
CROP_DIR = PROJECT_ROOT / "dev" / "nisb" / "data" / "seed0"
CROP_IMG = CROP_DIR / "img_crop_988-988-575_x1024_y1024_z200.h5"

WHOLE_CFG = HERE / "whole.yaml"
CHUNKED_CFG = HERE / "chunked.yaml"

# Affinity values are in [0, 1] after sigmoid; tighten this once we
# confirm the two paths agree.
MAX_ABS_TOL = 1e-4
MAX_FRAC_OUTLIERS = 1e-4  # fraction of voxels allowed above tolerance


def _resolve_ckpt(cli_value: str | None) -> Path | None:
    if cli_value:
        return Path(cli_value).resolve()
    env = os.environ.get("NISB_TEST_CKPT")
    if env:
        return Path(env).resolve()
    return None


EXPERIMENT_NAMES = {
    "whole.yaml": "test_nisb_consistency_whole",
    "chunked.yaml": "test_nisb_consistency_chunked",
}


def _run_inference(cfg: Path, ckpt: Path) -> Path:
    """Invoke `scripts/main.py --mode test` and return the saved prediction h5."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "main.py"),
        "--config", str(cfg),
        "--mode", "test",
        "--checkpoint", str(ckpt),
    ]
    print(f"\n[test_nisb_consistency] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    exp_name = EXPERIMENT_NAMES[cfg.name]
    results_dir = PROJECT_ROOT / "outputs" / exp_name / "results"
    candidates = sorted(results_dir.glob("*_prediction.h5"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_prediction.h5 in {results_dir} after running {cfg.name}."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_prediction(path: Path) -> np.ndarray:
    """Load the (C, Z, Y, X) prediction array from the h5 file."""
    with h5py.File(path, "r") as f:
        # The runtime saves under a single dataset; pick the largest array.
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets in {path}")
        # In practice the saver writes to "main"; fall back to the largest.
        if "main" in keys:
            return f["main"][...]
        biggest = max(keys, key=lambda k: f[k].size)
        return f[biggest][...]


def compare(whole: np.ndarray, chunked: np.ndarray) -> dict:
    """Return diff statistics between two predictions of identical shape."""
    assert whole.shape == chunked.shape, (
        f"shape mismatch: whole={whole.shape}, chunked={chunked.shape}"
    )
    diff = np.abs(whole.astype(np.float32) - chunked.astype(np.float32))
    return {
        "shape": whole.shape,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "frac_above_tol": float((diff > MAX_ABS_TOL).mean()),
    }


@pytest.fixture(scope="module")
def ckpt() -> Path:
    path = _resolve_ckpt(None)
    if path is None:
        pytest.skip("set NISB_TEST_CKPT=path/to/checkpoint.ckpt to enable this test")
    if not path.exists():
        pytest.skip(f"checkpoint not found: {path}")
    if not CROP_IMG.exists():
        pytest.skip(f"crop volume not found: {CROP_IMG}")
    return path


def test_whole_vs_chunked_predictions_match(ckpt):
    whole_h5 = _run_inference(WHOLE_CFG, ckpt)
    chunked_h5 = _run_inference(CHUNKED_CFG, ckpt)

    whole = _load_prediction(whole_h5)
    chunked = _load_prediction(chunked_h5)

    stats = compare(whole, chunked)
    print(f"\n[test_nisb_consistency] {stats}", flush=True)

    assert stats["max_abs"] <= MAX_ABS_TOL or stats["frac_above_tol"] <= MAX_FRAC_OUTLIERS, (
        f"whole vs chunked predictions differ beyond tolerance: {stats}\n"
        f"  whole:  {whole_h5}\n"
        f"  chunked:{chunked_h5}"
    )


# Ad-hoc CLI entry — `python tests/inference/test_nisb/test_consistency.py --ckpt X`.
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=False, default=None)
    p.add_argument(
        "--whole-h5", default=None,
        help="Skip running inference; just compare two existing prediction h5 files.",
    )
    p.add_argument("--chunked-h5", default=None)
    args = p.parse_args()

    if args.whole_h5 and args.chunked_h5:
        whole = _load_prediction(Path(args.whole_h5))
        chunked = _load_prediction(Path(args.chunked_h5))
    else:
        ckpt_path = _resolve_ckpt(args.ckpt)
        if ckpt_path is None or not ckpt_path.exists():
            sys.exit("ERROR: provide --ckpt path/to/checkpoint.ckpt or NISB_TEST_CKPT env var")
        whole = _load_prediction(_run_inference(WHOLE_CFG, ckpt_path))
        chunked = _load_prediction(_run_inference(CHUNKED_CFG, ckpt_path))

    stats = compare(whole, chunked)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["max_abs"] > MAX_ABS_TOL and stats["frac_above_tol"] > MAX_FRAC_OUTLIERS:
        sys.exit("FAIL: predictions differ beyond tolerance")
    print("OK: predictions match within tolerance")
