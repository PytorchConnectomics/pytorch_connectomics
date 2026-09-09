"""Run the SNEMI Docker tutorial through ABISS to a challenge-crop metric.

Training and raw inference use the canonical CLI. Decoding calls the standalone
stage; the existing SNEMI evaluator owns the crop and metric implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from connectomics.config import load_config, resolve_default_profiles, save_config
from connectomics.config.pipeline.config_io import resolve_data_paths
from connectomics.data.io import read_hdf5, write_hdf5
from connectomics.decoding.stage import run_decoding_stage

from .checkpoint_dispatch import get_checkpoint_test_output_dir
from .preflight import validate_runtime_coherence

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], log: Path, manifest: dict) -> None:
    manifest["commands"].append(command)
    (log.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Running {log.stem}; log: {log}", flush=True)
    with log.open("w") as handle:
        subprocess.run(command, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _single_file(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {len(paths)}: {paths}")
    return paths[0]


def _read_score(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 1 or rows[0]["status"] != "ok":
        raise ValueError("SNEMI evaluation must produce exactly one successful score.")
    row = rows[0]
    value = float(row["adapted_rand_error"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid adapted Rand error: {value}")
    return {
        "metric": "adapted_rand_error",
        "value": value,
        "lower_is_better": True,
        "evaluation_support": "snemi3d_gc_challenge_crop",
        "crop": row["crop"],
        "crop_shape": row["crop_shape"],
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
    }


def run_benchmark(args: argparse.Namespace) -> float:
    config_path = args.config.resolve()
    output = args.output.resolve()
    if args.resume_to is not None and (args.checkpoint is None or args.resume_to < 1):
        raise ValueError("--resume-to requires --checkpoint and a positive total epoch count.")
    cfg = resolve_data_paths(resolve_default_profiles(load_config(config_path), mode="test"))
    validate_runtime_coherence(cfg)
    steps = cfg.decoding.steps
    if cfg.decoding.graph is not None or len(steps) != 1 or steps[0].name != "decode_abiss":
        raise ValueError("This benchmark requires exactly one decode_abiss step.")
    if not cfg.decoding.enabled or not cfg.evaluation.enabled:
        raise ValueError("The benchmark requires decoding and evaluation to be enabled.")
    if cfg.evaluation.metrics != ["adapted_rand"]:
        raise ValueError("The benchmark requires evaluation.metrics: [adapted_rand].")
    ws_binary = Path(steps[0].kwargs["cli_args"]["ws_binary"])
    if not ws_binary.is_file() or not os.access(ws_binary, os.X_OK):
        raise FileNotFoundError(
            f"ABISS ws binary is unavailable: {ws_binary}. Build the ABISS image."
        )
    label = cfg.data.test.label
    if not isinstance(label, str) or not Path(label).is_file():
        raise ValueError("test.data.test.label must resolve to one existing HDF5 label volume.")
    label_path = Path(label).resolve()
    # Check shape/dtype before training; scoring always reads the original labels.
    with h5py.File(label_path, "r") as handle:
        gt = handle["main"]
        if gt.ndim != 3 or not np.issubdtype(gt.dtype, np.integer):
            raise ValueError("Ground truth must be a 3D integer HDF5 main dataset.")
        expected_shape = gt.shape
    source = args.prediction or args.checkpoint
    if source is not None and not source.is_file():
        raise FileNotFoundError(source)
    if args.prediction is None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Training/inference requires CUDA. Use --prediction for CPU decoding."
            )
    # mkdir is exclusive: a retry cannot accidentally reuse stale affinities or metrics.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_source": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "prediction_source": str(args.prediction.resolve()) if args.prediction else None,
        "resume_to_total_epochs": args.resume_to,
        "abiss_ws_sha256": _sha256(ws_binary),
        "commands": [],
    }
    manifest_path = output / "manifest.json"
    main_command = [
        sys.executable, str(REPO_ROOT / "scripts/main.py"), "--config", str(config_path)
    ]
    try:
        save_config(cfg, output / "resolved_test.yaml")
        manifest["resolved_test_sha256"] = _sha256(output / "resolved_test.yaml")
        manifest["ground_truth"] = str(label_path)
        manifest["ground_truth_sha256"] = _sha256(label_path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        checkpoint = None
        if args.checkpoint is not None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            checkpoint = output / "training" / stamp / "checkpoints" / "last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            shutil.copy2(args.checkpoint, checkpoint)
            manifest["checkpoint_source_sha256"] = _sha256(checkpoint)
        if args.prediction is None and (checkpoint is None or args.resume_to is not None):
            command = main_command + ["--mode", "train", f"save_path={output / 'training'}"]
            if checkpoint is not None:
                command += [
                    "--checkpoint", str(checkpoint), "--reset-max-epochs", str(args.resume_to)
                ]
            _run(command, output / "train.log", manifest)
            checkpoint = _single_file(
                list((output / "training").glob("*/checkpoints/last.ckpt")), "last checkpoint"
            )
        if args.prediction is not None:
            raw_path = args.prediction.resolve()
        else:
            assert checkpoint is not None
            manifest["checkpoint_sha256"] = _sha256(checkpoint)
            _run(
                main_command
                + [
                    "--mode",
                    "test",
                    "--checkpoint",
                    str(checkpoint),
                    "decoding.enabled=false",
                    "evaluation.enabled=false",
                ],
                output / "infer.log",
                manifest,
            )
            prediction_dir = get_checkpoint_test_output_dir(checkpoint)
            assert prediction_dir is not None
            raw_path = _single_file(list(prediction_dir.glob("*/raw_*.h5")), "raw prediction")
        manifest["raw_prediction"] = str(raw_path)
        manifest["raw_prediction_sha256"] = _sha256(raw_path)
        predictions = read_hdf5(str(raw_path), dataset="main")
        if predictions.shape != (3, *expected_shape):
            raise ValueError(
                f"Expected XYZ affinities in CZYX shape {(3, *expected_shape)}, "
                f"got {predictions.shape}. No implicit crop or transpose is applied."
            )
        if not np.isfinite(predictions).all() or predictions.min() < 0 or predictions.max() > 1:
            raise ValueError("Raw affinities must be finite probabilities in [0, 1].")
        print("Running ABISS decoding on CPU", flush=True)
        result = run_decoding_stage(cfg, predictions)
        segmentation = np.asarray(result.postprocessed)
        if segmentation.shape != expected_shape or not np.issubdtype(
            segmentation.dtype, np.integer
        ):
            raise ValueError("ABISS must return a ZYX integer segmentation matching ground truth.")
        segmentation_path = output / "segmentation.h5"
        write_hdf5(str(segmentation_path), segmentation, dataset="main")
        del predictions, result, segmentation
        manifest["segmentation_sha256"] = _sha256(segmentation_path)
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/evaluate_snemi3d.py"),
                str(segmentation_path),
                "--ground-truth",
                str(label_path),
                "--crop",
                "challenge",
                "--fail-on-skip",
                "--output",
                str(output / "metrics.tsv"),
            ],
            output / "evaluate.log",
            manifest,
        )
        score = _read_score(output / "metrics.tsv")
        (output / "metrics.json").write_text(json.dumps(score, indent=2) + "\n")
        (output / "metric.txt").write_text(f"{score['value']:.12g}\n")
        manifest["status"] = "complete"
        print(f"adapted_rand_error (SNEMI challenge crop) = {score['value']:.12g}", flush=True)
        return float(score["value"])
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent run directory")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--checkpoint", type=Path, help="Evaluate a saved checkpoint; copy into run"
    )
    source.add_argument("--prediction", type=Path, help="Decode saved XYZ/CZYX affinities on CPU")
    parser.add_argument("--resume-to", type=int, help="With --checkpoint, resume to N total epochs")
    args = parser.parse_args()
    # Local relative dataset paths have the same meaning as in Docker /workspace.
    args.config = args.config.resolve()
    args.output = args.output.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    if args.prediction is not None:
        args.prediction = args.prediction.resolve()
    os.chdir(REPO_ROOT)
    run_benchmark(args)


if __name__ == "__main__":
    main()
