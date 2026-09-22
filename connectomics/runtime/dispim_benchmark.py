"""Separate Docker stages for paired diSPIM training, inference, and CPU scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from connectomics.config import load_config, resolve_default_profiles

from .checkpoint_dispatch import get_checkpoint_test_output_dir
from .snemi_benchmark import run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[2]


def checkpoint_record(path: Path, expected_epochs: int) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    epoch, step = int(state["epoch"]), int(state["global_step"])
    if epoch != expected_epochs - 1 or step < 1:
        raise ValueError(f"Expected {expected_epochs} completed epochs; got epoch={epoch}, step={step}")
    return {
        "checkpoint": str(path), "completed_epochs": epoch + 1, "global_step": step,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run_stage(args: argparse.Namespace) -> dict:
    config, output = args.config.resolve(), args.output.resolve()
    train_cfg = resolve_default_profiles(load_config(config), mode="train")
    if train_cfg.model.arch.type != "mednext_dual_view" or train_cfg.model.in_channels != 2:
        raise ValueError("This workflow requires a two-channel mednext_dual_view model")
    expected_epochs = int(train_cfg.optimization.max_epochs)
    if args.stage == "score":
        if args.prediction is None:
            raise ValueError("score requires --prediction from the GPU inference stage")
        score = run_benchmark(argparse.Namespace(
            config=config, output=output, prediction=args.prediction.resolve(),
            checkpoint=None, resume_to=None,
        ), crop="full")
        return {"adapted_rand_error": score}

    import torch

    required = int(train_cfg.system.num_gpus) if args.stage == "train" else 1
    if not torch.cuda.is_available() or torch.cuda.device_count() < required:
        raise RuntimeError(f"{args.stage} requires {required} visible CUDA GPUs; CPU fallback forbidden")
    if args.stage == "infer" and args.checkpoint is None:
        raise ValueError("infer requires --checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "stage": args.stage, "config": str(config),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "cuda_devices": torch.cuda.device_count(),
    }
    manifest_path = output / "manifest.json"
    try:
        # Test-stage setup resolves paths again. Anchor the Docker data root so
        # repeated resolution cannot prepend a relative root twice.
        data_root = (REPO_ROOT / train_cfg.data.root_path).resolve()
        command = [
            sys.executable, str(REPO_ROOT / "scripts/main.py"), "--config", str(config),
        ]
        if args.stage == "train":
            command += ["--mode", "train"]
            if args.checkpoint is not None:
                source = args.checkpoint.resolve()
                state = torch.load(source, map_location="cpu", weights_only=False)
                completed = int(state["epoch"]) + 1
                step = int(state["global_step"])
                if not 0 < completed < expected_epochs or step != completed * int(
                    train_cfg.optimization.n_steps_per_epoch
                ):
                    raise ValueError(f"Invalid resume checkpoint: epoch={completed - 1}, step={step}")
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                copied = output / "training" / stamp / "checkpoints" / "resume.ckpt"
                copied.parent.mkdir(parents=True)
                shutil.copy2(source, copied)
                manifest.update(
                    resumed_from_completed_epochs=completed,
                    resumed_from_global_step=step,
                    resume_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                )
                command += ["--checkpoint", str(copied), "--reset-max-epochs", str(expected_epochs)]
            command += [f"save_path={output / 'training'}"]
        else:
            manifest.update(checkpoint_record(args.checkpoint.resolve(), expected_epochs))
            copied = output / "training" / "final" / "checkpoints" / "final.ckpt"
            copied.parent.mkdir(parents=True)
            shutil.copy2(args.checkpoint, copied)
            command += [
                "--mode", "test", "--checkpoint", str(copied),
                "system.num_gpus=1", "decoding.enabled=false", "evaluation.enabled=false",
            ]
        command += [f"data.root_path={data_root}"]
        manifest["command"] = command
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with (output / f"{args.stage}.log").open("w") as log:
            subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.stage == "train":
            candidates = list((output / "training").glob("*/checkpoints/last*.ckpt"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one final checkpoint from fresh training: {candidates}")
            record = checkpoint_record(candidates[0], expected_epochs)
            expected_steps = expected_epochs * int(train_cfg.optimization.n_steps_per_epoch)
            if record["global_step"] != expected_steps:
                raise ValueError(f"Expected {expected_steps} optimizer steps: {record}")
            manifest.update(record)
            shutil.copy2(candidates[0], output / "final.ckpt")
        else:
            directory = get_checkpoint_test_output_dir(copied)
            candidates = list(directory.glob("*/raw_*.h5"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one raw affinity artifact: {candidates}")
            shutil.copy2(candidates[0], output / "affinities.h5")
            manifest["raw_prediction_sha256"] = hashlib.sha256(
                (output / "affinities.h5").read_bytes()
            ).hexdigest()
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "infer", "score"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prediction", type=Path)
    print(json.dumps(run_stage(parser.parse_args()), indent=2))
