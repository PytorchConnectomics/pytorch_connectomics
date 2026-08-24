#!/usr/bin/env python3
"""Prepare restartable sharded ABISS task lists without running the chunks."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
for import_path in (REPOSITORY, REPOSITORY / "scripts"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path, required=True)
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import run_seuron_provenance as replay
    from connectomics.runtime.abiss_chunk import (
        _prepare_execution_outputs,
        _prepare_runtime_secrets_view,
        _write_param,
    )

    cli_args = replay.parse_args(
        [
            "--config",
            str(args.config),
            "--name",
            args.name,
            "--out-root",
            str(args.out_root),
            "--mode",
            args.mode,
            "--execute",
        ]
    )
    resolved = replay.resolve_replay(cli_args)
    replay._preflight_abiss(resolved)
    replay._preflight_nucleus(resolved)
    affinity = replay._preflight_affinity(resolved)
    build_id = replay._abiss_build_id(resolved.abiss_home)
    expected_manifest = replay._expected_manifest(resolved, abiss_build_id=build_id)
    replay._apply_output_mode(resolved, expected_manifest)
    replay._write_manifest(resolved, expected_manifest)

    prepared = replay.prepare_execution(resolved, affinity)
    _prepare_execution_outputs(prepared)
    _prepare_runtime_secrets_view(prepared)
    _write_param(prepared.param_path, prepared.param_payload)
    runtime_param = prepared.effective_secrets_dir / "param"
    _write_param(runtime_param, prepared.param_payload)

    environment = dict(os.environ)
    environment.update(
        {
            "WORKER_HOME": str(prepared.abiss_home),
            "SECRETS": str(prepared.effective_secrets_dir),
            "STAGE": "ws",
            "AIRFLOW_TMP_DIR": str(prepared.workdir / ".airflow_init"),
        }
    )
    environment.update(prepared.extra_env or {})
    scripts = prepared.abiss_home / "scripts"
    prepared.workdir.mkdir(parents=True, exist_ok=True)

    config_sh = prepared.effective_secrets_dir / "config.sh"
    temporary_config = config_sh.with_suffix(".sh.tmp")
    with temporary_config.open("w", encoding="utf-8") as handle:
        subprocess.run(
            [sys.executable, str(scripts / "set_env.py"), str(runtime_param)],
            stdout=handle,
            cwd=prepared.workdir,
            env=environment,
            check=True,
        )
    temporary_config.replace(config_sh)

    for script_name in ("chunk_volume.py", "generate_batches.py"):
        subprocess.run(
            [sys.executable, str(scripts / script_name), prepared.root_tag, str(runtime_param)],
            cwd=prepared.workdir,
            env=environment,
            check=True,
        )

    layers = {}
    for layer in range(prepared.top_mip + 1):
        task_file = prepared.workdir / f"{layer}.txt"
        if task_file.is_file():
            layers[layer] = sum(bool(line.strip()) for line in task_file.read_text().splitlines())
    runtime = {
        "repository": str(REPOSITORY),
        "workdir": str(prepared.workdir),
        "abiss_home": str(prepared.abiss_home),
        "secrets": str(prepared.effective_secrets_dir),
        "cloud_volume_dir": (prepared.extra_env or {}).get("CLOUD_VOLUME_DIR", ""),
        "param": str(runtime_param),
        "root_tag": prepared.root_tag,
        "top_mip": prepared.top_mip,
        "layers": layers,
        "seg_path": str(prepared.param_payload["SEG_PATH"]),
    }
    args.runtime_json.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_json.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
