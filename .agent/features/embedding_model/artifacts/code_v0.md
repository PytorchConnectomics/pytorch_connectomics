# Code v0
## Overview

Implemented the approved auxiliary 12-channel instance embedding alongside the six BANIS+ affinity channels. Audited the previous interrupted session's draft against plan_v2 and the binding NONFINITE amendment. All executable CPU unit/regression checks pass. GPU gates remain for the planner; no training or Slurm submission was launched.

## What Changed

- Added EmbeddingMeanLoss and canonical registry/metadata routing; no architecture, orchestrator, transform, inference, decoding, or public facade changes.
- Added the 18-channel tutorial, paired seed-43 sbatch, watchdog and contract tests.
- This continuation strengthened the draft's connectivity fixture to use unequal component sizes (two versus three voxels), added four CPU work-dtype/autocast cases, and added four tests of the real cancellation wrapper with mocked Slurm/clock calls. The production draft already satisfied the audited rank, fp64, denominator, checkpoint and NONFINITE contracts; no speculative production rewrite was needed.

## Implementation Details

The loss normalizes GT/mask by rank, preserves singleton spatial dimensions, rejects mismatched shapes, and requires emit_gt_seg. It casts fp16/bf16/fp32 predictions to fp32 inside disabled autocast, preserving fp64. Positive GT intersected with all routed mask channels defines eligible voxels. CC runs on that valid foreground with 26-connectivity by default; components sharing a parent are excluded from push. Object-balanced pull and mean norm combine with chunked ordered-pair push divided by N(N-1), including excluded pairs in the denominator. Non-empty samples are averaged; an all-empty batch returns graph-connected zero. Checkpoint chunks pass row bounds as arguments, avoiding late-bound closure errors during backward.

The 18-output single head uses BCE on 0:6 and embedding loss on 6:18, both weight 1.0. Embedding target_slice 0:3 routes validity only. Raw GT is copied before per-target affinity erosion=2; global erosion=0. Inference retains channels 0,1,2. Pair settings inherit defaults 1024/False pending GPU memory measurement.

Watchdog scans every available gate for NaN or either infinity before missing-data and finite-ratio decisions. NONFINITE cancels immediately and returns 8 after verified cancellation, 7 on cancel failure, or 8 with WOULD CANCEL in dry-run. Both finite ratios must exceed 1.10 for ordinary failure (3). Baseline/missing/ended/stale behavior follows the plan.

Read the local Kisuk wiki and supplementary note, cached upstream /tmp/deepem_mean.py, NISB contrastive proposal and lessons (ABISS, erosion/occupancy and thin identity), and the actual registry, metadata, MalisLoss, orchestrator dispatch, label transforms, crop/affinity code and relevant tests. Both direct web fetch attempts for the upstream GitHub/raw URLs returned Cache miss; the locally cached source was used to audit the oracle. The oracle retains upstream mathematics, with unused downsampling removed and scalar accumulator dtype changed to preserve fp64, as explicitly documented in its header. Prior proposal is still marked Proposed; this stage does not claim its Phase 0 ran.

Full contents of the four git-excluded/ignored deliverables follow.

### tutorials/neuron_nisb/base_banis+_embed12.yaml

```yaml
_base_: [base_banis+.yaml]

experiment_name: nisb_base_banis+_embed12
description: BANIS+ affinities with an auxiliary 12-channel instance mean embedding.
save_path: outputs/nisb_base_banis+_embed12

default:
  model:
    out_channels: 18
    loss:
      losses:
        - function: PerChannelBCEWithLogitsLoss
          weight: 1.0
          pred_slice: "0:6"
          target_slice: "0:6"
          kwargs:
            auto_pos_weight: true
            max_pos_weight: 10.0
        - function: EmbeddingMeanLoss
          weight: 1.0
          pred_slice: "6:18"
          # Routes short-range affinity validity (geometry + ignore, not foreground).
          # Target values are unused by the embedding loss.
          target_slice: "0:3"
          kwargs:
            alpha: 1.0
            beta: 1.0
            gamma: 0.001
            delta_v: 0.0
            delta_d: 1.5
            recompute_ext: true
            connectivity: 26
  data:
    label_transform:
      erosion: 0
      emit_gt_seg: true
      targets:
        - name: affinity
          kwargs:
            offsets: ["1-0-0", "0-1-0", "0-0-1", "10-0-0", "0-10-0", "0-0-10"]
            affinity_mode: banis
            erosion: 2

train:
  monitor:
    logging:
      scalar:
        loss:
          - train_loss_total_epoch
          - val_loss_total
          - train_loss_term_0_weighted
          - train_loss_term_1_weighted
```

### slurm_jobs/nisb_banis_plus_embed12_train.sbatch

```bash
#!/bin/bash
# Paired with seed-43 BANIS+: 20260905_212323, job 2968810, 200k steps in 2d05h.
# Same 4 GPUs x batch 2 and full 200k cosine; auxiliary embedding starts from scratch.
#SBATCH --job-name=bemb12_train
#SBATCH --account=weilab
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=g001,g002,gdw001,gdw002
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_jobs/logs/nisb_banis_plus_embed12_train_%j.out
#SBATCH --error=slurm_jobs/logs/nisb_banis_plus_embed12_train_%j.err
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=29712
python scripts/main.py \
    --config tutorials/neuron_nisb/base_banis+_embed12.yaml \
    system.seed=43 \
    experiment_name=nisb_base_banis+_embed12_seed43 \
    save_path=outputs/nisb_base_banis+_embed12_seed43
```

### dev/nisb/scripts/train_gate_watch.py

```python
#!/usr/bin/env python3
"""Watch pinned training scalars and verify cancellation on failed gates."""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path


def read_scalars(run_dir):
    """Read all scalar events, keeping the latest wall-time for duplicate steps."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged = {}
    for path in sorted(Path(run_dir).rglob("events.out.tfevents.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0})
        events.Reload()
        for tag in events.Tags()["scalars"]:
            steps = merged.setdefault(tag, {})
            for event in events.Scalars(tag):
                previous = steps.get(event.step)
                if previous is None or event.wall_time >= previous[2]:
                    steps[event.step] = (event.step, event.value, event.wall_time)
    return {tag: sorted(steps.values()) for tag, steps in merged.items()}


def job_state(job_id):
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    states = result.stdout.strip().splitlines()
    return states[0].strip() if states else None


def cancel(job_id):
    try:
        result = subprocess.run(["scancel", str(job_id)], check=False, timeout=30)
        if result.returncode != 0:
            return False
        deadline = time.monotonic() + 120
        while True:
            if job_state(job_id) not in {"RUNNING", "PENDING"}:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(5, remaining))
    except (OSError, subprocess.SubprocessError):
        return False


def _cancel_result(reason, job_id, dry_run, cancel, code):
    if dry_run:
        print(f"{reason} — WOULD CANCEL {job_id}", flush=True)
        return code
    if cancel(job_id):
        message = f"{reason} — CANCELLED {job_id}"
        print(message, flush=True)
        print(message, file=sys.stderr, flush=True)
        return code
    print(f"{reason} — CANCEL FAILED {job_id}", flush=True)
    return 7


def watch(
    run_dir,
    job_id,
    baseline_logdir,
    *,
    pair="val_loss_term_0_weighted=val_loss_total",
    train_tag="train_loss_total_step",
    steps=(4999, 9999),
    max_ratio=1.10,
    gate_grace_steps=1500,
    stale_minutes=45,
    poll_seconds=300,
    dry_run=False,
    read_scalars=read_scalars,
    job_state=job_state,
    cancel=cancel,
    now=time.time,
    sleep=time.sleep,
):
    """Return a gate/liveness exit code; all external effects are injectable."""
    run_tag, base_tag = pair.split("=", 1)
    baseline = {s: v for s, v, _ in read_scalars(baseline_logdir).get(base_tag, [])}
    if not steps or any(s not in baseline or not math.isfinite(baseline[s]) for s in steps):
        print("BAD BASELINE", flush=True)
        return 2
    while True:
        scalars = read_scalars(run_dir)
        gates = {s: v for s, v, _ in scalars.get(run_tag, [])}
        last_step = max((s for s, _, _ in scalars.get(train_tag, [])), default=-1)
        # Scan every available gate first: a later non-finite value takes priority
        # over an earlier missing gate, and can never be certified as PASS.
        for step in steps:
            if step in gates and not math.isfinite(gates[step]):
                return _cancel_result(f"GATE NONFINITE at step {step}", job_id, dry_run, cancel, 8)
        ratios = []
        for step in steps:
            if step in gates:
                value, base = gates[step], baseline[step]
                ratio = value / base if base != 0 else (1.0 if value == 0 else math.inf)
                ratios.append(ratio)
                print(f"GATE {step} run={value} base={base} ratio={ratio}", flush=True)
            elif last_step >= step + gate_grace_steps:
                print(f"GATE DATA MISSING at {step}", flush=True)
                return 6
        if len(ratios) == len(steps):
            if all(ratio > max_ratio for ratio in ratios):
                return _cancel_result("GATE FAIL", job_id, dry_run, cancel, 3)
            print("GATE PASS", flush=True)
            return 0
        state = job_state(job_id)
        if state is None or state in {
            "BOOT_FAIL",
            "CANCELLED",
            "COMPLETED",
            "DEADLINE",
            "FAILED",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "TIMEOUT",
        }:
            print(f"JOB ENDED EARLY last_step={last_step}", flush=True)
            return 4
        newest = max((wall for values in scalars.values() for _, _, wall in values), default=0)
        if state == "RUNNING" and now() - newest > stale_minutes * 60:
            print(f"STALE last_step={last_step}", flush=True)
            return 5
        sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--baseline-logdir", required=True)
    parser.add_argument("--pair", default="val_loss_term_0_weighted=val_loss_total")
    parser.add_argument("--train-tag", default="train_loss_total_step")
    parser.add_argument("--steps", default="4999,9999")
    parser.add_argument("--max-ratio", type=float, default=1.10)
    parser.add_argument("--gate-grace-steps", type=int, default=1500)
    parser.add_argument("--stale-minutes", type=float, default=45)
    parser.add_argument("--poll-seconds", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.steps = tuple(int(step) for step in args.steps.split(","))
    if "=" not in args.pair or not all(args.pair.split("=", 1)):
        parser.error("--pair must be RUN_TAG=BASE_TAG")
    return watch(**vars(args))


if __name__ == "__main__":
    sys.exit(main())
```

### dev/nisb/scripts/test_train_gate_watch.py

```python
import math
import subprocess
from types import SimpleNamespace

import pytest
import train_gate_watch as gate
from train_gate_watch import read_scalars, watch


def events(values, tag="val_loss_term_0_weighted"):
    return {tag: [(step, value, 10000) for step, value in values]}


def run_watch(run, *, state="RUNNING", cancelled=True, baseline=None, **kwargs):
    calls = []
    base = events([(4999, 1), (9999, 1)], "val_loss_total") if baseline is None else baseline

    def cancel(job):
        calls.append(job)
        return cancelled

    code = watch(
        "run",
        "123",
        "base",
        read_scalars=lambda path: base if path == "base" else run,
        job_state=lambda job: state,
        cancel=cancel,
        now=lambda: 10000,
        sleep=lambda seconds: pytest.fail("Unexpected poll"),
        **kwargs,
    )
    return code, calls


@pytest.mark.parametrize("values", [(1, 1.1), (1.2, 1)])
def test_pass_and_mixed(values, capsys):
    assert run_watch(events(list(zip((4999, 9999), values)))) == (0, [])
    assert "GATE PASS" in capsys.readouterr().out


@pytest.mark.parametrize(
    "cancelled,dry_run,expected,message,calls",
    [
        (True, False, 3, "CANCELLED", ["123"]),
        (False, False, 7, "CANCEL FAILED", ["123"]),
        (True, True, 3, "WOULD CANCEL", []),
    ],
)
def test_finite_failure(cancelled, dry_run, expected, message, calls, capsys):
    assert run_watch(events([(4999, 1.2), (9999, 1.2)]), cancelled=cancelled, dry_run=dry_run) == (
        expected,
        calls,
    )
    captured = capsys.readouterr()
    assert message in captured.out
    if message == "CANCELLED":
        assert message in captured.err


@pytest.mark.parametrize(
    "values", [(math.nan, 1), (1, math.inf), (-math.inf, 1), (math.nan, math.nan)]
)
def test_nonfinite_never_passes(values, capsys):
    assert run_watch(events(list(zip((4999, 9999), values)))) == (8, ["123"])
    captured = capsys.readouterr()
    assert "NONFINITE" in captured.out and "CANCELLED" in captured.err
    assert "PASS" not in captured.out


@pytest.mark.parametrize(
    "cancelled,dry_run,expected,calls,message",
    [
        (False, False, 7, ["123"], "CANCEL FAILED"),
        (True, True, 8, [], "WOULD CANCEL"),
        (True, False, 8, ["123"], "CANCELLED"),
    ],
)
def test_nonfinite_immediate(cancelled, dry_run, expected, calls, message, capsys):
    assert run_watch(events([(4999, math.nan)]), cancelled=cancelled, dry_run=dry_run) == (
        expected,
        calls,
    )
    assert message in capsys.readouterr().out


def test_nonfinite_precedes_missing_gate():
    data = events([(9999, math.nan)])
    data.update(events([(12000, 1)], "train_loss_total_step"))
    assert run_watch(data) == (8, ["123"])


def test_missing_gate():
    assert run_watch(events([(6499, 1)], "train_loss_total_step")) == (6, [])


def test_stale_running():
    assert run_watch({"train_loss_total_step": [(10, 1, 0)]}) == (5, [])


def test_pending_does_not_go_stale():
    clock = [10000]
    polls = [events([(10, 1)], "train_loss_total_step"), events([(4999, 1), (9999, 1)])]

    def read(path):
        if path == "base":
            return events([(4999, 1), (9999, 1)], "val_loss_total")
        return polls.pop(0)

    def sleep(seconds):
        clock[0] += seconds

    assert (
        watch(
            "run",
            "123",
            "base",
            read_scalars=read,
            job_state=lambda job: "PENDING",
            now=lambda: clock[0],
            sleep=sleep,
        )
        == 0
    )
    assert clock[0] == 10300


@pytest.mark.parametrize("state", [None, "COMPLETED", "FAILED"])
def test_early_end(state):
    assert run_watch({}, state=state) == (4, [])


@pytest.mark.parametrize("baseline", [{}, events([(4999, math.nan)], "val_loss_total")])
def test_bad_baseline(baseline):
    assert run_watch({}, baseline=baseline) == (2, [])


def test_real_tfevents(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(str(tmp_path / "logs")) as writer:
        writer.add_scalar("loss", 1.25, 4999, walltime=100)
        writer.add_scalar("loss", 1.5, 4999, walltime=101)
        writer.add_scalar("loss", 0.5, 9999, walltime=102)
    assert read_scalars(tmp_path) == {"loss": [(4999, 1.5, 101), (9999, 0.5, 102)]}


@pytest.mark.parametrize("outcome", ["ended", "timeout", "rc_failure", "query_failure"])
def test_real_cancel_verifies_job_exit(monkeypatch, outcome):
    clock = [0.0]
    commands = []
    polls = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if outcome == "rc_failure" else 0)

    def state(job):
        polls.append(job)
        if outcome == "query_failure":
            raise subprocess.CalledProcessError(1, "squeue")
        return None if outcome == "ended" and len(polls) > 1 else "RUNNING"

    def sleep(seconds):
        clock[0] += seconds

    monkeypatch.setattr(gate.subprocess, "run", run)
    monkeypatch.setattr(gate, "job_state", state)
    monkeypatch.setattr(gate.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gate.time, "sleep", sleep)
    assert gate.cancel("123") is (outcome == "ended")
    assert commands == [["scancel", "123"]]
    if outcome == "rc_failure":
        assert polls == []
    if outcome == "timeout":
        assert clock[0] == 120
```

## Files Changed
| File | Purpose |
|---|---|
| `connectomics/models/losses/embedding.py` | Batch-aware L1 mean embedding loss, valid-foreground CC, chunked/checkpointed pair push. |
| `connectomics/models/losses/build.py` | Canonical loss import and registry entry. |
| `connectomics/models/losses/metadata.py` | Route spatial validity mask and raw gt_seg. |
| `tests/unit/test_embedding_mean_loss.py` | DeepEM value/gradient oracle, shape/dtype/mask/batch/pair contracts, CUDA gates and orchestrator integration. |
| `tests/unit/test_embedding_label_pipeline.py` | Real config-built transforms preserve raw thin identity and baseline affinity targets. |
| `tutorials/neuron_nisb/base_banis+_embed12.yaml` | 18-output BANIS+ configuration with two loss terms and per-target erosion. |
| `slurm_jobs/nisb_banis_plus_embed12_train.sbatch` | Seed-43 paired four-GPU submission script; not submitted. |
| `dev/nisb/scripts/train_gate_watch.py` | Pinned tfevents gate, liveness, verified cancellation and NONFINITE exit 8. |
| `dev/nisb/scripts/test_train_gate_watch.py` | Injected watchdog cases, cancellation verification, and real tfevents reader test. |

## Git Baseline
run_start_ref: 1546f47ece4777e20bcddd1028fb1bb8908ae02d
current_head: 1546f47ece4777e20bcddd1028fb1bb8908ae02d

HEAD is unchanged. No commits, branches, stashes or staging. Git status contains only the five planned code/test files; git check-ignore confirms all four other deliverables are excluded/ignored. No coordinator prereg, research log, run-state or sentinel was written.

## Verification

Commands ran from the repository root. `ACTIVATE` below expands exactly to `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc &&`; `THREADS` expands to `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. These are command abbreviations, not shell variables used during execution. Tests report six existing SWIG deprecation warnings.

| Command | Actual result |
|---|---|
| `ACTIVATE pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py` before continuation edits | 92 passed, 1 skipped, 11.58 s. Matches coordinator-recorded pre-original-edit baseline of 92 passed, 1 skipped. |
| `ACTIVATE pytest -q tests/unit/test_embedding_mean_loss.py tests/unit/test_embedding_label_pipeline.py dev/nisb/scripts/test_train_gate_watch.py` draft audit | 73 passed, 2 skipped, 20.61 s. |
| `ACTIVATE THREADS pytest -q tests/unit/test_embedding_mean_loss.py tests/unit/test_embedding_label_pipeline.py dev/nisb/scripts/test_train_gate_watch.py > /tmp/ccc_resume_unit_final.log 2>&1` after test strengthening | 81 passed, 2 skipped, 18.97 s. |
| `ACTIVATE THREADS pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py > /tmp/ccc_resume_regression_final.log 2>&1` | 92 passed, 1 skipped, 11.84 s. No new failures. |
| `ACTIVATE python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_nisb/base_banis+_embed12.yaml' > /tmp/ccc_resume_config.log 2>&1` | Infrastructure failure, exit 130: OpenBLAS pthread_create failed, then KeyboardInterrupt during SciPy import. |
| `ACTIVATE THREADS python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_nisb/base_banis+_embed12.yaml' > /tmp/ccc_resume_config_retry.log 2>&1` | Exit 0: 21 canonical configs validated, 3 custom workflow YAMLs skipped. CLI includes its default tutorials/*.yaml glob as well as the requested file. |
| `ACTIVATE python dev/nisb/scripts/train_gate_watch.py --dry-run --run-dir outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323 --baseline-logdir outputs/nisb_base_banis_v3_erosion2_seed43/20260905_212323/logs --pair val_loss_total=val_loss_total --job-id 0 > /tmp/ccc_resume_watch.log 2>&1` | Exit 0, GATE PASS. 4999: 0.5034302473068237 / same = 1.0; 9999: 0.4521530270576477 / same = 1.0. No Slurm call needed. |
| `ACTIVATE black --workers 1 --check tests/unit/test_embedding_mean_loss.py` | Exit 0, one file unchanged. |
| `ACTIVATE black --workers 1 --check dev/nisb/scripts/test_train_gate_watch.py` | Exit 0, one file unchanged. Both Black checks warn about configured Python 3.12 target under Python 3.11. No formatting mutation. |
| `git diff --check` | Exit 0, no whitespace errors. |
| `bash -n slurm_jobs/nisb_banis_plus_embed12_train.sbatch` | Exit 0. Syntax check only. |
| `git status --short`, `git rev-parse HEAD`, `git diff`, `git check-ignore <four excluded files>` | Expected scoped files, unchanged baseline and all four exclusions confirmed. |

CUDA tests test_cuda_mixed_precision and test_cuda_memory_stress were skipped because CUDA is unavailable on this login node. No GPU AMP, memory, trunk-gradient or matched training smoke claim is made. Regression's existing CUDA MALIS test also skipped.

### Resolved config

The probe helper dumped `OmegaConf.to_yaml(OmegaConf.structured(cfg))` to `/tmp/ccc_embed_resolved.yaml` after default/train profile resolution and data-path resolution. Assertions passed for out_channels=18, global erosion=0, emit_gt_seg=true and target erosion=2. The two resolved loss terms are BCE weight 1.0, pred/target 0:6, auto_pos_weight=true/max_pos_weight=10; embedding weight 1.0, pred 6:18/target 0:3, alpha=beta=1, gamma=.001, delta_v=0, delta_d=1.5, recompute_ext=true/connectivity=26. Resolved lazy_zarr=true, batch_size=2, patch_size=128 cubed, max_steps=200000, precision=16-mixed and deep_supervision=false.

### Dataloader probe

Bounded to 20 batches, num_workers=0, CPU torch threads=4, seed=43, real lazy-zarr datamodule. Loss measurement is forward-only with random 12-channel predictions under no_grad; CC measurements include valid-label preparation and unique-ID counting. No model or optimizer is instantiated.

Attempts:

1. `ACTIVATE PYTHONPATH=. timeout 1200 python /tmp/ccc_embed_probe.py > /tmp/ccc_resume_probe.log 2>&1`: exit 1, helper omitted resolve_data_paths and failed with Training data not found: seed*/data.zarr/img. Fixed the helper to call the same config path resolver as runtime.
2. `ACTIVATE THREADS PYTHONPATH=. timeout 1200 python /tmp/ccc_embed_probe.py > /tmp/ccc_resume_probe_retry.log 2>&1`: stalled during datamodule setup, manually interrupted (exit 130), no batches.
3. `ACTIVATE THREADS PYTHONPATH=. timeout -k 5 180 python -u /tmp/ccc_embed_probe.py > /tmp/ccc_resume_probe_diagnostic.log 2>&1`: added faulthandler dumps at 60-second intervals. Timed out at 180 seconds (exit 124). Both dumps show the main thread waiting in zarr/core/sync.py:152, called from zarr.open in LazyZarrVolumeDataset._open_array:170 while opening the first image at dataset initialization line 86. The Zarr event-loop thread is in selectors.select. This identifies the observed stall; its lower-level cause is not established.

**Actual probe numbers:** 0/20 batches obtained; setup did not finish within the 180-second diagnostic bound. Real-data max N, per-sample GT/CC counts, CC/loss timings, mask fraction, and batch shapes/dtypes are **unavailable**, not zero and not inferred from the synthetic tests. The planner must rerun the supplied helper outside this stalled environment before selecting the stress-test N or certifying GPU memory/speed. Resolved paths point to /projects/weilab/dataset/nisb/base/train/seed*/data.zarr/{img,seg}; glob resolution succeeded.

```text
/home/weidf/.config/matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /tmp/matplotlib-jtrq_vjg because there was an issue with the default path (/home/weidf/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
CONFIG {"out_channels": 18, "losses": [{"function": "PerChannelBCEWithLogitsLoss", "weight": 1.0, "pred_slice": "0:6", "target_slice": "0:6", "kwargs": {"auto_pos_weight": true, "max_pos_weight": 10.0}}, {"function": "EmbeddingMeanLoss", "weight": 1.0, "pred_slice": "6:18", "target_slice": "0:3", "kwargs": {"alpha": 1.0, "beta": 1.0, "gamma": 0.001, "delta_v": 0.0, "delta_d": 1.5, "recompute_ext": true, "connectivity": 26}}], "label_transform": {"normalize": true, "erosion": 0, "emit_gt_seg": true, "skeleton_distance": "SkeletonDistanceConfig(enabled=False, resolution=[1.0, 1.0, 1.0], alpha=0.8, smooth=True, bg_value=-1.0)", "edge_mode": "EdgeModeConfig(mode='all', thickness=1, processing_mode='2d')", "keys": ["label"], "stack_outputs": true, "retain_original": false, "output_dtype": "float32", "output_key_format": "{key}_{task}", "allow_missing_keys": false, "relabel_connected_components": false, "relabel_connectivity": 6, "segment_id": null, "boundary_thickness": 1, "resolution": null, "cache_dir": "", "targets": [{"name": "affinity", "kwargs": {"offsets": ["1-0-0", "0-1-0", "0-0-1", "10-0-0", "0-10-0", "0-0-10"], "affinity_mode": "banis", "erosion": 2}}]}, "lazy_zarr": true}
Timeout (0:01:00)!
Thread 0x00001554349c5640 (most recent call first):
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/selectors.py", line 468 in select
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/asyncio/base_events.py", line 1898 in _run_once
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/asyncio/base_events.py", line 608 in run_forever
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 982 in run
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 1045 in _bootstrap_inner
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 1002 in _bootstrap

Thread 0x000015555541e400 (most recent call first):
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 327 in wait
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 629 in wait
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/concurrent/futures/_base.py", line 305 in wait
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/site-packages/zarr/core/sync.py", line 152 in sync
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/site-packages/zarr/api/synchronous.py", line 216 in open
  File "/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/data/datasets/dataset_volume_zarr_lazy.py", line 170 in _open_array
  File "/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/data/datasets/dataset_volume_zarr_lazy.py", line 86 in __init__
  File "/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/training/lightning/data_factory.py", line 1098 in create_datamodule
  File "/tmp/ccc_embed_probe.py", line 35 in <module>
Timeout (0:01:00)!
Thread 0x00001554349c5640 (most recent call first):
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/selectors.py", line 468 in select
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/asyncio/base_events.py", line 1898 in _run_once
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/asyncio/base_events.py", line 608 in run_forever
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 982 in run
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 1045 in _bootstrap_inner
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 1002 in _bootstrap

Thread 0x000015555541e400 (most recent call first):
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 327 in wait
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/threading.py", line 629 in wait
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/concurrent/futures/_base.py", line 305 in wait
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/site-packages/zarr/core/sync.py", line 152 in sync
  File "/projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/site-packages/zarr/api/synchronous.py", line 216 in open
  File "/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/data/datasets/dataset_volume_zarr_lazy.py", line 170 in _open_array
  File "/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/data/datasets/dataset_volume_zarr_lazy.py", line 86 in __init__
  File "/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/training/lightning/data_factory.py", line 1098 in create_datamodule
  File "/tmp/ccc_embed_probe.py", line 35 in <module>

```

Probe helper (temporary verification code, not a repository change):

```python
import itertools
import faulthandler
import json
import time

import cc3d
import numpy as np
import torch
from monai.utils import set_determinism
from omegaconf import OmegaConf

from connectomics.config import load_config, resolve_default_profiles
from connectomics.config.pipeline import resolve_data_paths
from connectomics.models.losses.embedding import EmbeddingMeanLoss
from connectomics.training.lightning.data_factory import create_datamodule

torch.set_num_threads(4)
faulthandler.dump_traceback_later(60, repeat=True)
set_determinism(seed=43)
cfg = resolve_default_profiles(load_config("tutorials/neuron_nisb/base_banis+_embed12.yaml"))
cfg = resolve_data_paths(cfg)
with open("/tmp/ccc_embed_resolved.yaml", "w") as f:
    f.write(OmegaConf.to_yaml(OmegaConf.structured(cfg)))
print("CONFIG", json.dumps({"out_channels": cfg.model.out_channels,
    "losses": cfg.model.loss.losses, "label_transform": vars(cfg.data.label_transform),
    "lazy_zarr": cfg.data.dataloader.use_lazy_zarr}, default=str), flush=True)
assert cfg.model.out_channels == 18
assert cfg.data.label_transform.erosion == 0
assert cfg.data.label_transform.emit_gt_seg
assert cfg.data.label_transform.targets[0]["kwargs"]["erosion"] == 2
cfg.system.num_workers = 0
cfg.system.seed = 43
cfg.data.dataloader.pin_memory = False
start = time.perf_counter()
dm = create_datamodule(cfg, mode="train")
loader = dm.train_dataloader()
print("DATASET", type(loader.dataset).__name__, "SETUP_SECONDS", time.perf_counter()-start, flush=True)
assert type(loader.dataset).__name__ == "LazyZarrVolumeDataset"
loss_fn = EmbeddingMeanLoss()
max_n = 0
for j, batch in enumerate(itertools.islice(loader, 20)):
    ids = torch.as_tensor(batch["gt_seg"])
    if ids.ndim == 5:
        ids = ids[:, 0]
    valid = batch["label_mask"][:, :3].bool().all(1) & (ids > 0)
    cc_start = time.perf_counter()
    samples = []
    for b in range(len(ids)):
        a = ids[b].numpy().round().astype(np.int64)
        foreground = np.where(valid[b].numpy(), a, 0).astype(np.uint64)
        components, n = cc3d.connected_components(foreground, connectivity=26, return_N=True)
        samples.append({"gt_ids": int(len(np.unique(a[a>0]))),
                        "valid_gt_ids": int(len(np.unique(foreground[foreground>0]))), "N": n})
        max_n = max(max_n, n)
    cc_time = time.perf_counter() - cc_start
    pred = torch.randn(len(ids), 12, *ids.shape[1:]) * .05
    loss_start = time.perf_counter()
    with torch.no_grad():
        loss = loss_fn(pred, None, mask=batch["label_mask"][:, :3], gt_seg=ids)
    record = {"batch": j, "shapes": {k: list(batch[k].shape) for k in ("label", "label_mask", "gt_seg")},
              "dtypes": {k: str(batch[k].dtype) for k in ("label", "label_mask", "gt_seg")},
              "mask_fraction": batch["label_mask"][:, :3].float().mean().item(),
              "samples": samples, "cc_seconds": cc_time,
              "loss_seconds": time.perf_counter()-loss_start, "loss": loss.item()}
    print("BATCH", json.dumps(record), flush=True)
print("SUMMARY", json.dumps({"batches": j+1, "max_N": max_n,
                              "total_seconds": time.perf_counter()-start}), flush=True)

```

## Review Focus

- Confirm oracle values/gradients, ordered-pair denominator and unequal-component pull balance.
- Run the skipped CUDA tests and record both pair_checkpoint memory peaks; set EMBEDDING_PROBE_MAX_N from a successful real-data probe before the memory stress run.
- Complete the planner's fixed-batch trunk-gradient rule, matched baseline/embed12 smoke, AMP scaler and speed gates before approval/launch.
- Confirm the pinned-run watchdog amendment and full excluded-file contents above.

## Risks and Unknowns

- No production deviation from plan_v2 plus the binding amendment. The oracle's fp64 scalar accumulators are a documented test adaptation required by the planned fp64 comparison.
- GPU safety/performance is unverified; pair_checkpoint=False is provisional until its measured peak is below 1.5 GB. Weight remains 1.0 pending the one-time planner rule.
- Dataloader probe completion and any unavailable real-data statistics are recorded above; synthetic unit fixtures do not substitute for real-data max N or throughput.
- General inherited risks remain: trunk interference, host CC synchronization, loss of identity supervision between disconnected same-parent components, and total-loss checkpoint ranking.
- No training, prereg or research-log work was performed; these belong to later stages.

## Changes Since Previous Code Version
Initial implementation.
