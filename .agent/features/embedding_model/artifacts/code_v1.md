# Code v1
## Overview

Addressed all requested code changes from review_v0. The embedding weight is now 0.3, differentiable voxel gathers use index_select, the CUDA comparison allows 1e-5 rounding tolerance, and watchdog state-query failures follow the retry/exit contract. No training or Slurm commands were launched. GPU certification remains with review_v1.

## What Changed

- Applied the already-selected weight without re-applying the gradient rule.
- Replaced only the two differentiable advanced-index gathers; preserved loss math, CC semantics, index_add_ sums, chunking, and checkpoint behavior.
- Added non-contiguous prediction value and gradient parity for both recompute_ext modes.
- Caught OSError and subprocess.SubprocessError in watch(), retrying unknown state and resetting the count on a successful query. Three consecutive failures print JOB STATE UNAVAILABLE last_step=... and return 4 without cancellation. Gate evaluation retains its priority, including the binding NONFINITE amendment.
- Added nine injected scheduler-error tests and a resolved-config weight assertion. Existing orchestrator test weights are synthetic routing inputs, not tutorial-config assertions, and remain unchanged.

## Implementation Details

The valid spatial mask is flattened in logical Z/Y/X order. Its nonzero indices select columns of pred[b].reshape(D, -1), then transpose to voxel-by-channel rows. reshape handles non-contiguous views, including those requiring a copy; autograd preserves the path back to the source storage. Means use index_select(0, inv). Advanced indexing remaining on detached labels has no prediction autograd path.

The watchdog catches query errors in watch() rather than changing job_state() or cancel(). Thus failed verification queries still make cancel() return False. Unknown queries do not masquerade as an empty successful squeue response; an actual None response still means the job ended. Tests cover CalledProcessError, TimeoutExpired, and OSError with one failure, three consecutive failures, and two pairs of failures separated by successful queries. Every new case asserts no cancellation.

Full current contents of embedding.py and all three ignored/excluded files changed in this stage follow. The submission script from code_v0 is unchanged.

### connectomics/models/losses/embedding.py

```python
"""Dense instance embeddings with the DeepEM L1 mean loss."""

from __future__ import annotations

import cc3d
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class EmbeddingMeanLoss(nn.Module):
    """Object-balanced pull, inter-object push, and mean-norm regularization.

    Based on ZettaAI/DeepEM ``feature/sr-zero-pad/deepem/loss/mean.py``.
    Unlike DeepEM, average non-empty batch samples, always intersect objects
    with the validity mask, ignore non-positive IDs, and compute connected
    components inside the loss (26-connectivity by default). Components with
    the same parent ID are excluded from push, but still count in its ordered
    pair denominator. There is no background-as-object mode.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        delta_v: float = 0.0,
        delta_d: float = 1.5,
        recompute_ext: bool = True,
        connectivity: int = 26,
        pair_chunk_size: int = 1024,
        pair_checkpoint: bool = False,
    ):
        super().__init__()
        if connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18, or 26")
        if not delta_d > 0 or not delta_v >= 0:
            raise ValueError("delta_d must be positive and delta_v non-negative")
        if not isinstance(pair_chunk_size, int) or pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be an integer >= 1")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.recompute_ext = recompute_ext
        self.connectivity = connectivity
        self.pair_chunk_size = pair_chunk_size
        self.pair_checkpoint = pair_checkpoint

    def forward(self, pred, target, mask=None, gt_seg=None):
        """Consume ``[B,D,Z,Y,X]`` embeddings and raw GT; target is unused."""
        if pred.ndim != 5:
            raise ValueError("pred must have shape [B,D,Z,Y,X]")
        if gt_seg is None:
            raise ValueError("EmbeddingMeanLoss requires data.label_transform.emit_gt_seg: true")
        ids = torch.as_tensor(gt_seg, device=pred.device).detach()
        if ids.ndim == 5 and ids.shape[1] == 1:
            ids = ids[:, 0]
        expected = (pred.shape[0], *pred.shape[2:])
        if ids.ndim != 4 or tuple(ids.shape) != expected:
            raise ValueError("gt_seg must be [B,1,Z,Y,X] or [B,Z,Y,X] matching pred")
        if mask is not None:
            mask = torch.as_tensor(mask, device=pred.device).detach().bool()
            if mask.ndim == 5:
                mask = mask.all(dim=1)
            if mask.ndim != 4 or tuple(mask.shape) != expected:
                raise ValueError("mask must be [B,C,Z,Y,X] or [B,Z,Y,X] matching pred")
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.to(torch.float64 if pred.dtype == torch.float64 else torch.float32)
            ids = (ids.round() if ids.is_floating_point() else ids).long()
            losses = []
            for b in range(pred.shape[0]):
                valid = ids[b] > 0
                if mask is not None:
                    valid = valid & mask[b]
                if not valid.any():
                    continue
                labels = ids[b][valid]
                if self.recompute_ext:
                    volume = torch.where(valid, ids[b], 0).cpu().numpy().astype(np.uint64)
                    components = cc3d.connected_components(volume, connectivity=self.connectivity)
                    labels = torch.as_tensor(components.astype(np.int64), device=pred.device)[valid]
                obj, inv = torch.unique(labels, return_inverse=True)
                n = len(obj)
                idx = valid.flatten().nonzero().squeeze(1)
                e = pred[b].reshape(pred.shape[1], -1).index_select(1, idx).T
                counts = torch.bincount(inv, minlength=n).to(pred.dtype)
                means = e.new_zeros(n, e.shape[1]).index_add_(0, inv, e) / counts[:, None]
                pull = (
                    (e - means.index_select(0, inv))
                    .abs()
                    .sum(1)
                    .sub(self.delta_v)
                    .clamp_min(0)
                    .square()
                )
                loss_int = (e.new_zeros(n).index_add_(0, inv, pull) / counts).mean()
                parents = obj
                if self.recompute_ext:
                    # Every voxel in a component has the same parent; repeated writes agree.
                    parents = torch.zeros_like(obj).scatter_(0, inv, ids[b][valid])
                loss_ext = means.sum() * 0
                if n > 1:
                    for start in range(0, n, self.pair_chunk_size):
                        end = min(start + self.pair_chunk_size, n)
                        if self.pair_checkpoint:
                            chunk = checkpoint(
                                self._pair_sum, means, parents, start, end, use_reentrant=False
                            )
                        else:
                            chunk = self._pair_sum(means, parents, start, end)
                        loss_ext = loss_ext + chunk
                    loss_ext = loss_ext / (n * (n - 1))
                loss_nrm = means.abs().sum(1).mean()
                losses.append(self.alpha * loss_int + self.beta * loss_ext + self.gamma * loss_nrm)
            return torch.stack(losses).mean() if losses else (pred * 0).sum()

    def _pair_sum(self, means, parents, start, end):
        distance = (means[start:end, None, :] - means[None, :, :]).abs().sum(-1)
        # Unique object IDs also exclude the diagonal when recompute_ext=False.
        keep = parents[start:end, None] != parents[None, :]
        return ((2 * self.delta_d - distance).clamp_min(0).square() * keep).sum()
```

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
          weight: 0.3
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
    state_failures = 0
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
        try:
            state = job_state(job_id)
        except (OSError, subprocess.SubprocessError):
            state_failures += 1
            if state_failures >= 3:
                print(f"JOB STATE UNAVAILABLE last_step={last_step}", flush=True)
                return 4
            sleep(poll_seconds)
            continue
        state_failures = 0
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


@pytest.mark.parametrize("failures", [1, 3, 4])
@pytest.mark.parametrize(
    "error", [subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError]
)
def test_job_state_failures(failures, error, capsys):
    # Four failures split by a successful query must not exhaust the retry limit.
    states = iter([False, False, True, False, False, True] if failures == 4 else [False] * failures)
    queries = []
    sleeps = []
    cancels = []

    def read(path):
        if path == "base":
            return events([(4999, 1), (9999, 1)], "val_loss_total")
        if len(sleeps) == (6 if failures == 4 else 1) and failures != 3:
            return events([(4999, 1), (9999, 1)])
        return events([(10, 1)], "train_loss_total_step")

    def state(job):
        queries.append(job)
        if not next(states):
            raise error(1, "squeue")
        return "RUNNING"

    code = watch(
        "run",
        "123",
        "base",
        read_scalars=read,
        job_state=state,
        cancel=cancels.append,
        now=lambda: 10000,
        sleep=sleeps.append,
    )
    assert code == (4 if failures == 3 else 0)
    assert cancels == []
    assert len(queries) == (6 if failures == 4 else failures)
    assert sleeps == [300] * (2 if failures == 3 else 6 if failures == 4 else 1)
    output = capsys.readouterr().out
    if failures == 3:
        assert "JOB STATE UNAVAILABLE last_step=10" in output
    else:
        assert "GATE PASS" in output
        assert "JOB STATE UNAVAILABLE" not in output


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
| `connectomics/models/losses/embedding.py` | Flat index_select voxel gathers and mean lookup; index_add_ reductions retained. |
| `tests/unit/test_embedding_mean_loss.py` | Two non-contiguous CPU value/gradient cases and CUDA comparison tolerance correction. |
| `tests/unit/test_embedding_label_pipeline.py` | Assert resolved embedding loss weight 0.3 in the existing real-config test. |
| `tutorials/neuron_nisb/base_banis+_embed12.yaml` | Set the pre-registered auxiliary loss weight to 0.3. |
| `dev/nisb/scripts/train_gate_watch.py` | Retry failed state queries; report-only exit 4 after three consecutive failures. |
| `dev/nisb/scripts/test_train_gate_watch.py` | Nine scheduler-error cases covering recovery, exhaustion, reset, and no cancellation. |

These are the six files revised since code_v0. The artifact itself is `.agent/features/embedding_model/artifacts/code_v1.md.tmp`. Existing code_v0 changes in build.py and metadata.py, and the submission script, were preserved. No unrelated files were edited.

## Git Baseline
run_start_ref: 1546f47ece4777e20bcddd1028fb1bb8908ae02d
current_head: 1546f47ece4777e20bcddd1028fb1bb8908ae02d

HEAD is unchanged. No commits, branches, stashes, staging, or HEAD changes. The pre-existing tracked modifications remain build.py and metadata.py; embedding.py and the two embedding test files remain untracked. The three revised tutorial/watchdog files are ignored/excluded and therefore reproduced in full above. No run state, review artifact, or .done sentinel was written.

## Verification

Commands ran at the repository root. Actual commands and results:

| Command | Result |
|---|---|
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/unit/test_embedding_mean_loss.py tests/unit/test_embedding_label_pipeline.py dev/nisb/scripts/test_train_gate_watch.py > /tmp/ccc_code_v1_unit.log 2>&1` | 92 passed, 2 skipped, 6 warnings in 264.18s (0:04:24) |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q tests/unit/test_loss_orchestrator.py tests/unit/test_loss_functions.py tests/unit/test_malis_loss.py tests/unit/test_banis_reproduction_transforms.py > /tmp/ccc_code_v1_regression.log 2>&1` | 92 passed, 1 skipped, 6 warnings in 211.35s (0:03:31); matches the 92 passed / 1 skipped baseline. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_nisb/base_banis+_embed12.yaml' > /tmp/ccc_code_v1_config.log 2>&1` | Exit 0; full output below. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /tmp/ccc_code_v1_bench.py > /tmp/ccc_code_v1_bench.log 2>&1` | Exit 0; timing method, script, and full output below. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && black --workers 1 connectomics/models/losses/embedding.py` | Exit 0; formatted the changed expression only. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && black --workers 1 dev/nisb/scripts/test_train_gate_watch.py` | Exit 0; formatted new parametrization only. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && black --workers 1 --check tests/unit/test_embedding_mean_loss.py` | Initial exit 1: would reformat changed assert. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && black --workers 1 tests/unit/test_embedding_mean_loss.py` | Exit 0; wrapped changed CUDA assertion. |
| `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && black --workers 1 --check tests/unit/test_embedding_mean_loss.py` | Final exit 0; unchanged. |
| `git diff --check` | Exit 0. |
| `git diff --cached`, `git status --short`, `git rev-parse HEAD`, `git check-ignore tutorials/neuron_nisb/base_banis+_embed12.yaml dev/nisb/scripts/train_gate_watch.py dev/nisb/scripts/test_train_gate_watch.py` | Empty staged diff; expected existing code_v0 worktree; unchanged HEAD; three exclusions confirmed. |

New/changed CPU tests are included in the full targeted command. The two CUDA loss tests skipped on this node, including the changed mixed-precision assertion. The regression CUDA MALIS test also skipped. No CUDA pass claim is made. Black emitted the existing Python 3.12 target / Python 3.11 runtime warning; each invocation used exactly one file and one worker. The config validator emitted a Matplotlib cache-location warning.

### Startup-delay diagnostic

While the original checks were still silent, launched this bounded diagnostic:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout -k 5 120 python -u -c 'import faulthandler; faulthandler.dump_traceback_later(30, repeat=True); import pytest; raise SystemExit(pytest.main(["-q", "tests/unit/test_embedding_mean_loss.py", "tests/unit/test_embedding_label_pipeline.py", "dev/nisb/scripts/test_train_gate_watch.py"]))' > /tmp/ccc_code_v1_diagnostic.log 2>&1
```

Actual result: exit 124 at the 120-second timeout; no completed tests in this diagnostic. Stack dumps show importlib filesystem path lookup during pytest Zarr-plugin loading; the session header appeared before timeout, but collection had not completed. This does not establish the underlying cause of the slow imports. The original full targeted suite and regression both completed successfully with the counts above; no additional test pass is claimed from the diagnostic.

### CPU timing

Same seeded random fp32 input (2 x 12 x 128 x 128 x 128) for both implementations, with 40 unequal contiguous block components per sample, float32 labels, and 98.4375% foreground. This is a synthetic approximation of component count and volume size, not a real-data probe. CPU torch threads=1; CC recomputation enabled, pair_chunk_size=1024, pair_checkpoint=False. Timed region includes complete forward plus backward, including host CC; excludes input construction, gradient clearing, and explicit garbage collection. One warmup per version, five measured runs each, alternating order. Before imports the exact pre-edit snapshot from /tmp/ccc_code_v1_before/embedding.py; after imports the current file. Other validation processes were launched concurrently on the shared login node, so the small CPU difference should not be treated as a reliable throughput claim.

Median before: **2.724932 s**; median after: **2.656169 s** (2.52% lower). Scalar loss was identical in all runs; the parity suite validates gradients. **GPU timing is unmeasured in code_v1**. Planner target remains <=0.10 s loss forward+backward on L40S and <=1.15x matched baseline step time.

```text
{"shape": [2, 12, 128, 128, 128], "threads": 1, "components": [40, 40], "foreground_fraction": 0.984375}
{"version": "before", "iteration": 0, "seconds": 3.0241355327889323, "loss": 104.67192077636719}
{"version": "after", "iteration": 0, "seconds": 2.8381503908894956, "loss": 104.67192077636719}
{"version": "after", "iteration": 1, "seconds": 2.736290675122291, "loss": 104.67192077636719}
{"version": "before", "iteration": 1, "seconds": 2.826024789363146, "loss": 104.67192077636719}
{"version": "before", "iteration": 2, "seconds": 2.7249322016723454, "loss": 104.67192077636719}
{"version": "after", "iteration": 2, "seconds": 2.6561692939139903, "loss": 104.67192077636719}
{"version": "after", "iteration": 3, "seconds": 2.7147626630030572, "loss": 104.67192077636719}
{"version": "before", "iteration": 3, "seconds": 2.7484983182512224, "loss": 104.67192077636719}
{"version": "before", "iteration": 4, "seconds": 2.5859924340620637, "loss": 104.67192077636719}
{"version": "after", "iteration": 4, "seconds": 2.5009226421825588, "loss": 104.67192077636719}
{"version": "after", "iteration": 5, "seconds": 2.46811485523358, "loss": 104.67192077636719}
{"version": "before", "iteration": 5, "seconds": 2.6752792592160404, "loss": 104.67192077636719}
{"version": "before", "median_seconds": 2.7249322016723454, "measurements": [2.826024789363146, 2.7249322016723454, 2.7484983182512224, 2.5859924340620637, 2.6752792592160404]}
{"version": "after", "median_seconds": 2.6561692939139903, "measurements": [2.736290675122291, 2.6561692939139903, 2.7147626630030572, 2.5009226421825588, 2.46811485523358]}
```

Benchmark helper (temporary verification code):

```python
import gc
import importlib.util
import json
import statistics
import time

import cc3d
import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EmbeddingMeanLoss()


torch.set_num_threads(1)
torch.manual_seed(43)
pred = torch.randn(2, 12, 128, 128, 128, requires_grad=True)
# Forty unequal rectangular instances per sample, with a background boundary.
axis = torch.arange(128)
labels = (axis[:, None, None] * 5 // 128) * 8
labels = labels + (axis[None, :, None] * 4 // 128) * 2 + axis[None, None, :] * 2 // 128 + 1
labels[:2] = 0
ids = torch.stack((labels, labels.flip(0))).float()
print(json.dumps({"shape": list(pred.shape), "threads": torch.get_num_threads(),
                  "components": [int(cc3d.connected_components(x.numpy().astype('uint64'), return_N=True)[1]) for x in ids],
                  "foreground_fraction": (ids > 0).float().mean().item()}), flush=True)
losses = {
    "before": load('/tmp/ccc_code_v1_before/embedding.py', 'before'),
    "after": load('connectomics/models/losses/embedding.py', 'after'),
}
results = {key: [] for key in losses}
for iteration in range(6):
    for key in (list(losses) if iteration % 2 == 0 else list(reversed(losses))):
        pred.grad = None
        gc.collect()
        start = time.perf_counter()
        value = losses[key](pred, None, gt_seg=ids)
        value.backward()
        elapsed = time.perf_counter() - start
        if iteration:
            results[key].append(elapsed)
        print(json.dumps({"version": key, "iteration": iteration, "seconds": elapsed, "loss": value.item()}), flush=True)
        del value
for key, values in results.items():
    print(json.dumps({"version": key, "median_seconds": statistics.median(values), "measurements": values}), flush=True)
```

Config validation output:

```text
/home/weidf/.config/matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /tmp/matplotlib-u6_zhf7g because there was an issue with the default path (/home/weidf/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Validated 21 canonical tutorial configs successfully; skipped 3 custom workflow YAMLs.
Custom workflows:
  - tutorials/seuron_provenance_replay.yaml
  - tutorials/waterz_decoding_large.yaml
  - tutorials/waterz_decoding_large_DL288B_crop1_test.yaml
```

## Review Focus

- Re-measure optimized loss forward+backward on L40S and run the fresh matched smoke at fixed weight 0.3; do not re-apply the weight-selection rule.
- Re-gate affinity interference, seconds per step, and AMP skipped updates against every planned smoke criterion.
- Execute CUDA mixed precision and memory tests on the planner node; the local node cannot verify the changed CUDA assertion.
- Inspect non-contiguous storage gradients, counter reset, and report-only scheduler failure handling alongside existing NONFINITE cancellation tests.

## Risks and Unknowns

- CPU timing does not establish the GPU speed target. The preserved host CC round-trip remains a possible cost. No scientific smoke success is inferred from unit tests.
- Affinity interference and AMP skipped-update outcomes at 0.3 remain unknown pending review_v1. Further weight/schedule/head changes require the human decision described in review_v0 if the fresh smoke fails.
- The planner-reported max N=73 and L40S evidence belong to review_v0; no real-data or GPU re-measurement was performed here.

## Changes Since Previous Code Version

| review_v0 finding | Disposition and actual change |
|---|---|
| Major: selected weight 0.3 vs configured 1.0 | Fixed: tutorial embedding weight changed to 0.3; existing resolved-config pipeline test now asserts function and weight. BCE stays 1.0. |
| Major: speed gate fails | Fixed requested implementation: flattened valid indices plus index_select for predictions, index_select for means, existing index_add_ sums retained; two non-contiguous value/gradient cases added. CPU before/after measured above. GPU gate remains unmeasured, for planner review_v1. |
| Major: exact CUDA equality | Fixed: only the specified comparison tolerance changes to rtol=1e-5, atol=1e-5 (plus line wrapping); remaining CUDA test behavior unchanged. Skipped locally without CUDA. |
| Major: affinity interference at weight 1.0 | No separate action per review: covered by weight 0.3 revision; planner re-gates fresh smoke. |
| Minor: AMP gate at weight 1.0 | No action per review beyond selected weight; planner re-gates fresh smoke. |
| Minor: watchdog squeue exception | Fixed: errors poll as unknown, counter resets on success, third consecutive error reports JOB STATE UNAVAILABLE last_step=... and returns 4 without cancel. Nine new tests cover failure types, recovery, exhaustion, reset, and no cancel. |
| Minor: corrected gradient-probe exclusion | No action per review: corrected reviewer evidence already selects 0.3; no probe or model changes. |
| Minor: smoke scheduler t_max=500 | No action per review: matched reviewer smoke configuration already corrected; no production scheduler or submission changes. |

All requested findings accepted; none disputed. Actual revision consists only of the six source/config/test files listed above and this summary artifact.
