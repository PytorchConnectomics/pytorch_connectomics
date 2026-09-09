"""Prepare the explicitly documented 3 -> 20 epoch continuation checkpoint."""
import hashlib
import json
from pathlib import Path

import torch

from connectomics.training.optimization.lr_scheduler import WarmupCosineLR

root = Path(__file__).resolve().parent
source = root / "original.ckpt"
target = root / "resume20.ckpt"
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
assert checkpoint["epoch"] == 2 and checkpoint["global_step"] == 600
state = checkpoint["lr_schedulers"][0]
assert state["max_iters"] == 3 and state["warmup_iters"] == 1 and state["last_epoch"] == 3
old_lr = sorted(set(state["_last_lr"]))
state["max_iters"] = 20
# Use the canonical scheduler's formula, without resetting its counters.
scheduler = object.__new__(WarmupCosineLR)
scheduler.__dict__.update(state)
lrs = scheduler.get_lr()
state["_last_lr"] = lrs
groups = checkpoint["optimizer_states"][0]["param_groups"]
assert len(groups) == len(lrs)
for group, lr in zip(groups, lrs):
    group["lr"] = lr
torch.save(checkpoint, target)
provenance = {
    "source_gcs": "gs://donglai/em/snemi/cloud-runs/snemi_4l4_20260903b/training/20260903_151723/checkpoints/last.ckpt",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "derived_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "epoch": checkpoint["epoch"],
    "global_step": checkpoint["global_step"],
    "old_max_iters": 3,
    "new_max_iters": 20,
    "warmup_iters_unchanged": 1,
    "last_epoch_unchanged": 3,
    "old_lr": old_lr,
    "new_lr": sorted(set(lrs)),
    "changes": ["lr_schedulers[0].max_iters", "lr_schedulers[0]._last_lr", "optimizer_states[0].param_groups[*].lr"],
    "preserved": ["model weights", "optimizer moments", "optimizer steps", "epoch", "global_step", "scheduler counters", "Lightning loops", "callbacks"],
}
(root / "resume-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print(json.dumps(provenance, indent=2))
