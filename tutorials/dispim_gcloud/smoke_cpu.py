"""Build-time check of the real dual-view architecture and CPU ABISS scorer."""
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from connectomics.config import load_config, resolve_default_profiles
from connectomics.config.hardware.gpu_utils import SUPPORTED_ACCELERATORS
from connectomics.data.io import write_hdf5
from connectomics.models import build_model
from connectomics.runtime.snemi_benchmark import run_benchmark

torch.set_num_threads(4)
config_path = Path("tutorials/dispim_gcloud/dispim_snemi_20epoch.yaml")
cfg = resolve_default_profiles(load_config(config_path), mode="train")
assert cfg.system.accelerator in SUPPORTED_ACCELERATORS
model = build_model(cfg)
model.train()
out = model(torch.randn(1, 2, 32, 32, 32))
assert out["output"].shape == (1, 9, 32, 32, 32)
loss = sum(v.square().mean() for v in out.values())
loss.backward()
assert torch.isfinite(loss)
assert model.right_stem.weight.grad is not None
assert model.trunk.stem.weight.grad is not None
print("Real dual-view forward/backward passed", float(loss.detach()))

with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    labels = np.ones((16, 32, 32), dtype=np.uint32)
    labels[:, :, 16:] = 2
    raw = np.zeros((3, *labels.shape), dtype=np.float32)
    for ch, axis in enumerate((2, 1, 0)):
        cur = [slice(None)] * 3
        prev = [slice(None)] * 3
        cur[axis], prev[axis] = slice(1, None), slice(None, -1)
        raw[ch][tuple(cur)] = labels[tuple(cur)] == labels[tuple(prev)]
    write_hdf5(str(root / "labels.h5"), labels)
    write_hdf5(str(root / "raw.h5"), raw)
    config = {"_base_": str(config_path.resolve()), "test": {"data": {"test": {
        "path": "", "label": str(root / "labels.h5"),
    }}}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    score = run_benchmark(argparse.Namespace(
        config=root / "config.yaml", output=root / "score", prediction=root / "raw.h5",
        checkpoint=None, resume_to=None,
    ), crop="full")
    assert score == 0.0
    print("Real ABISS full-grid perfect-affinity score:", score)
