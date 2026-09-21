"""Prove the stripped release checkpoint is functionally identical to the source.

Builds the model from the frozen run config, loads both checkpoints, and compares
a real 128^3 forward pass on an actual crop of the ExPID82_1 val volume.
"""
import sys

import numpy as np
import torch
import zarr

sys.path.insert(0, "/projects/weilab/weidf/lib/pytorch_connectomics")

from connectomics.config.pipeline.config_io import load_config
from connectomics.config.pipeline.stage_resolver import resolve_default_profiles
from connectomics.models.build import build_model

RUN = "outputs/liconn_final_banis_plus_tube/20260728_032436"
SRC = f"{RUN}/checkpoints/step=00200000.ckpt"
REL = "artifacts/hf_liconn/affinity_expid82_18nm_128x128x128.ckpt"
VAL = "/projects/weilab/dataset/liconn/pytc/final_proofread/val/data.zarr/img"

cfg = resolve_default_profiles(
    load_config("tutorials/neuron_nisb/liconn_final_banis+_tube.yaml"), mode="test"
)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {dev}")


def load(path):
    # checkpoint keys are model.model.* : LightningModule.model -> wrapper.model
    wrapper = build_model(cfg)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    wrapper.load_state_dict(sd, strict=True)
    return wrapper.to(dev).eval()


# real tissue, not noise: centre crop of the held-out val volume
z = zarr.open(VAL, mode="r")
c = [s // 2 for s in z.shape]
crop = z[c[0] - 64:c[0] + 64, c[1] - 64:c[1] + 64, c[2] - 64:c[2] + 64]
print(f"crop {crop.shape} uint8 range [{crop.min()}, {crop.max()}]")
x = torch.from_numpy(crop.astype(np.float32) / 255.0)[None, None].to(dev)

# TF32 makes conv output vary run-to-run; pin it so "identical weights" is testable
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def fwd(path):
    with torch.no_grad():
        out = load(path)(x)
    return out[0] if isinstance(out, (list, tuple)) else out


a = fwd(SRC)
a2 = fwd(SRC)          # control: same file twice -> isolates run-to-run noise
b = fwd(REL)

print(f"output {tuple(a.shape)} {a.dtype}")
print(f"source logits  min {a.min():.4f} max {a.max():.4f} mean {a.mean():.4f}")
print(f"release logits min {b.min():.4f} max {b.max():.4f} mean {b.mean():.4f}")

d_ctrl = (a - a2).abs().max().item()
d_rel = (a - b).abs().max().item()
print(f"\ncontrol  max|src - src |  = {d_ctrl:.3e}")
print(f"release  max|src - rel |  = {d_rel:.3e}")

sig = torch.sigmoid(0.2 * b)  # the scale_sigmoid the pipeline stores
print(f"scale_sigmoid(release) range [{sig.min():.4f}, {sig.max():.4f}] "
      f"mean {sig.mean():.4f}")

assert torch.equal(a, a2), f"forward pass is not deterministic (control {d_ctrl:.3e})"
assert torch.equal(a, b), f"RELEASE DIVERGES: {d_rel:.3e} vs control {d_ctrl:.3e}"
print("\nPASS: deterministic forward, release bit-identical to source on real tissue.")
