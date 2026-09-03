#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX.

    python scripts/export_onnx.py --config tutorials/neuron_j0126/1_train.yaml \
        --checkpoint outputs/.../checkpoints/step=00200000.ckpt --output model.onnx

The graph is the MAIN head only. Deep supervision is switched off before tracing, so the
auxiliary scales are neither computed nor emitted -- which is what the inference path
consumes. The output is RAW LOGITS: the activation named by the config
(``inference.channel_activations``, e.g. ``scale_sigmoid``) is applied downstream by the
inference stage, not inside the model, and an ONNX consumer has to apply it too.

Spatial dims are FIXED at the training window (``model.input_size`` unless ``--window``
overrides it); only the batch axis is dynamic. This is deliberate for normalization layers
that carry no running statistics -- MedNeXt uses ``GroupNorm(num_groups=C, num_channels=C)``,
so its statistics are computed over the window's spatial extent and the forward pass is
window-size dependent. A dynamic spatial axis would export cleanly and then return silently
wrong output at any other size.

``--check`` verifies the export by running onnxruntime against PyTorch on the same input and
reporting the difference; it is skipped with a warning when onnxruntime is not installed.

Neither ``onnx`` nor ``onnxruntime`` is a dependency of this project, and ``torch.onnx.export``
needs ``onnx`` to serialize. Rather than adding them to a shared training environment, layer a
venv over it::

    python -m venv --system-site-packages /tmp/onnx-venv
    /tmp/onnx-venv/bin/pip install onnx onnxruntime
    /tmp/onnx-venv/bin/python scripts/export_onnx.py ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

# Add parent directory to path for direct script execution.
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from connectomics.config import load_config  # noqa: E402
from connectomics.config.pipeline.stage_resolver import resolve_default_profiles  # noqa: E402
from connectomics.models import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="config YAML the checkpoint was trained with")
    parser.add_argument("--checkpoint", required=True, help="Lightning .ckpt to export")
    parser.add_argument("--output", required=True, help="destination .onnx path")
    parser.add_argument(
        "--window",
        type=int,
        nargs=3,
        default=None,
        metavar=("D", "H", "W"),
        help="fixed spatial input size; defaults to model.input_size from the config",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--reference",
        default=None,
        help="optional .npz to write with the traced input and PyTorch output",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run onnxruntime against PyTorch and report the max absolute difference",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="max RELATIVE difference accepted by --check (|diff|max / |reference|max)",
    )
    return parser.parse_args()


def load_checkpoint_into(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a Lightning checkpoint's model weights, strictly.

    Lightning stores the module under ``model.``; the wrapper this function receives is that
    submodule, so exactly one prefix is stripped. Strict loading is the point -- a silently
    partial load exports an ONNX file full of freshly initialized weights.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    stripped = {k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")}
    if not stripped:
        raise ValueError(f"{checkpoint_path} has no 'model.'-prefixed weights in its state_dict")
    model.load_state_dict(stripped, strict=True)


def disable_deep_supervision(model: torch.nn.Module) -> None:
    """Trace the main head alone, without gradient checkpointing.

    Deep-supervision trunks return a list of scales, which would make the ONNX graph carry
    four outputs no inference path reads. Gradient checkpointing re-enters the graph during
    tracing and is unnecessary under no_grad.
    """
    trunk = getattr(model, "model", model)
    if hasattr(trunk, "do_ds"):
        trunk.do_ds = False
    model.supports_deep_supervision = False
    if getattr(trunk, "outside_block_checkpointing", False):
        trunk.outside_block_checkpointing = False
    for module in trunk.modules():
        if hasattr(module, "do_checkpointing"):
            module.do_checkpointing = False


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = resolve_default_profiles(load_config(args.config), mode="test")
    window = tuple(args.window) if args.window else tuple(cfg.model.input_size)
    print(
        f"arch={cfg.model.arch.type} in={cfg.model.in_channels} "
        f"out={cfg.model.out_channels} window={list(window)}"
    )

    model = build_model(cfg)
    load_checkpoint_into(model, args.checkpoint)
    model.eval()
    disable_deep_supervision(model)

    torch.manual_seed(0)
    dummy = torch.randn(1, cfg.model.in_channels, *window)
    with torch.no_grad():
        reference = model(dummy)
    if not torch.is_tensor(reference):
        raise TypeError(
            f"expected a single output tensor after disabling deep supervision, got {type(reference)}"
        )
    print(
        f"torch: {tuple(dummy.shape)} -> {tuple(reference.shape)} "
        f"range [{reference.min():.4f}, {reference.max():.4f}]"
    )

    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB, opset {args.opset})")

    if args.reference:
        np.savez_compressed(args.reference, image=dummy.numpy(), logits=reference.numpy())
        print(f"wrote {args.reference}")

    if args.check:
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            print("WARNING: onnxruntime is not installed; export NOT verified")
            return
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        actual = session.run(["logits"], {"image": dummy.numpy()})[0]
        expected = reference.numpy()
        diff = float(np.abs(actual - expected).max())
        # Relative, not absolute: two conv implementations differ by float32 accumulation
        # noise proportional to the magnitude of the logits, which is unbounded.
        relative = diff / max(float(np.abs(expected).max()), 1e-12)
        print(
            f"onnxruntime vs torch: max |diff| = {diff:.3e}, relative = {relative:.3e} "
            f"(tolerance {args.tolerance:.0e})"
        )
        if relative > args.tolerance:
            raise SystemExit(f"ONNX output differs from PyTorch by {relative:.3e} relative")
        print("export verified")


if __name__ == "__main__":
    main()
