"""Pre-flight validation checks for training runs."""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

from ..data.processing.build import count_stacked_label_transform_channels
from ..models.architectures.registry import get_architecture_info
from ..utils.channel_slices import infer_min_required_channels
from ..utils.model_outputs import resolve_output_heads


def _architecture_supports_deep_supervision(arch_type: str) -> bool:
    """Infer deep-supervision support from architecture registry metadata."""
    arch_info = get_architecture_info().get(arch_type)
    if arch_info is None:
        return True

    module_name = arch_info.get("module", "")
    return not module_name.endswith("monai_models")


def validate_runtime_coherence(cfg) -> None:
    """Validate cross-section runtime coherence that depends on data/model helpers."""
    model_input = list(cfg.model.input_size)
    patch_size = list(cfg.data.dataloader.patch_size)
    if model_input != patch_size:
        raise ValueError(
            "Cross-section validation failed: model.input_size "
            f"{model_input} must match data.dataloader.patch_size {patch_size}."
        )

    out_channels = cfg.model.out_channels
    model_heads = getattr(cfg.model, "heads", None) or {}
    primary_head = getattr(cfg.model, "primary_head", None)
    sole_head = next(iter(model_heads.keys())) if len(model_heads) == 1 else None
    required_output_channels: List[tuple[str, int]] = []

    label_cfg = getattr(cfg.data, "label_transform", None)
    stacked_label_channels = (
        count_stacked_label_transform_channels(label_cfg) if label_cfg is not None else None
    )

    def _resolve_selector_head(entry: Any, *, selector_key: str) -> Optional[str]:
        if selector_key == "pred2_slice":
            selector_head = entry.get("pred2_head", entry.get("pred_head", None))
        else:
            selector_head = entry.get("pred_head", None)

        if selector_head is None:
            selector_head = primary_head or sole_head
        return selector_head

    def _validate_head_channel_capacity(
        *,
        selector_key: str,
        selector_head: str,
        min_channels: int,
        loss_idx: int,
    ) -> bool:
        if selector_head not in model_heads:
            return False

        head_channels = int(getattr(model_heads[selector_head], "out_channels", 0))
        if min_channels > head_channels:
            raise ValueError(
                "Cross-section validation failed: "
                f"model.loss.losses[{loss_idx}].{selector_key} requires at least "
                f"{min_channels} channels in head '{selector_head}', but "
                f"model.heads.{selector_head}.out_channels is {head_channels}."
            )
        return True

    def _validate_label_channel_capacity(selector_value: Any, *, path: str) -> None:
        min_channels = infer_min_required_channels(selector_value, context=path)
        if min_channels is None:
            return

        if stacked_label_channels is not None:
            if min_channels > stacked_label_channels:
                raise ValueError(
                    "Cross-section validation failed: "
                    f"{path} requires at least {min_channels} stacked label channels, but "
                    f"data.label_transform.targets produces {stacked_label_channels}."
                )
            return

        if not model_heads:
            required_output_channels.append((path, min_channels))

    model_loss_cfg = getattr(cfg.model, "loss", None)
    losses_cfg = getattr(model_loss_cfg, "losses", None) if model_loss_cfg else None
    if losses_cfg is not None:
        for i, entry in enumerate(losses_cfg):
            if not isinstance(entry, dict):
                continue
            selector_keys = ("pred_slice", "target_slice", "mask_slice", "pred2_slice")
            for selector_key in selector_keys:
                min_channels = infer_min_required_channels(
                    entry.get(selector_key),
                    context=f"model.loss.losses[{i}].{selector_key}",
                )
                if min_channels is not None:
                    path = f"model.loss.losses[{i}].{selector_key}"
                    if selector_key in {"pred_slice", "pred2_slice"} and model_heads:
                        selector_head = _resolve_selector_head(entry, selector_key=selector_key)
                        if selector_head is not None and _validate_head_channel_capacity(
                            selector_key=selector_key,
                            selector_head=selector_head,
                            min_channels=min_channels,
                            loss_idx=i,
                        ):
                            continue
                    if selector_key in {"target_slice", "mask_slice"}:
                        _validate_label_channel_capacity(entry.get(selector_key), path=path)
                        continue
                    required_output_channels.append((path, min_channels))

    if not model_heads and stacked_label_channels:
        required_output_channels.append(("data.label_transform.targets", stacked_label_channels))

    if model_heads:
        for head_name, head_cfg in model_heads.items():
            target_slice = getattr(head_cfg, "target_slice", None)
            if target_slice is None:
                continue
            _validate_label_channel_capacity(
                target_slice,
                path=f"model.heads.{head_name}.target_slice",
            )

    decoding_section = getattr(cfg, "decoding", None)
    decoding_cfg = getattr(decoding_section, "steps", None)
    decode_has_channel_selection = False
    decode_available_channels = out_channels
    decode_channel_scope = "model output"
    if decoding_cfg:
        for decode_step in decoding_cfg:
            kwargs = getattr(decode_step, "kwargs", None)
            if isinstance(kwargs, dict) and any(key.endswith("_channels") for key in kwargs):
                decode_has_channel_selection = True
                break

        if model_heads and decode_has_channel_selection:
            decode_heads = resolve_output_heads(cfg, purpose="decode channel selection")
            if len(model_heads) > 1 and not decode_heads:
                raise ValueError(
                    "Cross-section validation failed: decode channel selectors require "
                    "inference.head or model.primary_head when model.heads has multiple "
                    f"entries ({sorted(model_heads.keys())})."
                )
            if len(decode_heads) > 1:
                decode_available_channels = sum(
                    int(getattr(model_heads[h], "out_channels", 0)) for h in decode_heads
                )
                decode_channel_scope = f"merged heads {decode_heads}"
            elif decode_heads:
                decode_output_head = decode_heads[0]
                if decode_output_head in model_heads:
                    decode_available_channels = int(
                        getattr(model_heads[decode_output_head], "out_channels", out_channels)
                    )
                    decode_channel_scope = f"head '{decode_output_head}'"

        for i, decode_step in enumerate(decoding_cfg):
            kwargs = getattr(decode_step, "kwargs", None)
            if not isinstance(kwargs, dict):
                continue
            for key, value in kwargs.items():
                if not key.endswith("_channels"):
                    continue
                min_channels = infer_min_required_channels(
                    value,
                    context=f"decoding[{i}].kwargs.{key}",
                )
                if min_channels is not None:
                    path = f"decoding[{i}].kwargs.{key}"
                    if model_heads and decode_has_channel_selection:
                        if min_channels > decode_available_channels:
                            raise ValueError(
                                "Cross-section validation failed: "
                                f"{path} requires at least {min_channels} channels in "
                                f"{decode_channel_scope}, but only "
                                f"{decode_available_channels} are available."
                            )
                        continue
                    required_output_channels.append((path, min_channels))

    tta_cfg = getattr(cfg.inference, "test_time_augmentation", None)
    channel_activations = getattr(tta_cfg, "channel_activations", None) if tta_cfg else None
    select_channel = getattr(cfg.inference, "select_channel", None)
    inference_has_channel_selection = bool(channel_activations) or select_channel is not None
    tta_heads = (
        resolve_output_heads(cfg, purpose="inference channel selection") if model_heads else []
    )
    tta_output_head = tta_heads[0] if tta_heads else None
    if model_heads and len(model_heads) > 1 and inference_has_channel_selection and not tta_heads:
        raise ValueError(
            "Cross-section validation failed: inference channel selectors require inference.head "
            "or model.primary_head when model.heads has multiple entries "
            f"({sorted(model_heads.keys())})."
        )
    if len(tta_heads) > 1:
        tta_available_channels = sum(
            int(getattr(model_heads[h], "out_channels", 0)) for h in tta_heads
        )
        tta_channel_scope = f"merged heads {tta_heads}"
    else:
        tta_available_channels = (
            int(getattr(model_heads[tta_output_head], "out_channels", out_channels))
            if tta_output_head in model_heads
            else out_channels
        )
        tta_channel_scope = (
            f"head '{tta_output_head}'" if tta_output_head in model_heads else "model output"
        )

    def _validate_tta_channel_capacity(selector_value: Any, *, path: str) -> None:
        min_selector_channels = infer_min_required_channels(
            selector_value,
            context=path,
        )
        if min_selector_channels is None:
            return
        if min_selector_channels > tta_available_channels:
            raise ValueError(
                "Cross-section validation failed: "
                f"{path} requires at least {min_selector_channels} channels in "
                f"{tta_channel_scope}, but only {tta_available_channels} are available."
            )

    if isinstance(channel_activations, list):
        for i, spec in enumerate(channel_activations):
            if not isinstance(spec, dict):
                raise ValueError(
                    "Cross-section validation failed: "
                    f"inference.test_time_augmentation.channel_activations[{i}] "
                    "must be a mapping with 'channels' and 'activation'."
                )
            if "channels" not in spec or "activation" not in spec:
                raise ValueError(
                    "Cross-section validation failed: "
                    f"inference.test_time_augmentation.channel_activations[{i}] "
                    "must define both 'channels' and 'activation'."
                )
            _validate_tta_channel_capacity(
                spec["channels"],
                path=f"inference.test_time_augmentation.channel_activations[{i}].channels",
            )
    _validate_tta_channel_capacity(
        select_channel,
        path="inference.select_channel",
    )

    if required_output_channels:
        required_max = max(req for _, req in required_output_channels)
        if required_max > out_channels:
            details = ", ".join(
                f"{path} needs >= {req}"
                for path, req in sorted(required_output_channels, key=lambda x: x[1], reverse=True)
            )
            raise ValueError(
                "Cross-section validation failed: model.out_channels is "
                f"{out_channels}, but resolved pipeline components require at least "
                f"{required_max} channels ({details})."
            )

    deep_supervision = (
        getattr(model_loss_cfg, "deep_supervision", False) if model_loss_cfg else False
    )
    if deep_supervision:
        arch_type = getattr(cfg.model.arch, "type", "")
        if not _architecture_supports_deep_supervision(arch_type):
            raise ValueError(
                "Cross-section validation failed: model.loss.deep_supervision=True but "
                f"architecture '{arch_type}' does not support deep supervision. "
                "Use MedNeXt/RSUNet or disable deep supervision."
            )


def preflight_check(cfg) -> list:
    """Run pre-flight checks before training."""
    issues = []

    def _is_virtual_data_path(path_value: str) -> bool:
        return isinstance(path_value, str) and path_value.startswith("random://")

    def _iter_data_paths(path_value):
        if path_value is None:
            return []
        if isinstance(path_value, list):
            return path_value
        return [path_value]

    def _validate_training_paths(path_value, kind: str) -> None:
        for raw_path in _iter_data_paths(path_value):
            if _is_virtual_data_path(raw_path):
                continue
            if "*" in raw_path or "?" in raw_path:
                if not glob(raw_path):
                    issues.append(f"ERROR: Training {kind} pattern matched no files: {raw_path}")
            elif not Path(raw_path).exists():
                issues.append(f"ERROR: Training {kind} not found: {raw_path}")

    _validate_training_paths(cfg.data.train.image, "image")
    _validate_training_paths(cfg.data.train.label, "label")

    if cfg.system.num_gpus > 0 and not torch.cuda.is_available():
        issues.append(f"ERROR: {cfg.system.num_gpus} GPU(s) requested but CUDA not available")

    if cfg.system.num_gpus > torch.cuda.device_count():
        issues.append(
            f"ERROR: {cfg.system.num_gpus} GPU(s) requested but only "
            f"{torch.cuda.device_count()} available"
        )

    if torch.cuda.is_available() and cfg.system.num_gpus > 0:
        try:
            patch_volume = np.prod(cfg.data.dataloader.patch_size)
            estimated_gb = (
                cfg.data.dataloader.batch_size * patch_volume * cfg.model.in_channels * 4 * 10 / 1e9
            )
            available_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

            if estimated_gb > available_gb * 0.8:
                issues.append(
                    "WARNING: Estimated memory ({estimated:.1f}GB) may exceed available "
                    "({available:.1f}GB)".format(estimated=estimated_gb, available=available_gb)
                )
                issues.append("   Tip: Consider reducing batch_size or patch_size")
        except Exception:
            pass

    if cfg.data.dataloader.patch_size:
        patch_size = cfg.data.dataloader.patch_size
        if min(patch_size) < 16:
            issues.append(
                f"WARNING: Very small patch size: {patch_size} (may not capture enough context)"
            )
        if max(patch_size) > 256:
            issues.append(f"WARNING: Very large patch size: {patch_size} (may cause GPU OOM)")

    optimizer_cfg = getattr(getattr(cfg, "optimization", None), "optimizer", None)
    lr = getattr(optimizer_cfg, "lr", None)
    if lr is not None:
        if lr > 1e-2:
            issues.append(f"WARNING: Learning rate very high: {lr} (may cause instability)")
        if lr < 1e-6:
            issues.append(f"WARNING: Learning rate very low: {lr} (training may be very slow)")

    return issues


def print_preflight_issues(issues: list) -> None:
    """Print preflight check issues and optionally stop interactive runs."""
    if not issues:
        return

    print("\n" + "=" * 60)
    print("PRE-FLIGHT CHECK WARNINGS")
    print("=" * 60)
    for issue in issues:
        print(f"  {issue}")
    print("=" * 60 + "\n")

    import sys

    is_non_interactive = not sys.stdin.isatty() or os.environ.get("SLURM_JOB_ID") is not None
    if is_non_interactive:
        print("Non-interactive environment detected. Continuing automatically...\n")
        return

    try:
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Aborted by user")
            raise SystemExit(1)
    except KeyboardInterrupt as exc:
        print("\nAborted by user")
        raise SystemExit(1) from exc


__all__ = ["preflight_check", "print_preflight_issues", "validate_runtime_coherence"]
