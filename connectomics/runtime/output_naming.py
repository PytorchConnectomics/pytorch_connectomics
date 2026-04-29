"""Runtime output naming helpers for prediction, decoding, and tuning artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..config import Config
from ..utils.model_outputs import resolve_output_head


def compute_tta_passes(cfg: Config, spatial_dims: int = 3) -> int:
    """Return the total number of TTA inference passes from config."""
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return 1
    tta_cfg = getattr(inference_cfg, "test_time_augmentation", None)
    if tta_cfg is None or not bool(getattr(tta_cfg, "enabled", False)):
        return 1
    from ..inference.tta_combinations import resolve_tta_augmentation_combinations

    return len(
        resolve_tta_augmentation_combinations(
            tta_cfg,
            spatial_dims=spatial_dims,
        )
    )


def format_select_channel_tag(cfg: Config) -> str:
    """Return a compact channel-selection tag for prediction filenames."""
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return ""
    sel = getattr(inference_cfg, "select_channel", None)
    if sel is None:
        return ""

    if isinstance(sel, (list, tuple)):
        indices = [int(x) for x in sel]
    elif isinstance(sel, int):
        indices = [sel]
    elif isinstance(sel, str):
        s = sel.strip()
        if s == ":" or s == "":
            return ""
        return f"_ch{s.replace(':', '-')}"
    else:
        return ""

    return "_ch" + "-".join(str(i) for i in indices)


def format_output_head_tag(cfg: Config, *, output_head: Optional[str] = None) -> str:
    """Return a compact head-selection tag for prediction filenames."""
    if isinstance(output_head, str) and "+" in output_head:
        safe_head = re.sub(r"[^A-Za-z0-9._=-]+", "-", output_head).strip("-")
        return f"_head-{safe_head}" if safe_head else ""

    head_name = resolve_output_head(
        cfg,
        requested_head=output_head,
        purpose="prediction filename generation",
        allow_none=True,
    )
    if not head_name:
        return ""

    safe_head = re.sub(r"[^A-Za-z0-9._=-]+", "-", str(head_name)).strip("-")
    if not safe_head:
        return ""
    return f"_head-{safe_head}"


def format_decode_tag(cfg: Config) -> str:
    """Return a compact decoding-parameter tag for final prediction filenames."""

    def _sanitize_decode_component(text: str) -> str:
        safe_text = re.sub(r"[^A-Za-z0-9._=]+", "-", text)
        safe_text = re.sub(r"-{2,}", "-", safe_text)
        return safe_text.strip("-")

    gated_value_groups = {
        "branch_merge": [
            "iou_threshold",
            "branch_iou_threshold",
            "best_buddy",
            "branch_best_buddy",
            "one_sided_threshold",
            "branch_one_sided_threshold",
            "one_sided_min_size",
            "branch_one_sided_min_size",
            "affinity_threshold",
            "branch_affinity_threshold",
        ],
        "dust_merge": [
            "dust_merge_size",
            "dust_merge_affinity",
            "dust_remove_size",
        ],
        "use_aff_uint8": [],
        "use_seg_uint32": [],
    }
    gate_labels = {
        "use_aff_uint8": "aff8",
        "use_seg_uint32": "seg32",
    }

    def _flatten_decode_values(value) -> list[str]:
        if hasattr(value, "items"):
            value_dict = dict(value)
            gated_keys = set()
            for gate_key, value_keys in gated_value_groups.items():
                gate_value = value_dict.get(gate_key)
                if isinstance(gate_value, bool):
                    gated_keys.add(gate_key)
                    gated_keys.update(k for k in value_keys if k in value_dict)

            result: list[str] = []
            for key, nested_value in sorted(value_dict.items()):
                if key in gated_keys:
                    if key in gated_value_groups and nested_value is True:
                        if key in gate_labels:
                            result.append(gate_labels[key])
                        for grouped_key in gated_value_groups[key]:
                            if grouped_key in value_dict:
                                result.extend(_flatten_decode_values(value_dict[grouped_key]))
                    continue
                result.extend(_flatten_decode_values(nested_value))
            return result
        if isinstance(value, (list, tuple)):
            result: list[str] = []
            for nested_value in value:
                result.extend(_flatten_decode_values(nested_value))
            return result
        if isinstance(value, bool):
            return ["true" if value else "false"]
        if value is None:
            return ["none"]
        if isinstance(value, float):
            return [format(value, "g")]
        return [str(value)]

    decoding_cfg = getattr(cfg, "decoding", None)
    if decoding_cfg is None:
        return ""
    decoding = getattr(decoding_cfg, "steps", None)
    if not decoding:
        return ""

    try:
        steps = list(decoding)
    except TypeError:
        steps = [decoding]

    parts = []
    for step in steps:
        name = getattr(step, "name", None) or (step.get("name") if isinstance(step, dict) else None)
        if not name:
            continue

        short = name.replace("decode_", "")
        kwargs = getattr(step, "kwargs", None)
        if kwargs is None and isinstance(step, dict):
            kwargs = step.get("kwargs", {})

        value_tokens = _flatten_decode_values(kwargs) if kwargs is not None else []
        if not value_tokens:
            parts.append(short)
            continue

        kwargs_tag = _sanitize_decode_component("-".join(value_tokens))
        parts.append(f"{short}_{kwargs_tag}" if kwargs_tag else short)

    if not parts:
        return ""
    return "_" + "__".join(parts)


def format_checkpoint_name_tag(checkpoint_path: Optional[str | Path]) -> str:
    """Return a compact checkpoint tag for prediction cache filenames."""
    if checkpoint_path is None:
        return ""

    path_value = str(checkpoint_path).strip()
    if not path_value:
        return ""

    stem = Path(path_value).expanduser().stem.strip()
    if not stem:
        return ""

    safe_stem = re.sub(r"[^A-Za-z0-9._=-]+", "-", stem).strip("-")
    if not safe_stem:
        return ""

    return f"_ckpt-{safe_stem}"


def final_prediction_output_tag(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the final decoded prediction tag used in output filenames."""
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    head = format_output_head_tag(cfg, output_head=output_head)
    ch = format_select_channel_tag(cfg)
    ckpt = format_checkpoint_name_tag(checkpoint_path)
    dec = format_decode_tag(cfg)
    label = "_decoding" if dec else "_prediction"
    return f"x{n}{head}{ch}{ckpt}{label}{dec}"


def tta_cache_suffix(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the TTA prediction cache suffix."""
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    head = format_output_head_tag(cfg, output_head=output_head)
    ch = format_select_channel_tag(cfg)
    ckpt = format_checkpoint_name_tag(checkpoint_path)
    return f"_tta_x{n}{head}{ch}{ckpt}_prediction.h5"


def tta_cache_suffix_candidates(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> list[str]:
    """Return exact TTA cache suffix candidates ordered from most to least specific."""
    candidates = [
        tta_cache_suffix(
            cfg,
            spatial_dims=spatial_dims,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
        )
    ]
    if checkpoint_path is not None:
        return candidates

    legacy_suffix = tta_cache_suffix(cfg, spatial_dims=spatial_dims, output_head=output_head)
    if legacy_suffix not in candidates:
        candidates.append(legacy_suffix)
    return candidates


def tuning_artifact_tag(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the cache-style tuning tag without leading underscore or file extension."""
    return Path(
        tta_cache_suffix(
            cfg,
            spatial_dims=spatial_dims,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
        )
    ).stem.lstrip("_")


def tuning_best_params_filename(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the checkpoint/channel-aware best-params filename for tune outputs."""
    return (
        "best_params_"
        f"{tuning_artifact_tag(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path, output_head=output_head)}.yaml"
    )


def tuning_best_params_filename_candidates(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> list[str]:
    """Return ordered candidate best-params filenames, including legacy fallback."""
    candidates = [
        tuning_best_params_filename(
            cfg,
            spatial_dims=spatial_dims,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
        )
    ]
    legacy_name = "best_params.yaml"
    if legacy_name not in candidates:
        candidates.append(legacy_name)
    return candidates


def tuning_study_db_filename(
    cfg: Config,
    study_name: str,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return a checkpoint/channel-aware SQLite filename for saved Optuna studies."""
    safe_study = re.sub(r"[^A-Za-z0-9._=-]+", "-", str(study_name)).strip("-")
    if not safe_study:
        safe_study = "parameter_optimization"
    return (
        f"{safe_study}_"
        f"{tuning_artifact_tag(cfg, spatial_dims=spatial_dims, checkpoint_path=checkpoint_path, output_head=output_head)}.db"
    )


def resolve_prediction_cache_suffix(
    cfg: Config,
    mode: str,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the expected prediction cache suffix for the current runtime mode."""
    inference_cfg = getattr(cfg, "inference", None)
    save_prediction_cfg = getattr(inference_cfg, "save_prediction", None)
    configured_suffix = getattr(save_prediction_cfg, "cache_suffix", "_x1_prediction.h5")

    if mode in ("tune", "tune-test"):
        return tta_cache_suffix(cfg, checkpoint_path=checkpoint_path, output_head=output_head)

    if mode == "test":
        tta_cfg = getattr(inference_cfg, "test_time_augmentation", None)
        if tta_cfg is not None and bool(getattr(tta_cfg, "enabled", False)):
            return tta_cache_suffix(cfg, checkpoint_path=checkpoint_path, output_head=output_head)

    head = format_output_head_tag(cfg, output_head=output_head)
    ckpt = format_checkpoint_name_tag(checkpoint_path)
    if head or ckpt:
        return f"_x1{head}{ckpt}_prediction.h5"
    return configured_suffix


def is_tta_cache_suffix(suffix: str | None) -> bool:
    """Return True for any TTA intermediate prediction suffix."""
    if not suffix:
        return False
    return suffix.startswith("_tta_x") and suffix.endswith("_prediction.h5")


__all__ = [
    "compute_tta_passes",
    "format_checkpoint_name_tag",
    "format_decode_tag",
    "format_output_head_tag",
    "format_select_channel_tag",
    "final_prediction_output_tag",
    "is_tta_cache_suffix",
    "resolve_prediction_cache_suffix",
    "tta_cache_suffix",
    "tta_cache_suffix_candidates",
    "tuning_artifact_tag",
    "tuning_best_params_filename",
    "tuning_best_params_filename_candidates",
    "tuning_study_db_filename",
]
