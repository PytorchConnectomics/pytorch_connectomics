"""Runtime output naming helpers for prediction, decoding, and tuning artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..utils.model_outputs import get_inference_select_channel, resolve_output_head


_UNINFORMATIVE_STEMS = {"img", "image", "raw", "em", "main", "data"}

# Container-style directory suffixes that should be skipped when walking up
# from an image path: e.g. `/data/seed101/data.zarr/img` should resolve to
# `seed101`, not `data.zarr`. Lowercase, includes the dot.
_CONTAINER_PARENT_SUFFIXES = (".zarr", ".n5", ".ome.zarr")


def _is_container_dir_name(name: str) -> bool:
    lname = name.strip().lower()
    if not lname:
        return False
    return any(lname.endswith(suffix) for suffix in _CONTAINER_PARENT_SUFFIXES)


def _resolve_data_input_for_mode(cfg: Config, mode: str) -> Any:
    """Return the relevant DataInputConfig for the active runtime mode."""
    data_cfg = getattr(cfg, "data", None)
    if mode in ("tune", "tune-test"):
        return getattr(data_cfg, "val", None)
    return getattr(data_cfg, "test", None)


def _expand_paths(value: Any) -> list[str]:
    """Expand a path/glob/list-of-paths value to a flat list of strings."""
    from connectomics.training.lightning.path_utils import expand_file_paths

    if value is None:
        return []
    try:
        return list(expand_file_paths(value))
    except Exception:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value if v is not None]
        return []


def _stem_from_image_path(path: str | Path) -> str:
    """Derive a per-volume stem from an image path with deterministic fallbacks.

    1. Filename stem if informative.
    2. Walk up the parent chain, skipping uninformative names (``img``, ``data``,
       …) and container directories (``*.zarr``, ``*.n5``, ``*.ome.zarr``).
    3. Fall back to filename stem unconditionally, else ``"volume"``.

    No `base` parameter: the resolver and the writer-side filename helper
    must derive identical stems for the same path; the `base` kwarg used to
    cause the resolver to stop early when the dataset root coincided with
    the volume identity dir (e.g. NISB's ``seed101/data.zarr/img``).

    Examples:
        ``/data/seed101/data.zarr/img`` → ``seed101``
        ``/data/seed101/img.h5``        → ``seed101``  (stem ``img`` skipped)
        ``/data/sample.h5``             → ``sample``
        ``/data/seed101/raw_aff.h5``    → ``raw_aff``
    """
    p = Path(str(path))
    stem = p.stem.strip()
    if stem and stem.lower() not in _UNINFORMATIVE_STEMS:
        return stem

    # Walk up parents, skipping container dirs and uninformative names.
    parent = p.parent
    while parent and parent != parent.parent:
        name = parent.name.strip()
        if not name:
            break
        if _is_container_dir_name(name) or name.lower() in _UNINFORMATIVE_STEMS:
            parent = parent.parent
            continue
        return name

    # No informative parent. Only return the original stem if it is itself
    # informative; otherwise fall back to a generic placeholder so multiple
    # uninformative-stem volumes don't collide on the same per-volume dir.
    if stem and stem.lower() not in _UNINFORMATIVE_STEMS:
        return stem
    return "volume"


def resolve_dataset_volume_stems(cfg: Config, mode: str) -> list[str]:
    """Return canonical per-volume names for the active data split.

    Resolution order per dataset item:
    1. Explicit ``data.{val|test}.name`` value (str applied uniformly, or
       indexed list).
    2. Parent dir of ``decoding.load_prediction_path`` /
       ``inference.load_tta_path`` if either is set (decode-only flow).
    3. ``Path(image_filename).stem`` if informative.
    4. Parent dir name if stem is uninformative.
    5. ``volume_<idx>`` last resort.
    """
    decoding_cfg = getattr(cfg, "decoding", None)
    load_pred = getattr(decoding_cfg, "load_prediction_path", "") or ""
    if load_pred:
        return [Path(load_pred).expanduser().parent.name or "volume_0"]

    inference_cfg = getattr(cfg, "inference", None)
    load_tta = getattr(inference_cfg, "load_tta_path", "") or ""
    if load_tta:
        return [Path(load_tta).expanduser().parent.name or "volume_0"]

    data_input = _resolve_data_input_for_mode(cfg, mode)
    if data_input is None:
        return []

    image_value = getattr(data_input, "image", None)
    image_paths = _expand_paths(image_value)

    explicit_name = getattr(data_input, "name", None)
    if isinstance(explicit_name, str):
        if not image_paths:
            return [explicit_name]
        return [explicit_name] * len(image_paths)
    if isinstance(explicit_name, (list, tuple)):
        names = [str(n) for n in explicit_name]
        if image_paths and len(names) == len(image_paths):
            return names
        if not image_paths:
            return names

    if not image_paths:
        return []

    # Use the canonical helper without `base` so the resolver and the writer
    # path (`resolve_output_filenames` -> `_stem_from_image_path` w/o base)
    # always derive identical stems for the same image path. Passing `base`
    # caused the walker to stop early when `data.{val,test}.path` happened to
    # coincide with the volume identity dir (e.g. NISB's `seed101/data.zarr/img`),
    # making the resolver fall back to "volume" while the writer still wrote
    # under "seed101".
    return [
        _stem_from_image_path(p) or f"volume_{idx}"
        for idx, p in enumerate(image_paths)
    ]


def resolve_volume_save_dir(cfg: Config, mode: str, volume_stem: str) -> Path:
    """Return ``<inference.save_path>/<volume_stem>``.

    Used by the test/tune output writers to materialize per-volume subfolders
    under the checkpoint-derived results dir.
    """
    inference_cfg = getattr(cfg, "inference", None)
    save_path = getattr(inference_cfg, "save_path", "") or ""
    if not save_path:
        # Fallback: parent of decoding.load_prediction_path if present.
        decoding_cfg = getattr(cfg, "decoding", None)
        load_pred = getattr(decoding_cfg, "load_prediction_path", "") or ""
        if load_pred:
            save_path = str(Path(load_pred).expanduser().parent.parent)
    return Path(save_path) / volume_stem


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
    sel = get_inference_select_channel(cfg)
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


def _format_one_decode_step(step) -> str:
    """Encode a single decode step as ``{name}_{kwargs_tokens}`` (no leading underscore)."""

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

    name = getattr(step, "name", None) or (step.get("name") if isinstance(step, dict) else None)
    if not name:
        return ""
    short = name.replace("decode_", "")
    kwargs = getattr(step, "kwargs", None)
    if kwargs is None and isinstance(step, dict):
        kwargs = step.get("kwargs", {})

    value_tokens = _flatten_decode_values(kwargs) if kwargs is not None else []
    if not value_tokens:
        return short
    kwargs_tag = _sanitize_decode_component("-".join(value_tokens))
    return f"{short}_{kwargs_tag}" if kwargs_tag else short


def format_intermediate_decode_suffix(cfg: Config, step) -> str:
    """Return per-step intermediate suffix, e.g. ``_decoding_affinity_cc_numba-0-0.75``."""
    encoded = _format_one_decode_step(step)
    if not encoded:
        return ""
    return f"_decoding_{encoded}"


def format_decode_tag(cfg: Config) -> str:
    """Return a compact decoding-parameter tag for final prediction filenames."""
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

    parts = [encoded for encoded in (_format_one_decode_step(step) for step in steps) if encoded]
    if not parts:
        return ""
    return "_" + "__".join(parts)


def format_decoding_output_suffix_tag(cfg: Config) -> str:
    """Return an optional user-controlled suffix for decoded output filenames."""
    decoding_cfg = getattr(cfg, "decoding", None)
    if decoding_cfg is None:
        return ""
    suffix = getattr(decoding_cfg, "save_suffix", "")
    if suffix is None:
        return ""

    safe_suffix = re.sub(r"[^A-Za-z0-9._=-]+", "-", str(suffix).strip())
    safe_suffix = re.sub(r"-{2,}", "-", safe_suffix).strip("-_")
    if not safe_suffix:
        return ""
    return f"_{safe_suffix}"


def _is_chunked_raw_prediction(cfg: Config) -> bool:
    inference_cfg = getattr(cfg, "inference", None)
    if inference_cfg is None:
        return False
    strategy = str(getattr(inference_cfg, "strategy", "whole_volume")).lower()
    chunking_cfg = getattr(inference_cfg, "chunking", None)
    if chunking_cfg is None:
        return False
    enabled = strategy == "chunked" or bool(getattr(chunking_cfg, "enabled", False))
    output_mode = str(getattr(chunking_cfg, "output_mode", "decoded")).lower()
    return enabled and output_mode == "raw_prediction"


def format_chunked_raw_cache_tag(cfg: Config) -> str:
    """Return a cache tag that separates chunked raw outputs from whole-volume raw caches."""
    if not _is_chunked_raw_prediction(cfg):
        return ""

    chunking_cfg = cfg.inference.chunking
    parts = ["chunked-raw"]
    chunk_size = getattr(chunking_cfg, "chunk_size", None)
    if chunk_size:
        parts.append("cs" + "x".join(str(int(v)) for v in chunk_size))
    halo = getattr(chunking_cfg, "halo", None)
    if halo and any(int(v) != 0 for v in halo):
        parts.append("halo" + "x".join(str(int(v)) for v in halo))

    suffix = format_decoding_output_suffix_tag(cfg).lstrip("_")
    if suffix:
        parts.append(suffix)
    return "_" + "_".join(parts)


def _sanitized_checkpoint_stem(checkpoint_path: Optional[str | Path]) -> str:
    """Return the sanitised stem of a checkpoint path, or empty string.

    Collapses Lightning's ``auto_insert_metric_name=True`` artifact
    (``step-step=N`` → ``step=N``) and replaces non-portable characters
    with ``-``. Shared by checkpoint directory naming and prediction
    cache filename tagging so both stay consistent.
    """
    if checkpoint_path is None:
        return ""

    path_value = str(checkpoint_path).strip()
    if not path_value:
        return ""

    stem = Path(path_value).expanduser().stem.strip()
    if not stem:
        return ""

    stem = re.sub(r"(^|-)([A-Za-z_][A-Za-z0-9_]*)-\2=", r"\1\2=", stem)
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", stem).strip("-")


def format_checkpoint_name_tag(checkpoint_path: Optional[str | Path]) -> str:
    """Return a compact checkpoint tag for prediction cache filenames."""
    safe_stem = _sanitized_checkpoint_stem(checkpoint_path)
    if not safe_stem:
        return ""
    return f"_ckpt-{safe_stem}"


def format_checkpoint_dir_suffix(checkpoint_path: Optional[str | Path]) -> str:
    """Return a sanitised, filesystem-safe tag for a checkpoint path.

    Used by ``checkpoint_dispatch`` to build per-checkpoint directory
    names (e.g. ``test_step=00050000``, ``test_last``). Same sanitiser
    as :func:`format_checkpoint_name_tag` minus the leading ``_ckpt-``
    prefix.
    """
    safe_stem = _sanitized_checkpoint_stem(checkpoint_path)
    if not safe_stem:
        return "ckpt"
    return safe_stem


def final_prediction_output_tag(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the final decoded artifact filename for the per-volume layout.

    Filename format: ``decoded_x{n}{head}{ch}_<dec>{user}.h5`` (or
    ``prediction_x{n}{head}{ch}{user}.h5`` when no decoders are configured).
    The dataset stem and checkpoint identity are encoded by the parent
    directory (``<save_path>/<volume_stem>``); they are no longer in the
    filename. ``checkpoint_path`` is accepted for API compatibility but is
    not used.
    """
    del checkpoint_path  # encoded by parent dir
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    head = format_output_head_tag(cfg, output_head=output_head)
    ch = format_select_channel_tag(cfg)
    dec = format_decode_tag(cfg)
    suffix = format_decoding_output_suffix_tag(cfg)
    label = "decoded" if dec else "prediction"
    return f"{label}_x{n}{head}{ch}{dec}{suffix}.h5"


def intermediate_decode_step_output_tag(
    cfg: Config,
    step: Any,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the per-step intermediate decoded artifact filename.

    Filename format: ``decoded_step<idx>_x{n}{head}{ch}_<step_tag>{user}.h5``
    where ``<step_tag>`` is the per-step encoding from
    ``format_intermediate_decode_suffix`` (without leading
    ``_decoding_``). Stem + checkpoint encoded by parent dir.
    """
    del checkpoint_path
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    head = format_output_head_tag(cfg, output_head=output_head)
    ch = format_select_channel_tag(cfg)
    step_tag_full = format_intermediate_decode_suffix(cfg, step)
    # format_intermediate_decode_suffix returns `_decoding_<encoded>`; strip
    # the legacy `_decoding_` prefix because the per-volume layout already
    # encodes type via the leading `decoded_` keyword.
    step_payload = step_tag_full.removeprefix("_decoding_")
    suffix = format_decoding_output_suffix_tag(cfg)
    return f"decoded_x{n}{head}{ch}_{step_payload}{suffix}.h5"


def raw_cache_suffix(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the raw prediction artifact filename for the per-volume layout.

    Filename format: ``raw_x{n}{head}{ch}.h5``. All cached/saved model
    predictions use the same ``raw_`` prefix regardless of whether TTA is
    enabled. The ``_x{n}`` token carries the TTA pass count. Stem and
    checkpoint identity are encoded by the parent directory.
    """
    del checkpoint_path
    n = compute_tta_passes(cfg, spatial_dims=spatial_dims)
    head = format_output_head_tag(cfg, output_head=output_head)
    ch = format_select_channel_tag(cfg)
    return f"raw_x{n}{head}{ch}.h5"


def intermediate_prediction_cache_suffix(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the raw/intermediate prediction cache filename for the active strategy.

    Returns ``raw_x{n}{head}{ch}.h5`` for whole-volume; appends a
    chunked-raw token (and any user save_suffix) for chunked raw mode.
    """
    suffix = raw_cache_suffix(
        cfg,
        spatial_dims=spatial_dims,
        checkpoint_path=checkpoint_path,
        output_head=output_head,
    )
    strategy_tag = format_chunked_raw_cache_tag(cfg)
    if not strategy_tag:
        return suffix
    return suffix.removesuffix(".h5") + f"{strategy_tag}.h5"


def intermediate_prediction_cache_suffix_candidates(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> list[str]:
    """Return cache suffix candidates. Per-volume layout encodes ckpt by parent
    dir, so this returns a single candidate; the parameter is kept for API
    compat."""
    return [
        intermediate_prediction_cache_suffix(
            cfg,
            spatial_dims=spatial_dims,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
        )
    ]


def raw_cache_suffix_candidates(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> list[str]:
    """Per-volume layout returns a single canonical candidate."""
    return [
        raw_cache_suffix(
            cfg,
            spatial_dims=spatial_dims,
            checkpoint_path=checkpoint_path,
            output_head=output_head,
        )
    ]


def tuning_artifact_tag(
    cfg: Config,
    spatial_dims: int = 3,
    checkpoint_path: Optional[str | Path] = None,
    output_head: Optional[str] = None,
) -> str:
    """Return the cache-style tuning tag without leading underscore or file extension."""
    return Path(
        raw_cache_suffix(
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
    """Return the expected prediction cache filename for the current runtime mode."""
    return intermediate_prediction_cache_suffix(
        cfg, checkpoint_path=checkpoint_path, output_head=output_head
    )


def is_raw_cache_suffix(suffix: str | None) -> bool:
    """Return True for any raw/intermediate prediction cache filename."""
    if not suffix:
        return False
    name = Path(suffix).name
    return name.startswith("raw_x") and name.endswith(".h5")


__all__ = [
    "compute_tta_passes",
    "format_checkpoint_dir_suffix",
    "format_checkpoint_name_tag",
    "format_chunked_raw_cache_tag",
    "format_decode_tag",
    "format_decoding_output_suffix_tag",
    "format_intermediate_decode_suffix",
    "intermediate_decode_step_output_tag",
    "format_output_head_tag",
    "format_select_channel_tag",
    "final_prediction_output_tag",
    "intermediate_prediction_cache_suffix",
    "intermediate_prediction_cache_suffix_candidates",
    "is_raw_cache_suffix",
    "raw_cache_suffix",
    "raw_cache_suffix_candidates",
    "resolve_dataset_volume_stems",
    "resolve_prediction_cache_suffix",
    "resolve_volume_save_dir",
    "tuning_artifact_tag",
    "tuning_best_params_filename",
    "tuning_best_params_filename_candidates",
    "tuning_study_db_filename",
]
