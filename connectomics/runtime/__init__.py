"""Runtime and CLI support helpers."""

from importlib import import_module

from .output_naming import (
    compute_tta_passes,
    final_prediction_output_tag,
    format_checkpoint_name_tag,
    format_decode_tag,
    format_output_head_tag,
    format_select_channel_tag,
    is_tta_cache_suffix,
    resolve_prediction_cache_suffix,
    tta_cache_suffix,
    tta_cache_suffix_candidates,
    tuning_artifact_tag,
    tuning_best_params_filename,
    tuning_best_params_filename_candidates,
    tuning_study_db_filename,
)
from .preflight import preflight_check, print_preflight_issues

_LAZY_EXPORTS = {
    "load_and_apply_best_params": "connectomics.runtime.tune_runner",
    "run_tuning": "connectomics.runtime.tune_runner",
    "temporary_tuning_inference_overrides": "connectomics.runtime.tune_runner",
}


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "compute_tta_passes",
    "final_prediction_output_tag",
    "format_checkpoint_name_tag",
    "format_decode_tag",
    "format_output_head_tag",
    "format_select_channel_tag",
    "is_tta_cache_suffix",
    "load_and_apply_best_params",
    "preflight_check",
    "print_preflight_issues",
    "resolve_prediction_cache_suffix",
    "run_tuning",
    "temporary_tuning_inference_overrides",
    "tta_cache_suffix",
    "tta_cache_suffix_candidates",
    "tuning_artifact_tag",
    "tuning_best_params_filename",
    "tuning_best_params_filename_candidates",
    "tuning_study_db_filename",
]
