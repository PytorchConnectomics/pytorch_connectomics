"""Runtime and CLI support helpers."""

from importlib import import_module

_LAZY_EXPORTS = {
    "compute_tta_passes": "connectomics.runtime.output_naming",
    "final_prediction_output_tag": "connectomics.runtime.output_naming",
    "format_checkpoint_name_tag": "connectomics.runtime.output_naming",
    "format_decode_tag": "connectomics.runtime.output_naming",
    "format_output_head_tag": "connectomics.runtime.output_naming",
    "format_select_channel_tag": "connectomics.runtime.output_naming",
    "handle_test_cache_hit": "connectomics.runtime.cache_resolver",
    "has_assigned_test_shard": "connectomics.runtime.sharding",
    "is_tta_cache_suffix": "connectomics.runtime.output_naming",
    "load_and_apply_best_params": "connectomics.runtime.tune_runner",
    "maybe_enable_independent_test_sharding": "connectomics.runtime.sharding",
    "maybe_limit_test_devices": "connectomics.runtime.sharding",
    "parse_args": "connectomics.runtime.cli",
    "preflight_check": "connectomics.runtime.preflight",
    "preflight_test_cache_hit": "connectomics.runtime.cache_resolver",
    "print_preflight_issues": "connectomics.runtime.preflight",
    "resolve_prediction_cache_suffix": "connectomics.runtime.output_naming",
    "resolve_test_stage_runtime": "connectomics.runtime.sharding",
    "run_tuning": "connectomics.runtime.tune_runner",
    "setup_config": "connectomics.runtime.cli",
    "shard_test_datamodule": "connectomics.runtime.sharding",
    "temporary_tuning_inference_overrides": "connectomics.runtime.tune_runner",
    "tta_cache_suffix": "connectomics.runtime.output_naming",
    "tta_cache_suffix_candidates": "connectomics.runtime.output_naming",
    "tuning_artifact_tag": "connectomics.runtime.output_naming",
    "tuning_best_params_filename": "connectomics.runtime.output_naming",
    "tuning_best_params_filename_candidates": "connectomics.runtime.output_naming",
    "tuning_study_db_filename": "connectomics.runtime.output_naming",
    "try_cache_only_test_execution": "connectomics.runtime.cache_resolver",
    "validate_runtime_coherence": "connectomics.runtime.preflight",
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
    "handle_test_cache_hit",
    "has_assigned_test_shard",
    "load_and_apply_best_params",
    "maybe_enable_independent_test_sharding",
    "maybe_limit_test_devices",
    "parse_args",
    "preflight_test_cache_hit",
    "preflight_check",
    "print_preflight_issues",
    "resolve_prediction_cache_suffix",
    "resolve_test_stage_runtime",
    "run_tuning",
    "setup_config",
    "shard_test_datamodule",
    "temporary_tuning_inference_overrides",
    "try_cache_only_test_execution",
    "tta_cache_suffix",
    "tta_cache_suffix_candidates",
    "tuning_artifact_tag",
    "tuning_best_params_filename",
    "tuning_best_params_filename_candidates",
    "tuning_study_db_filename",
    "validate_runtime_coherence",
]
