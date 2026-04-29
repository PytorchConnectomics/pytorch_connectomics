"""V2 boundary contract tests."""

from __future__ import annotations

import importlib

import pytest


def test_data_utils_compatibility_path_is_removed():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("connectomics.data.utils")


def test_config_hardware_helpers_are_not_root_exports():
    import connectomics.config as config
    import connectomics.config.hardware as hardware

    assert not hasattr(config, "auto_plan_config")
    assert not hasattr(config, "resolve_runtime_resource_sentinels")
    assert hasattr(hardware, "auto_plan_config")
    assert hasattr(hardware, "resolve_runtime_resource_sentinels")


def test_decoding_tuning_helpers_are_not_root_exports():
    import connectomics.decoding as decoding
    import connectomics.decoding.tuning as tuning
    import connectomics.runtime as runtime

    assert not hasattr(decoding, "run_tuning")
    assert not hasattr(decoding, "load_and_apply_best_params")
    assert not hasattr(tuning, "run_tuning")
    assert not hasattr(tuning, "load_and_apply_best_params")
    assert hasattr(runtime, "run_tuning")
    assert hasattr(runtime, "load_and_apply_best_params")


def test_metrics_do_not_export_file_backed_evaluation():
    import connectomics.evaluation as evaluation
    import connectomics.metrics as metrics

    assert hasattr(metrics, "evaluate_image_pair")
    assert not hasattr(metrics, "evaluate_file_pair")
    assert not hasattr(metrics, "evaluate_directory")
    assert hasattr(evaluation, "evaluate_file_pair")
    assert hasattr(evaluation, "evaluate_directory")


def test_nnunet_dimension_aliases_are_removed():
    from connectomics.models.architectures import list_architectures

    architectures = set(list_architectures())
    assert "nnunet_2d_pretrained" not in architectures
    assert "nnunet_3d_pretrained" not in architectures
    assert "nnunet_pretrained" not in architectures
    assert "nnunet" in architectures


def test_removed_public_api_exports_are_absent():
    import connectomics.config as config
    import connectomics.data.io as data_io
    import connectomics.models as models
    import connectomics.models.losses as losses
    import connectomics.training.lightning.utils as lightning_utils
    from connectomics.config import schema

    for name in {
        "merge_configs",
        "update_from_cli",
        "to_dict",
        "from_dict",
        "print_config",
        "get_config_hash",
        "create_experiment_name",
        "resolve_data_paths",
        "to_plain",
    }:
        assert not hasattr(config, name)

    assert not hasattr(data_io, "SaveVolumed")
    assert not hasattr(data_io, "TileLoaderd")
    assert not hasattr(models, "create_combined_loss")
    assert not hasattr(models, "create_loss_from_config")
    assert not hasattr(losses, "create_combined_loss")
    assert not hasattr(losses, "create_loss_from_config")
    assert not hasattr(lightning_utils, "expand_file_paths")
    assert not hasattr(schema, "DataTransformProfileConfig")
    assert not hasattr(schema, "EdgeModeConfig")
