"""V3 refactor guardrails.

These tests document the intended package boundaries before the implementation
stages move code. Known current violations are marked strict xfail so later
stages must either fix and un-xfail them or keep the debt visible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_name_for_path(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_import_from(module_name: str, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    package_parts = module_name.split(".")
    if path_name := package_parts[-1]:
        if path_name != "__init__":
            package_parts = package_parts[:-1]
    base_parts = package_parts[: max(0, len(package_parts) - node.level + 1)]
    if node.module:
        base_parts.extend(node.module.split("."))
    return ".".join(base_parts)


def _forbidden_imports(root: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        module_name = _module_name_for_path(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [_resolve_import_from(module_name, node)]

            for imported_module in imported_modules:
                if imported_module.startswith(forbidden_prefixes):
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{node.lineno}: {imported_module}")
    return violations


@pytest.mark.xfail(strict=True, reason="V3 PR 2/5 removes decoding -> training imports")
def test_decoding_static_imports_do_not_reference_training():
    violations = _forbidden_imports(
        REPO_ROOT / "connectomics" / "decoding",
        ("connectomics.training",),
    )
    assert violations == []


@pytest.mark.xfail(strict=True, reason="V3 PR 6 moves streamed chunk decoding out of inference")
def test_inference_static_imports_do_not_reference_decoding():
    violations = _forbidden_imports(
        REPO_ROOT / "connectomics" / "inference",
        ("connectomics.decoding",),
    )
    assert violations == []


@pytest.mark.xfail(strict=True, reason="V3 PR 7 moves data-aware validation out of config")
def test_config_static_imports_do_not_reference_data_execution():
    violations = _forbidden_imports(
        REPO_ROOT / "connectomics" / "config",
        ("connectomics.data",),
    )
    assert violations == []


@pytest.mark.xfail(strict=True, reason="V3 PR 3 makes unknown top-level keys hard errors")
def test_config_load_raises_on_unknown_top_level_key(tmp_path):
    from connectomics.config import load_config

    config_yaml = tmp_path / "unknown_key.yaml"
    config_yaml.write_text("unknown_section: {}\n")

    with pytest.raises(ValueError, match="unknown_section"):
        load_config(config_yaml)


def test_connectomics_config_public_api_snapshot():
    import connectomics.config as config

    assert set(config.__all__) == {
        "Config",
        "load_config",
        "save_config",
        "merge_configs",
        "update_from_cli",
        "to_dict",
        "from_dict",
        "print_config",
        "validate_config",
        "get_config_hash",
        "create_experiment_name",
        "resolve_data_paths",
        "resolve_default_profiles",
        "to_plain",
        "as_plain_dict",
        "cfg_get",
    }


def test_connectomics_inference_public_api_snapshot():
    import connectomics.inference as inference

    assert set(inference.__all__) == {
        "InferenceManager",
        "PredictionArtifactMetadata",
        "read_prediction_artifact",
        "write_prediction_artifact",
        "write_prediction_artifact_attrs",
        "run_prediction_inference",
        "is_chunked_inference_enabled",
        "run_chunked_affinity_cc_inference",
        "run_chunked_prediction_inference",
        "apply_prediction_transform",
        "apply_storage_dtype_transform",
        "resolve_output_filenames",
        "write_outputs",
        "build_sliding_inferer",
        "resolve_inferer_roi_size",
        "resolve_inferer_overlap",
        "is_2d_inference_mode",
        "TTAPredictor",
    }
