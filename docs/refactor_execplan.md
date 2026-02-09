# Refactor Execution Plan (Maintainability + Developer Ergonomics)

## Baseline Snapshot (PHASE 0)
Date: 2026-02-08

- `python scripts/main.py --demo` -> failed immediately: `ModuleNotFoundError: No module named 'torch'`.
- Fastest declared suite from `tests/README.md` is `tests/unit/`:
  - `pytest tests/unit -q` -> command not found.
  - `python -m pytest tests/unit -q` -> `No module named pytest`.
- Existing lint/format commands configured in CI (`.github/workflows/tests.yml`) include `black`, `flake8`, `isort`, `mypy`:
  - `black --check connectomics/` -> command not found.
  - `python -m black --check connectomics/` -> `No module named black`.

Baseline conclusion: current environment is missing runtime/dev dependencies required to execute the repository's smoke/test/lint checks.

## Objectives
- Reduce high-complexity module hotspots into smaller, single-purpose units while preserving existing behavior.
- Eliminate duplicated helper logic and centralize shared utilities.
- Remove import-cycle and hidden coupling risks in dataset/training internals.
- Keep CLI, Hydra/OmegaConf config semantics, and public imports stable via compatibility facades/shims.
- Improve test coverage around refactor-sensitive behavior before moving logic.

## Non-Goals
- No training algorithm changes, model architecture changes, or metric/decoding logic changes.
- No changes to expected outputs/checkpoint formats beyond equivalent refactor-safe behavior.
- No config key renaming/removal unless fully backward-compatible shims are added.
- No large-scale package reorganization that breaks import paths in one step.
- No new heavyweight runtime dependencies.

## Invariants To Preserve
- `scripts/main.py` CLI contract (arguments, defaults, override passthrough, and mode behavior).
- Existing Hydra/OmegaConf structure and CLI override semantics for `system`, `data`, `model`, `optimization`, `monitor`, `inference`, `test`, and `tune`.
- Current public imports from:
  - `connectomics.config`
  - `connectomics.training.lit`
  - `connectomics.data.dataset`
- Current run/checkpoint directory behavior and config save behavior for train/test/tune/tune-test modes.

## PHASE 0 Findings (Coupling / Duplication / Boundaries / Config Sprawl)
- Tight coupling / circular import:
  - Static import graph found one cycle: `connectomics.data.dataset.build <-> connectomics.data.dataset.dataset_volume`.
  - `dataset_volume.py` imports `create_data_dicts_from_paths` from `build.py`, while `build.py` lazily imports volume datasets.
- Duplicated utilities:
  - `expand_file_paths` exists in both `connectomics/training/lit/config.py` and `connectomics/training/lit/utils.py`.
  - Validation iteration auto-calculation logic appears in multiple branches of `connectomics/training/lit/config.py`.
- Unclear module boundaries:
  - `connectomics/training/lit/config.py` mixes dataset building, interactive dataset download prompting, datamodule wrappers, run directory creation, and checkpoint mutation in one file (~1000 LOC).
  - `scripts/demo.py` duplicates training/datamodule/trainer assembly logic instead of reusing the same factories.
- Configuration sprawl / compatibility drift:
  - Data location concepts are split across `data.*`, `test.data.*`, `tune.data.*`, and legacy-looking `inference.data` references in tutorial configs.
  - `justfile` defines `--mode infer` while CLI parser mode choices are `train|test|tune|tune-test`.
  - Overlap in similarly named knobs (`data.pad_size`, `data.image_transform.pad_size`, `inference.sliding_window.pad_size`) increases ambiguity.

## Proposed Target Module Boundaries (Internal)
- `connectomics/training/lit/cli.py`
  - CLI parse + high-level config assembly (`parse_args`, `setup_config`) only.
- `connectomics/training/lit/data_factory.py`
  - Datamodule/dataset assembly and mode-specific dataset selection only.
- `connectomics/training/lit/runtime.py`
  - Run directory lifecycle and checkpoint state mutation only.
- `connectomics/training/lit/utils.py`
  - Small pure helpers only (no orchestration).
- `connectomics/data/dataset/data_dicts.py`
  - Shared data-dict constructors used by both builders and datasets (break cycle).
- Compatibility policy:
  - Keep old import locations (`lit/config.py`, `dataset/build.py`) as facades that re-export moved functions.

## Milestones (PR-Sized)

### Milestone 1: Characterization Tests + Guardrails
Files touched (planned):
- `tests/unit/test_lit_utils.py`
- `tests/unit/test_hydra_config.py`
- `tests/unit/test_main_cli_contract.py` (new)
- `tests/unit/test_run_directory_contract.py` (new)

Risk: Low

Verification commands:
- `python scripts/main.py --demo`
- `python -m pytest tests/unit/test_lit_utils.py tests/unit/test_hydra_config.py tests/unit/test_main_cli_contract.py tests/unit/test_run_directory_contract.py -q`

### Milestone 2: Utility Deduplication (No Behavior Change)
Files touched (planned):
- `connectomics/training/lit/utils.py`
- `connectomics/training/lit/config.py`
- `connectomics/training/lit/path_utils.py` (new)
- `tests/unit/test_lit_utils.py`

Scope:
- Single canonical `expand_file_paths` implementation.
- Single canonical validation-iter computation helper.
- Keep legacy function names as wrappers.

Risk: Low-Medium

Verification commands:
- `python scripts/main.py --demo`
- `python -m pytest tests/unit/test_lit_utils.py -q`

### Milestone 3: Extract Runtime/Checkpoint Orchestration
Files touched (planned):
- `connectomics/training/lit/config.py`
- `connectomics/training/lit/runtime.py` (new)
- `connectomics/training/lit/__init__.py`
- `scripts/main.py`
- `tests/unit/test_run_directory_contract.py`

Scope:
- Move `setup_run_directory`, `cleanup_run_directory`, `modify_checkpoint_state` into dedicated module.
- Preserve call signatures and re-export through existing import paths.

Risk: Medium

Verification commands:
- `python scripts/main.py --demo`
- `python -m pytest tests/unit/test_run_directory_contract.py tests/unit/test_lit_utils.py -q`

### Milestone 4: Break Dataset Import Cycle + Factory Hygiene
Files touched (planned):
- `connectomics/data/dataset/build.py`
- `connectomics/data/dataset/dataset_volume.py`
- `connectomics/data/dataset/data_dicts.py` (new)
- `connectomics/data/dataset/__init__.py`
- `tests/unit/test_monai_transforms.py`
- `tests/integration/test_dataset_multi.py`

Scope:
- Move shared data-dict helper(s) out of `build.py` to cycle-free module.
- Keep public factory API in `build.py` intact via re-export.

Risk: Medium

Verification commands:
- `python scripts/main.py --demo`
- `python -m pytest tests/integration/test_dataset_multi.py tests/unit/test_monai_transforms.py -q`

### Milestone 5: DataFactory Boundary Cleanup in Lightning Layer
Files touched (planned):
- `connectomics/training/lit/config.py`
- `connectomics/training/lit/data_factory.py` (new)
- `connectomics/training/lit/__init__.py`
- `tests/unit/test_lit_utils.py`
- `tests/integration/test_config_integration.py`

Scope:
- Move datamodule construction logic from `lit/config.py` into `lit/data_factory.py` in small steps.
- Keep existing function entrypoint (`create_datamodule`) and behavior.

Risk: Medium-High

Verification commands:
- `python scripts/main.py --demo`
- `python -m pytest tests/unit/test_lit_utils.py tests/integration/test_config_integration.py -q`

## Test Strategy
Per milestone:
- Always run smoke command: `python scripts/main.py --demo`.
- Run smallest relevant unit/integration slice for touched area.
- Run formatter/linter commands already configured by repo CI when available in env:
  - `black --check connectomics/`
  - `flake8 connectomics/ --max-line-length=100`
  - `isort --check connectomics/`

Before final completion of all milestones:
- `python scripts/main.py --demo`
- `python -m pytest tests/unit -q`
- `python -m pytest tests/integration -q`
- Optional (if environment/time allows): `python -m pytest tests/e2e -q`

## Definition of Done
- Demo smoke passes.
- Refactor-touched tests pass.
- No CLI argument regressions in `scripts/main.py`.
- No breakage for existing config hierarchy and CLI overrides.
- No import-path breakage for existing public module entrypoints.
- No new runtime dependencies introduced.
