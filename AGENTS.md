# AGENTS.md

This file is the Codex/OpenAI coding-agent memory for this repository. Claude
startup memory lives in `CLAUDE.md`; when the two differ, treat this file as the
Codex execution contract and use `CLAUDE.md` plus `.agent/architecture/` as deeper
background.

## Project

PyTorch Connectomics (PyTC) is a Hydra/OmegaConf + PyTorch Lightning + MONAI
framework for EM semantic and instance segmentation. The primary CLI entry is
`scripts/main.py`, which should remain thin: parse/setup config, then dispatch
through `connectomics.runtime`.

Core stack:
- PyTorch Lightning owns training orchestration, distributed execution, mixed
  precision, callbacks, logging, and checkpoints.
- MONAI is the domain toolkit for transforms, models, losses, and metrics.
- Hydra/OmegaConf with dataclass schemas is the only supported config system.

## Refactor Contract

The v2/v3 refactor is a clean-break architecture. Backward compatibility is
intentionally out of scope unless the user explicitly asks for it. The relevant
planning and review docs live under `.agent/architecture/`: `v3_plan.md`
(current contract), `v3_audit.md` (implementation audit), and
`v3_audit_rebuttal.md` (audit corrections).

Mandatory rules:
- One canonical owner per concept. Delete duplicate import paths, facade
  re-exports, and compatibility shims instead of preserving them.
- Strict config. Unknown top-level YAML keys raise during load. Removed fields
  raise. Do not add `getattr(cfg.x, "old_field", default)` ghost reads for
  undeclared schema fields.
- Stages are separate: `train -> infer -> decode -> evaluate -> tune`.
  Combined test mode may exist only as a wrapper that calls the same stage APIs.
- Public API is explicit and small. Keep `__all__` intentional and update
  public API snapshot tests when changing exports.
- Tests validate contracts, not legacy behavior. Update tests to the canonical
  paths and current stage boundaries.

Dependency direction:

```text
config
utils
data
models
metrics
training    -> config, data, models, metrics
inference   -> config, data, models
decoding    -> config, data, utils
evaluation  -> config, data, metrics
runtime     -> config, training, inference, decoding, evaluation
scripts     -> runtime
```

Do not introduce imports that violate this direction. In particular: config
must not import domain execution code; decoding must not import training;
inference must not run decoder-specific logic; evaluation must not depend on
`ConnectomicsModule` private methods.

## Current Package Ownership

- `connectomics/config/`: schema, YAML loading, profile/template expansion,
  stage resolution, and config-only utilities.
- `connectomics/runtime/`: CLI setup, preflight validation, mode dispatch,
  checkpoint/cache handling, output naming, sharding, and tuning orchestration.
- `connectomics/training/`: Lightning modules, datamodules, callbacks,
  optimizers, schedulers, training losses, and train/validation behavior.
- `connectomics/inference/`: model prediction over volumes, TTA, sliding-window
  and chunked raw prediction, and raw prediction artifacts.
- `connectomics/decoding/`: prediction arrays/artifacts to segmentation or
  synapse artifacts, decoder registry, decode pipelines, postprocessing, and
  pure decoding parameter tuning.
- `connectomics/evaluation/`: metric execution, NERL/ERL, reports, and
  evaluation context independent of Lightning modules.
- `connectomics/data/`: IO, datasets, MONAI transforms, augmentation,
  preprocessing, target generation, sampling, and splitting.
- `connectomics/models/`: architecture registry, model factory, loss factory,
  loss metadata, and model-side regularization.
- `connectomics/metrics/`: reusable metric implementations only.
- `connectomics/utils/`: genuinely cross-domain primitives only.

## Config Rules

- Tutorial configs should `_base_` shared registries in
  `connectomics/config/all_profiles.yaml`.
- Section-level registries live in `connectomics/config/profiles/*.yaml` and are
  selected only at canonical `*.profile` paths.
- Decoding list reuse uses explicit `template:` entries from
  `connectomics/config/templates/*.yaml`. Do not add `decoding_profile` or
  `- profile: decoding_*` aliases.
- Profile payloads are base values; explicit keys override them. Explicit lists
  replace profile lists unless an existing profile override mechanism says
  otherwise.
- Top-level `inference`, `decoding`, and `evaluation` sections are distinct.
  Do not put decode/evaluation fields back under `inference`.
- Custom large-volume workflow YAMLs under `tutorials/` may intentionally bypass
  the structured `Config` schema only when declared in
  `scripts/validate_tutorial_configs.py` as custom workflow roots.

## Environment

- Use conda env `pytc` for validation.
- Prefer `conda run -n pytc <command>` so commands are reproducible.
- Do not install dependencies into `base`.
- Do not add runtime dependencies unless explicitly approved.

## Verification

Use focused checks for the touched surface, and explain any skipped checks.
Common commands:

```bash
conda run -n pytc python scripts/main.py --demo
conda run -n pytc pytest -q
conda run -n pytc python scripts/validate_tutorial_configs.py --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'
```

For changed Python files, prefer changed-file scope unless intentionally fixing
global style debt:

```bash
conda run -n pytc black --check <changed_py_files>
conda run -n pytc isort --check-only <changed_py_files>
conda run -n pytc flake8 --max-line-length=100 <changed_py_files>
conda run -n pytc mypy --config-file .github/mypy_changed.ini <changed_py_files>
```

Repository-wide formatting/type checks may not be clean. Do not mix global
format churn into focused refactors.

## Change Style

- Keep diffs narrow and aligned with the current package ownership.
- Prefer deleting stale compatibility code over adding adapters.
- Add abstractions only when they remove real duplication or clarify ownership.
- Add or update tests when behavior, schema, public API, import boundaries, or
  artifact contracts change.
- Keep tutorial YAMLs runnable and validate them after schema/profile changes.
- Do not leave hidden behavior in `scripts/main.py` or
  `training/lightning/test_pipeline.py`; move orchestration to `runtime/` or
  the owning stage package.

## Git Safety

- The worktree may already be dirty. Treat existing changes as user-owned unless
  you made them in the current task.
- Do not use destructive git commands such as `git reset --hard` or checkout
  discards unless explicitly requested.
- Do not create new branches unless explicitly asked.
- Commit logically: one milestone or concern per commit. Include exact
  validation commands and results in PR/commit summaries when relevant.
