# AGENT.md

This file provides instructions for Codex and similar coding agents working in this repository.

## Project

PyTorch Connectomics (PyTC) is a Hydra/OmegaConf + PyTorch Lightning + MONAI codebase for EM segmentation.
Primary entry point: `scripts/main.py`.

## Mandatory Guardrails

- Preserve user-facing behavior unless explicitly requested.
- Preserve `scripts/main.py` CLI arguments and mode semantics.
- Preserve Hydra/OmegaConf key structure and override behavior.
- Prefer small, reviewable refactors over large rewrites.
- Do not add new runtime dependencies unless explicitly approved.

## Environment

- Use conda env `pytc` for all validation.
- Do not install into `base`.
- Prefer `conda run -n pytc <command>` for deterministic execution.

## Required Verification

Run these after meaningful changes (or explain why not possible):

- `conda run -n pytc python scripts/main.py --demo`
- `conda run -n pytc pytest -q`

For targeted changes, run focused tests plus the demo smoke test.

## Lint/Type Checks

Use changed-file scope unless specifically fixing global style/type debt:

- `black --check <changed_py_files>`
- `isort --check-only <changed_py_files>`
- `flake8 --max-line-length=100 <changed_py_files>`
- `mypy --config-file .github/mypy_changed.ini <changed_py_files>`

Note: repository-wide `black --check connectomics/` is not currently clean.

## Code Change Style

- Keep diffs minimal and localized.
- Avoid changing config keys and defaults without a clear migration plan.
- Keep module boundaries explicit (config, data, training, decoding).
- Add or update tests when behavior or contracts are touched.

## Git and PR Expectations

- Commit logically (one milestone/concern per commit).
- In PR descriptions, include:
  - what changed
  - why
  - exact validation commands run
  - key results

## Repository Hotspots

- Config system: `connectomics/config/`
- Data pipeline: `connectomics/data/`
- Lightning runtime: `connectomics/training/lightning/`
- Decoding/postprocess: `connectomics/decoding/`

## Safety

- Do not use destructive git commands (`reset --hard`, checkout discards) unless explicitly requested.
- If unexpected working-tree changes appear, pause and confirm intent before touching them.
