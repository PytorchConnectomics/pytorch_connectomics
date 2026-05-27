# PyTorch Connectomics — Agent Add-Architecture Prompt

You are registering a new model architecture in this repo. Follow the
steps below; only edit the files explicitly listed under "Files to
touch".

## Inputs to ask the user for first

- Architecture short name (e.g. `my_unet3d`).
- Source: existing PyTorch module (path) or external repo (URL).
- Expected I/O shape (e.g. `(B, 1, 128, 128, 128) → (B, K, ...)`).
- Deep-supervision support: yes / no.
- One tutorial YAML to smoke against (default: `tutorials/minimal.yaml`).

If any required input is missing, stop and ask.

## Steps

1. Read `connectomics/models/architectures/registry.py` for the
   `@register_architecture("name")` decorator interface.

2. Pick the closest existing wrapper as your template:
   `monai_models.py` (MONAI-style), `mednext_models.py`
   (predefined-size), or `rsunet.py` (raw PyTorch).

3. Add a builder function decorated with
   `@register_architecture("<name>")` in a new or existing file under
   `connectomics/models/architectures/`. **If you created a new
   file**, also append `from .<new_module> import *` (or an explicit
   import line) to `connectomics/models/architectures/__init__.py`,
   otherwise the decorator never runs and step 6 fails.

4. Add architecture-specific config params to
   `connectomics/config/schema/model.py` (or a sibling `model_*.py`).
   Strict-key validation will reject undeclared fields.

5. Smoke run on one batch:
   `python scripts/main.py --config tutorials/minimal.yaml model.arch.type=<name> --fast-dev-run`.

6. Confirm registration:
   `python -c "from connectomics.models.architectures import list_architectures; assert '<name>' in list_architectures()"`.

7. Run boundary tests:
   `python -m pytest tests/unit/test_v3_guardrails.py tests/unit/test_v2_boundaries.py -q`.

## Verification

- Step 5 exits 0; step 6 prints nothing (assert passes); step 7 is green.

## Stop conditions

- External dependency not in `pyproject.toml` → stop and ask.
- Touching files outside the two listed paths → stop and ask.
- Same shape/dtype error twice in a row → stop and report.

## Files to touch

- One new or existing file in `connectomics/models/architectures/`.
- `connectomics/models/architectures/__init__.py` — only if step 3 created a new file (one-line import).
- At most one schema file in `connectomics/config/schema/`.
