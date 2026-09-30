# PyTorch Connectomics — Agent Debug-Tutorial Prompt

You are diagnosing a failing canonical tutorial — one run via
`python scripts/main.py --config tutorials/<name>.yaml`. Every shipped tutorial
must pass the structured config validator.

## Inputs to ask the user for first

- Tutorial config path (under `tutorials/`).
- Error message / stack trace.
- Recent edits to the YAML or its data paths (if any).

If any required input is missing, stop and ask.

## Steps

1. Confirm the config exists under `tutorials/`.

2. Validate only the selected config:
   `python scripts/validate_tutorial_configs.py --glob '<path>'`.

3. Reproduce in one batch:
   `python scripts/main.py --config <path> --fast-dev-run`.

4. Classify the failure and target the right area:
   - Schema (raised at config load) → `connectomics/config/schema/`.
   - Shape / dtype (raised mid-batch) → check `model.in_channels`,
     `model.out_channels`, and `connectomics/models/losses/metadata.py`.
   - Data path (`FileNotFoundError`) → `data.train|val.image|label`.
   - Decode-stage error → check each `decoding` list entry's `template:` is a
     registered template under `connectomics/config/templates/decoding_*.yaml`
     and that the decoder is registered in
     `connectomics/decoding/registry.py` via the
     `register_decoder(name, fn)` function call.
   - GPU OOM / `cc3d` / NumPy ABI → see
     `INSTALLATION.md#common-install-issues`.
   - Convergence (no shape/schema error, just bad metric) →
     out of scope; refer the user to architecture / loss / LR tuning.

5. Apply the minimum fix; rerun `--fast-dev-run` from step 3.

## Verification

- Step 3 exits 0 after the fix.

## Stop conditions

- Same failure twice with the same fix attempt → stop and report.
- Root cause requires editing `connectomics/` source → stop and ask.

## Files to touch

- The offending tutorial YAML only; editing `connectomics/` source → stop and ask.
