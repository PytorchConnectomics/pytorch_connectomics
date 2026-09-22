# Code v1

## Overview

Adjusted the full-volume substrate oracle gate so eligible-only force splitting must match the
unsplit-sections baseline without being required to reach the more aggressive force-split-all
target. No decoding, splitting, relinking, selection, or output logic changed.

## What Changed

- Lowered `SUBSTRATE_ORACLE_FLOOR` from `0.78` to `0.76`, making the hard assertion a
  no-regression check against the unsplit-sections baseline.
- Retained `0.78` as `SUBSTRATE_ORACLE_TARGET`, used only for reporting.
- Expanded the substrate report to print the exact oracle-merge NERL and its signed delta from the
  `0.78` target.
- Updated the assertion failure message to identify the eligible substrate and the
  unsplit-sections floor.

## Implementation Details

The full-mode evaluation now reports
`substrate_oracle - SUBSTRATE_ORACLE_TARGET` with an explicit sign. The only hard substrate gate
remains `substrate_oracle >= SUBSTRATE_ORACLE_FLOOR`, where the floor is `0.76`. Thus a substrate
between `0.76` and `0.78` is reported as below target but does not abort the run.

## Files Changed

| File | Purpose |
|---|---|
| `dev/mit_liconn/decode_v2.py` | Separate the substrate no-regression floor from the reporting-only oracle target. |
| `.agent/features/decode_v2/artifacts/code_v1.md` | Record the CCC code-v1 fix and verification results. |

## Git Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
current_head: e8844b3da0f0a992431c901e7e2034486e7a678b

## Verification

- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python dev/mit_liconn/decode_v2.py --self-test` — exit 0; `decode_v2 self-test: PASS`.
- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m py_compile dev/mit_liconn/decode_v2.py` — exit 0.
- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && flake8 --max-line-length=100 dev/mit_liconn/decode_v2.py` — exit 0.
- The full SLURM job was not run, as requested.

## Review Focus

Confirm that `0.76` is the sole hard substrate oracle threshold and that `0.78` affects only the
reported signed target delta. Also confirm that no pipeline behavior outside this gate/report was
changed.

## Risks and Unknowns

The exact eligible-substrate oracle-merge NERL and the full-volume real/oracle NERL values remain
unknown until the full SLURM run. The revised gate prevents an expected `0.76`-to-`0.78` substrate
result from aborting that run while still rejecting regression below the unsplit-sections baseline.

## Changes Since Previous Code Version

Replaced the code-v0 substrate hard floor of `0.78` with the unsplit-sections no-regression floor
of `0.76`, retained `0.78` as a reporting-only target, and added the signed target delta to the
substrate output.
