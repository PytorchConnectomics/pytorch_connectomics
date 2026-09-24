# Code v0
## Overview

Implemented plan_v3’s S3 chunked ABISS decode and equivalence-test scaffolding.

## What Changed

Added canonical probability-space affinity conversion, strict chunked resolution/preflight guards, VOI equivalence checks, S3 configuration, and offline tests.

## Implementation Details

- Reuses ABISS affinity helpers and `uncompress`/`EPS`.
- Validates `mean` as the only chunked criterion.
- Enforces chunk alignment, channel order, and non-degenerate boundary/interior masks.
- Uses `W != 0` for chunk-size comparison.
- Converts affinity globally with destination storage and float32 output.
- Added max-integrity comparison support.

## Files Changed

| File | Purpose |
|---|---|
| `tutorials/neuron_liconn_moe/gcloud/make_prob_affinity.py` | Canonical affinity conversion |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | Pure resolution and preflight guards |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | VOI equivalence and integrity checks |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | S3 chunked configuration |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | Whole/chunked S3 orchestration |
| `tests/unit/test_abiss_s3_chunked.py` | Offline verification tests |
| `tutorials/neuron_liconn_ist/merge_fn_sweep.py` | Safe import guard for reused helpers |

## Git Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f  
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Verification

- `bash -n tutorials/neuron_liconn_moe/gcloud/run_volume.sh` — PASS.
- Pure `resolve` test with filesystem/network patched to raise — PASS.
- `pytest tests/unit/test_abiss_s3_chunked.py` — 6 passed.
- `pytest tests/unit/test_v3_guardrails.py tests/unit/test_v2_boundaries.py` — 18 passed.
- Combined focused suite — 24 passed.
- `git diff --check` — PASS.
- Required public API command — FAIL: `tests/unit/test_public_api_snapshot.py` does not exist.
- Black/flake8 checks — NOT RUN successfully; neither tool is installed.
- VM item 9 — NOT RUN.
- VM item 10 — NOT RUN.
- VM item 11 — NOT RUN.
- VM item 12 — NOT RUN.
- VM item 13 — NOT RUN.
- VM item 14 — NOT RUN.

## Review Focus

Check canonical tensor face handling, chunk-size alignment, criterion rejection messaging, masked VOI semantics, `W != 0` comparison, and `|B| > 0`, `|I| > 0` enforcement.

## Risks and Unknowns

Cloud execution, ABISS binaries, ExPID108 artifacts, and VM acceptance remain unverified. No cloud commands or VMs were used.

## Changes Since Previous Code Version

Initial implementation.