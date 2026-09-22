# Code v3

## Overview

Fixed all accepted review_v2 findings without restructuring the approved code_v2 implementation.

## What Changed

- Moved source-dataset discovery into the S3 branch.
- Added real run-volume stub coverage.
- Implemented task-flag key comparison against `SCRATCH_PATH/done/*.txt`.
- Fixed no-background relabeling.
- Recorded resolved `ws_high` and `ws_low`.
- Isolated dtype preflight testing.

## Implementation Details

Task keys use `${STATSD_PREFIX}_${STAGE}_${OP}_${CHUNK}` and are compared bidirectionally against stored `.txt` markers.

Diagnostic JSON records the exact thresholds passed to `ws`.

## Files Changed

| File | Purpose |
|---|---|
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | Corrected affinity discovery ordering and preserved the GPU half |
| `tutorials/neuron_liconn_moe/gcloud/resolve_chunked.py` | Added real task-flag inspection and resolved-threshold diagnostics |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | Fixed relabel offset for references without background |
| `tests/unit/test_abiss_s3_chunked.py` | Added stub, task-flag, relabel, diagnostic, and dtype tests |

## Git Baseline

run_start_ref: 7d819bdba52596744f2e49eeb4e26e77b017216f  
current_head: 7d819bdba52596744f2e49eeb4e26e77b017216f

## Verification

- S3 unit tests: 13 passed
- Required focused suites: 18 passed
- `bash -n tutorials/neuron_liconn_moe/gcloud/run_volume.sh`: PASS
- Python compilation: PASS
- `git diff --check`: PASS
- GPU-half diff against the baseline: EMPTY
- VM/cloud acceptance: not run

## Review Focus

The stub test covers inference with a missing affinity file, CPU source-dataset ordering before `ref_max_compressed`, and the default sweep path.

## Risks and Unknowns

ABISS VM execution, CloudVolume production I/O, and ExPID108 acceptance remain unverified. The existing m2 behavior remains: `resolution_xyz` is normalized to integers by `abiss_chunk.py`, so the template’s fractional `23.9811` value is not preserved in resolved layer metadata.

## Changes Since Previous Code Version

- **M1:** Restored the GPU affinity block exactly to the baseline form and moved `SOURCE_DATASET` discovery into the S3 branch after recomputing `AFF_H5`.
- **m-new-1:** Offset relabeled arrays and object indices when the smallest reference label is nonzero; added coverage.
- **m-new-2:** Added resolved `ws_high` and `ws_low` values to the diagnostic JSON.
- **m-new-3:** Task-key naming and storage now follow ABISS’s `${STATSD_PREFIX}_${STAGE}_${OP}_${CHUNK}` and `done/{key}.txt` format. The optional `MERGE_CRITERION` payload cleanup, single-source `AGG_THRESHOLD` cleanup, one-variant channel guard, and resume optimization were not changed.
- **m2:** Left integer `resolution_xyz` handling unchanged and documented the risk above.
- **m5:** Replaced the constant self-comparison with bidirectional comparison against actual task markers under `SCRATCH_PATH/done`; added missing-plus-extra flag coverage.
- **m6:** Fixed source-dataset discovery placement, so it is available for CPU-only S3 runs without breaking fresh GPU runs.
- **m8:** Added a real `run_volume.sh` stub test and made the dtype failure use a correctly shaped fixture.