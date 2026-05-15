# Code v0
## Overview
Implemented the terminal `plan_v2.md` affinity QC changes. The streaming accumulator now matches post-save Z-cut math for sampled interior means while preserving raw per-Z report arrays, streaming finalization is rank-0 guarded, and streaming setup now respects a pre-wired affinity mask.

## What Changed
- Updated streaming accumulator finalization to distinguish all-zero updates from partially missing Z slices, compute the interior baseline from `means[::z_stride]`, and scan only the configured refinement windows for head/tail cuts.
- Documented the report-array contract for streaming versus post-save reports in the `AffinityQCReport` docstring and markdown report output.
- Added the pre-wired `decoding.affinity_mask_path` short-circuit to `begin_streaming_qc`.
- Moved `finish_streaming_qc` to an early `image_path` requirement and removed the unreachable no-image mask branch.
- Guarded `finish_streaming_qc` in chunked inference so only rank 0 finalizes streaming QC in distributed runs.
- Extended runtime preflight to allow streaming QC with either `affinity_qc.image_path` or an existing `decoding.affinity_mask_path`, while still rejecting configs that provide neither.
- Added the five plan-specified unit tests for interior-mean parity, windowed head scanning, all-zero finalize messaging, preflight image/mask rules, and pre-wired mask streaming setup.

## Implementation Details
`AffinityQCAccumulator.finalize` keeps accepting the `img=` keyword and now returns raw stride-1 `z_idx`, `means`, and `stds` for streaming reports. The sampled interior population is used only for baseline and cutoff math, matching `scan_prediction`.

`begin_streaming_qc` returns `None` when `decoding.affinity_mask_path` is already set, so chunked inference does not build or finalize an accumulator for pre-wired masks. `finish_streaming_qc` now fails before finalizing if `affinity_qc.image_path` is missing, because streaming does not retain enough spatial prediction data to size the output mask.

The chunked inference hook remains duck-typed through `qc_streaming_callback`; no static `connectomics.decoding` import was added to `connectomics/inference/chunked.py`.

## Files Changed
| File | Purpose |
|---|---|
| `connectomics/decoding/qc/affinity.py` | Streaming QC accumulator math, report contract docs, pre-wired mask short-circuit, early streaming image-path validation. |
| `connectomics/runtime/preflight.py` | Streaming affinity QC coherence rule for `image_path` or existing `affinity_mask_path`. |
| `connectomics/training/lightning/test_pipeline.py` | Rank-0 guard around streaming QC finalization in chunked raw prediction inference. |
| `tests/unit/test_decoding_qc_affinity.py` | Plan-specified unit coverage for streaming parity, guardrails, and preflight behavior. |
| `.ccc/features/affinity_qc/artifacts/code_v0.md` | CCC code-stage implementation artifact. |

## Git Baseline
run_start_ref: fa8fce67202394ab8b44d0f37857daabd8ba0a68
current_head: fa8fce67202394ab8b44d0f37857daabd8ba0a68

## Verification
Command:

```bash
python -m pytest tests/unit/test_decoding_qc_affinity.py \
                 tests/unit/test_v3_guardrails.py \
                 tests/unit/test_v2_boundaries.py \
                 tests/unit/test_decoding_pipeline.py -q
```

Full output:

```text
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: cov-7.0.0, anyio-4.12.1
collected 41 items

tests/unit/test_decoding_qc_affinity.py ..............                   [ 34%]
tests/unit/test_v3_guardrails.py ..........                              [ 58%]
tests/unit/test_v2_boundaries.py ......                                  [ 73%]
tests/unit/test_decoding_pipeline.py ...........                         [100%]

=============================== warnings summary ===============================
<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 41 passed, 6 warnings in 22.43s ========================
```

Command:

```bash
python scripts/validate_tutorial_configs.py
```

Full output:

```text
/home/weidf/.config/matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /tmp/matplotlib-cn8qxy1p because there was an issue with the default path (/home/weidf/.config/matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
Validated 14 canonical tutorial configs successfully; skipped 2 custom workflow YAMLs.
Custom workflows:
  - tutorials/waterz_decoding_large.yaml
  - tutorials/waterz_decoding_large_abiss.yaml
```

## Review Focus
- Confirm the streaming raw-report-array contract is acceptable while `interior_mean`, `low_z`, and `high_z` remain aligned with post-save scans.
- Check the distributed rank guard placement around `finish_streaming_qc` in `test_pipeline.py`.
- Verify the pre-wired mask path is consistently accepted by preflight and skipped by streaming setup.

## Risks and Unknowns
- Existing repository state was already dirty before this implementation; this stage did not attempt to separate or revert unrelated changes.
- `validate_tutorial_configs.py` emitted the existing Matplotlib cache-directory warning, but exited successfully.

## Changes Since Previous Code Version
Initial implementation.
