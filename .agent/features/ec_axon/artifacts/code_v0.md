# Code v0

## Overview

Implemented Plan v6 with the current user's v6.1 naming and no-YAML-tuning
contract. The validated MIT-LiCONN algorithms are vendored into generic
`seg_2d`, `branch_link`, `branch_split`, and `branch_merge` graph operations;
generic statistics/LUT and completeness helpers live in their canonical
packages. The tutorial now runs the same affinity through exact naive-waterz
and staged branch decoding paths and reports base plus oracle-merge NERL at
`merge_threshold=10`.

## What Changed

- Replaced the legacy branch split/merge implementations with the validated
  combined stages, preserving both CUE LADDER module narratives.
- Ported the reachable `small=0` 2D-section and `no_force_split=True` linking
  closure without any package import from `dev/`.
- Added canonical one-pass `seg_stats`, chunked in-place `apply_lut`, and
  GT-free completeness reporting.
- Registered the four v6.1 graph operations with unary/binary arity checks.
- Made graph `output` prune downstream execution and encoded the validated
  node chain, ordered inputs, and selected output in artifact cache names.
- Added a fixed `naive_waterz` wrapper: exact 80-slice `decode_waterz` recipe,
  validated border stitching, global union relabel, and no YAML kwargs.
- Added dense-label propagation for decode-only evaluation, majority-GT
  oracle merging, report/log support, and independent oracle-only dispatch.
- Removed the dev-importing axon wrapper, its four registrations, and the old
  tracklet tutorial.
- Added the knob-free `tutorials/neuron_axon/` comparison.

## Implementation Details

`seg_2d` ports the exact strong-section seed constants and waterz path, then
assigns volume-unique IDs. `branch_link` uses the exact conservative 0.2 IoU
spine followed by 0.3 reciprocal best buddies with force splitting disabled.
`branch_split` performs link-cut change-point splitting before the relaxed
`host_both=False` confident tunnel carve. `branch_merge` performs completion,
iterated IoU-primary mutual/margin merging, then projected-mask weak-gap
bridging; the OM-negative radius completion remains opt-in through
`prefer_length=True`.

All relabel paths use `apply_lut`; split/merge stages accept `stats=` and
`inplace=`. `seg_stats` performs one `cc3d.statistics` call. No branch stage
uses `compute_bbox_all_3d`.

The naive baseline is chunked exactly like the saved reference workflow
because a monolithic `decode_waterz` invocation does not represent that
artifact's border stitching. The public YAML remains parameter-free.

## Files Changed

| Status | Files | Purpose |
|---|---|---|
| Created | `connectomics/decoding/decoders/branch/__init__.py` | Export only the four ready branch operations. |
| Created | `connectomics/decoding/decoders/branch/sections.py` | Exact strong-section `seg_2d` seed. |
| Created | `connectomics/decoding/decoders/branch/linking.py` | Exact no-force-split tracklet linking. |
| Created | `connectomics/decoding/decoders/branch/split.py` | Combined link-cut and confident tunnel split, with CUE LADDER. |
| Created | `connectomics/decoding/decoders/branch/merge.py` | Completion, mutual/margin merge, weak bridge, optional v4, with CUE LADDER. |
| Created | `connectomics/metrics/completeness.py` | GT-free completeness ranker. |
| Created | `connectomics/metrics/oracle.py` | Majority-GT oracle-merge transformation. |
| Created | `tests/unit/test_bbox_fast_completeness.py` | Utility and completeness fixtures. |
| Created | `tests/unit/test_branch_seed.py` | Section/link seed fixtures. |
| Created | `tests/unit/test_branch_split_port.py` | Link-cut and tunnel gate fixtures. |
| Created | `tests/unit/test_branch_merge_port.py` | Completion, margin, and weak-gap fixtures. |
| Created | `tests/unit/test_decode_experiment_log.py` | Graph experiment-log coverage. |
| Created | `tests/unit/test_nerl_oracle.py` | Oracle transform, dispatch, persistence, and output-path coverage. |
| Created | `tutorials/neuron_axon/README.md` | Commands and reference comparison. |
| Created | `tutorials/neuron_axon/axon_decode.yaml` | Informatively named four-node branch DAG with no kwargs. |
| Created | `tutorials/neuron_axon/waterz_baseline.yaml` | Same-input fixed naive-waterz baseline with no kwargs. |
| Modified | `connectomics/data/processing/bbox.py` | Add canonical `seg_stats` and `apply_lut`. |
| Modified | `connectomics/config/templates/decoding_templates.yaml` | Remove obsolete parameterized legacy branch templates. |
| Modified | `connectomics/decoding/__init__.py`, `connectomics/decoding/decoders/__init__.py` | Point lazy branch exports at the canonical modules. |
| Modified | `connectomics/decoding/decoders/waterz.py` | Add exact fixed chunk-and-stitch baseline wrapper. |
| Modified | `connectomics/decoding/registry.py` | Register v6.1 ops, arity adapters, and fixed baseline. |
| Modified | `connectomics/decoding/graph.py` | Prune execution to output ancestors. |
| Modified | `connectomics/decoding/experiment_log.py` | Record validated graph operation chains and oracle score. |
| Modified | `connectomics/evaluation/context.py`, `connectomics/evaluation/nerl.py`, `connectomics/evaluation/report.py` | Decode output resolution and base/oracle NERL reporting. |
| Modified | `connectomics/runtime/cache_resolver.py` | Propagate one dense GT label in decode-only execution without a stack copy. |
| Modified | `connectomics/runtime/output_naming.py` | Collision-resistant graph-aware decoded artifact tags. |
| Modified | `tests/unit/test_decode_graph.py`, `tests/unit/test_decoding_pipeline.py` | Early-stop, registration, arity, DAG, and baseline coverage. |
| Modified | `tests/unit/test_evaluation_metrics.py`, `tests/unit/test_main_runtime_stage_switch.py` | Dense-label/oracle integration coverage. |
| Modified | `tests/unit/test_lit_utils.py` | Graph cache-tag chain/input/output coverage. |
| Modified | `tests/unit/test_v3_guardrails.py` | Static no-`dev` package-import guard. |
| Deleted | `connectomics/decoding/decoders/branch_merge.py`, `connectomics/decoding/decoders/branch_split.py` | Remove superseded legacy algorithms. |
| Deleted | `connectomics/decoding/decoders/axon_tracklet.py` | Remove the untracked dev-importing wrapper present at run start. |
| Deleted | `tutorials/axon_decoding/tracklet.yaml` | Remove the superseded untracked tutorial present at run start. |

Pre-existing unrelated WIP listed in `state/run_start.status` was preserved,
including all user-excluded schema, inference, training, tuning, NERL metric,
segmentation decoder, NISB, root README, and docs paths.

## Git Baseline

run_start_ref: 25ed266b7f4da881bf4d529a7c5e4252892f4573
current_head: 25ed266b7f4da881bf4d529a7c5e4252892f4573

No commit was created.

## Verification

| Check | Result |
|---|---|
| Full MIT-LiCONN staged artifact parity | Exact: sections, tracklets, split, v2 completion/merge, and v3 weak outputs each had zero differing voxels versus research artifacts. Final labels: 236877, 27836, 28147, 22085, 22071 respectively. |
| Full naive-waterz artifact parity | Exact: zero differing voxels, 1223 labels, 811.2 seconds. |
| NERL, `merge_threshold=10` | Waterz 0.6530/0.7580; tracklets 0.8284/0.9424; split 0.7302/0.9631; v2 0.8377/0.9541; v3 0.8434/0.9525. |
| Consolidated focused suite | `151 passed, 2 skipped` across 153 tests. |
| Tutorial validation | `Validated 17 canonical tutorial configs successfully; skipped 3 declared custom workflows.` |
| Dev-freedom | Static guard and `rg` scan found no `dev` import/reference under `connectomics/`. |
| Formatting/import/lint | Black, isort, flake8, and `git diff --check` passed on the changed surface. |
| Types | Mypy passed on 18 changed production files. `bbox.py` still reports three pre-existing errors at lines 105, 179, and 223; none are in the added helpers. |
| Boundary/public API | `tests/unit/test_v3_guardrails.py` passed. The requested `tests/unit/test_public_api_snapshot.py` is absent from this repository; the current public/import boundary checks live in `test_v3_guardrails.py`. |

The full staged parity harness took 4097.5 seconds (about 68.3 minutes) in
this environment, so the plan's `<20 min` wall-time target was not met. Exact
algorithm order/constants and the required one-pass stats/chunked LUT behavior
were retained rather than retuned.

## Review Focus

- Compare strict gates and order in `branch/{sections,linking,split,merge}.py`
  against the named research sources; the full-volume zero-difference results
  are the behavioral gate.
- Confirm graph tags include pruned node names, operations, ordered inputs, and
  output so one-line early stops cannot reuse stale artifacts.
- Confirm decode-only dense labels reach evaluation and both tutorial configs
  use the same raw affinity, GT, skeleton, and threshold 10.
- Confirm no excluded WIP is attributed to this implementation.

## Risks and Unknowns

- `branch_merge` cannot subsume `branch_link` without behavior drift:
  `branch_link` consumes globally unique one-slice sections with 0.2 spine and
  0.3 best-buddy gates, while `branch_merge` applies completion, minimum-size,
  affinity-floor, ambiguity-margin, and weak-gap logic.
- `branch_link(inplace=True)` has object-identity semantics only for the
  production `uint32` input contract; dense `seg_2d` output satisfies it.
  Extremely sparse adversarial high IDs can also make its dense LUT large,
  though the canonical seed produces dense IDs and the tested production path
  is exact.
- The on-disk `plan_v6.md` contains a later v6.2 parameter-exposure amendment,
  but the current user request explicitly reinstates v6.1 naming and says YAML
  exposes no tuning parameters. This implementation follows the current,
  higher-priority request.
- No algorithm or numerical deviations were found. The only acceptance
  deviation is the measured full-DAG wall time noted above.

## Changes Since Previous Code Version

Initial implementation.
