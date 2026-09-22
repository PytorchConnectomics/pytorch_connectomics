# Code v0

## Overview

Implemented plan v3 without commits or staging. Competitive nucleus growth is now an instrumented,
retryable scan/flood/merge workflow with deterministic internal territory ids, explicit emitted-id
translation, input and code fingerprints, plan-bound fan-in, and transactional canonical manifest
publication. The real final-remap overlay path is covered by an oracle-bearing regression test.

This implementation is ready for the coordinator/operator to run the prescribed production
baselines. Efficiency acceptance gate 6 was not run and is not claimed.

## What Changed

- Split the former monolithic `nuccomp` execution into `scan`, fixed-capacity `flood`, and `merge`
  actions while retaining `run` as a serial baseline path.
- Added timestamped phase instrumentation for input fingerprinting, input open, `scan_geometry`,
  `map_to_watershed`, contact detection, unit planning, each flood, and merge. Successful evidence is
  atomically published as a run-scoped `stage_report.json`; failures preserve
  `stage_report.partial.json` without replacing the canonical manifest.
- Preallocated deterministic internal ids from `(parent, anchor)`. Merge selects the winner by
  largest pooled voxel count and then lowest anchor id, records both `internal_territory_id` and
  `emitted_id`, and validates the exact id domain, exactly one emitted parent, and uniqueness of all
  non-parent emissions.
- Added run-scoped staging, SHA-256 fingerprints for all specified inputs, exact `plan_digest`
  binding for unit records, atomic publication, a 64-task default capacity, and retry-safe Slurm
  scan/flood/merge launchers.
- Made the overlay validate the completion marker, expected plan, authoritative watershed identity,
  territory fingerprint, and explicit translation rather than re-deriving the largest territory.
- Replaced the monkeypatched B2 regression with a test that executes `cut_chunk_remap.py` through the
  real overlay and checks the emitted label array. Added coverage for all five fail-closed cases,
  stale units, last-good publication preservation, fixed-schema overlay behavior, and chunkmap reuse.
- Added durable `zero_repairs` observability to the stage and acceptance report.
- Preserved explicit `CHUNKMAP_INPUT`; it now defaults to `CHUNKMAP_OUTPUT` only when unset.
- Added the full R1--R7 review, efficiency decision rule, watershed-reuse recommendation, fresh
  cluster namespaces, frozen production fingerprints, exact submission/retry commands, expected
  artifacts, and downstream pass/fail criteria.

## Implementation Details

`scan` creates `.nuccomp-runs/<run-id>/units.json` and hashes its exact bytes as `plan_digest`. It
separates and records geometry and watershed-mapping time, fails above `NUC_MAX_UNITS`, audits
overlapping scopes and generated ids, and records `separation_claim: local_only`. Input identity
covers the raw param bytes; nucleus file SHA-256 or store identity plus member-size summary; the
authoritative watershed manifest digest; affinity index digest and chunk count; the canonical ABISS
native/runtime build identity; and a separate digest of Python sources under `lib/abiss/scripts/`.

Each `flood` task verifies the immutable plan and param fingerprint. Indices at or above the actual
unit count exit zero. Active tasks store a uint64 territory array containing preallocated internal
ids, an artifact hash, pooled counts, phase timings, and a record with the exact plan digest. A failed
unit can be rerun alone; completed sibling territories are not touched.

`merge` accepts only complete records with the exact plan digest, verified territory hashes and
metadata, the exact expected territory-id set, and positive counts for every anchor. It computes the
deterministic winner, validates emission semantics, writes the final staging report, and only then
atomically replaces `manifest.json`. A failed scan, flood, or merge cannot remove or overwrite the
previous canonical publication.

The final report uses the required critical-path model with full unit task times, including each
worker's input-open overhead:

```text
T_serial = T_nonflood + sum(t_i)
T_array  = T_nonflood + max(t_i) + scheduler_overhead
```

It also applies the terminal selection rule independently to measured geometry, mapping, and flood
fractions. Scan sharding remains designed and deferred; neither competition semantics nor the C++
cannot-link operator changed.

## Files Changed

| File | Purpose |
|---|---|
| `lib/abiss/scripts/nucleus_competition.py` | Instrumented scan/flood/merge stages, fingerprints, deterministic ids, validation, reports, and transactional publication. |
| `lib/abiss/scripts/nucleus_overlay.py` | Validate completed plan-bound manifests and apply explicit internal-to-emitted translations. |
| `connectomics/runtime/abiss_chunk.py` | Preserve an explicit `CHUNKMAP_INPUT` for supported watershed reuse. |
| `dev/zebrafinch/sbatch_nucleus_competition.sh` | Keep a validated serial-baseline launcher. |
| `dev/zebrafinch/sbatch_nuccomp_scan.sh` | Launch the instrumented full-volume scan. |
| `dev/zebrafinch/sbatch_nuccomp_flood.sh` | Launch one fixed-capacity array element and document exact retry recovery. |
| `dev/zebrafinch/sbatch_nuccomp_merge.sh` | Validate fan-in and publish the canonical manifest. |
| `dev/zebrafinch/submit_wholevol_sharded.sh` | Submit scan, throttled flood array, and merge with explicit dependencies and run id. |
| `dev/zebrafinch/nucleus_acceptance_report.py` | Judge a completed zero-repair intervention as an acceptance failure. |
| `tests/unit/test_abiss_nucleus_competition.py` | Cover schema/translation, real B2 wiring, fail-closed paths, staging, fingerprints, and chunkmap reuse. |
| `docs/nucleus_competition_review.md` | R1--R7 review, efficiency/reuse plan, frozen fingerprints, cluster commands, and gates. |
| `artifacts/code_v0.md` | CCC implementation and verification record. |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

## Verification

Passed:

- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` -- 18 passed.
- `black --check --fast` and `isort --check-only` on all changed Python files -- passed.
- `flake8 --max-line-length=100` on all changed Python files -- passed.
- `mypy --config-file .github/mypy_changed.ini connectomics/runtime/abiss_chunk.py tests/unit/test_abiss_nucleus_competition.py` -- success, no issues.
- `python -m py_compile` on all changed Python files -- passed.
- `bash -n` on the serial, scan, flood, merge, and whole-volume submission scripts -- passed.
- `git diff --check` in both the outer repository and nested `lib/abiss` worktree -- passed.
- `python dev/zebrafinch/nucleus_acceptance_report.py --tol 0.05` -- smoke test completed; existing schema-1.2 production manifests correctly report the new completed-publication judgement as pending.
- `nucleus_competition.py <param> fingerprint` on native96 and win144 -- completed. Frozen values are recorded in `docs/nucleus_competition_review.md`; both stores report 726 affinity chunks, and the nucleus file hash agrees across arms.
- `git rev-parse HEAD` -- still `c705458ae5b907bb9c32a75c63c85c6aad7edec7`.

Pending and not claimed:

- Verification gates 2--8 require fresh cluster runs/materialized volumes and were not executed.
- In particular, efficiency acceptance gate 6 did not pass locally; no 7--11 hour serial or array
  baseline was launched. The report supplies the commands and PASS/FAIL formula for the operator.
- `ccc-validate.sh` could not be run against this handoff because the prompt supplied no CCC run
  folder containing `run.md`, state baselines, and prior artifacts. The `code_v0.md` artifact itself
  follows the required heading, section, baseline, and initial-version contract.

## Review Focus

- Check the schema-2.0 contract across `units.json`, each unit record, the final manifest, and the
  overlay, especially exact id-domain validation and one-parent emission.
- Check the transactional boundary: all expensive or failure-prone validation must occur before the
  final `os.replace` of the canonical manifest.
- Review the critical-path accounting and selection rule using measured cluster reports; no speedup
  is inferred from unit tests.
- Review the real B2 test as the minimum graph-versus-volume regression: only chunk I/O is stubbed;
  the competition overlay and emitted-label translation are real.
- The outer repository ignores `dev/` and `lib/`, and `lib/abiss` is a nested worktree. Reviewers must
  inspect the explicit file table rather than relying only on the outer `git diff`.
- Confirm whether `NUC_MAX_UNITS=64` is acceptable policy; it is approximately seven times the
  observed 8--9 units but remains a configured fail-closed capacity.

## Risks and Unknowns

- R5 remains unresolved: equal-weight edge ordering and different hierarchy/shard layouts may attach
  a permissive `NONE` bridge to different protected owners. The cannot-link algorithm was out of
  scope, so the synthetic ownership oracle remains a ship gate.
- R6 remains unresolved until `z4_y6_x1` is compared with the local scorer and the whole-volume node
  LUT restricted to the same chunk nodes.
- All separation claims are `local_only` until the materialized outside-repair-box equality gate
  passes.
- Cross-arm effect sizes remain qualified because `ABISS_NUC_MIN_TAGGED` is 50 on win144 and 1024 on
  native96; no sweep was authorized or run.
- Hashing an affinity index plus chunk count does not detect an in-place chunk edit that preserves
  both. The already-clean-nucleus gate compares exact report fields, not every voxel of the 58 GB
  volume.
- Schema-1.2 canonical manifests are intentionally rejected by the schema-2.0 overlay. Existing runs
  require a fresh nuccomp publication before downstream agglomeration with this runtime.
- Scan sharding and a first-class watershed-reuse flag remain deferred. Only the chunkmap override
  bug was fixed.
- The inherited plan-v2 file allowlist was not present in the workspace; implementation was limited
  to files explicitly named by plan v3, its launcher additions, its required test/review outputs, and
  this mandated CCC artifact.

## Changes Since Previous Code Version

Initial implementation.