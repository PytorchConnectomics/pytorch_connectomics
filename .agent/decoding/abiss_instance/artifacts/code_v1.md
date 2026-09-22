# Code v1

## Overview

Revised code_v0 to make the legacy regression oracle executable without weakening the production
schema boundary. A new read-only comparison harness loads the completed schema-1.2 native96 and
schema-1.0 win144 publications, reproduces their legacy emitted labels, and compares them with a
completed schema-2.0 candidate for verification gates 2 and 3. Production `nucleus_overlay.py`
remains strict and now rejects legacy manifests with an actionable republication message.

The coordinator's scope decisions are implemented: the two-line `CHUNKMAP_INPUT` repair is kept as
an accepted scope amendment, the scan and merge launchers are retained as natural consequences of
the three-operation split, and `NUC_MAX_UNITS=64` is confirmed policy with observed/capacity
logging and fail-closed test coverage. The documentation now states unambiguously that gates 2--8
are unrun and the refactor is not yet validated or cleared to ship.

No commit was created and HEAD did not move. The focused suite passes with 22 tests. The 7--11 hour
cluster baselines were not run; verification gate 6 is pending and is not claimed.

## What Changed

- Added a comparison-tooling-only reader for nucleus-competition schemas 1.0, 1.2, and completed
  2.0. Legacy reads use `marker_labels` over the immutable marker-valued territory arrays; schema
  1.2 also replays its historical single-owner canonicalization.
- Added executable `labels` and `manifests` comparison commands. Gate 2 streams the authoritative
  watershed over the frozen repair boxes and requires byte-identical emitted arrays. Gate 3
  normalizes only the id representation and compares parent ids, boxes, pooling factors, anchors,
  pooled voxel counts, and bridge records.
- Kept production overlay compatibility closed. Schema 1.0/1.2 now fail with a message directing
  the operator to republish schema 2.0 and naming the read-only harness as the sole legacy route.
- Added a compact fixture derived from actual completed win144 and native96 territory boundaries,
  including source manifest/territory hashes and stored emitted-label expectations.
- Added the three requested tests: the miniature real-run gate-2 oracle, explicit legacy production
  rejection plus harness replay, and fail-closed capacity behavior with no truncated `units.json`.
- Confirmed the default capacity at 64. Scan prints `observed/configured`, records capacity in scan
  and stage reports, and aborts before planning when observed units exceed it.
- Rewrote the ship-status language and supplied exact reference paths, expected unit counts, gate
  commands, fresh-run submission commands, and failed-index retry commands.
- Explicitly recorded the accepted `abiss_chunk.py` scope amendment and the undeclared-but-required
  scan/merge launchers.

## Implementation Details

`dev/zebrafinch/compare_nucleus_competition.py` is independent test tooling and is never imported by
ABISS production execution. Its legacy reader accepts only manifest types
`abiss_nucleus_competition` with schema 1.0 or 1.2. It validates territory containment and stored
box/factor metadata, requires the marker-mapping domain to equal the exact nonzero territory-id
set, and applies `marker_labels` exactly. For schema 1.2 it canonicalizes globally single-owner
segments before applying competitive territories, matching the historical label overlay. Reads are
cached by immutable file identity for bounded-memory streaming across repair boxes; no reference
file is written or upgraded.

The same harness accepts schema 2.0 only when the completion marker agrees with `plan_digest`, the
declared `units.json` hashes to that digest, and each territory matches its manifest SHA-256. Its
gate-2 action opens the watershed named by the frozen reference parameter file, rejects a candidate
with another `base_watershed`, applies each manifest independently in bounded XYZ blocks, and stops
at the first differing emitted array. Its gate-3 action removes internal-id representation only;
geometry, parent/anchor membership, voxel counts, and bridge semantics remain exact or the command
fails.

The fixture stores two 8x4x4 emitted-label oracles sampled across real marker boundaries: win144
parent `72199226020331523`, anchors `[611, 651]`, schema 1.0; and native96 parent
`72128993574256642`, anchors `[22, 37]`, schema 1.2. Tests reconstruct only those miniature
territory archives and compare the complete emitted arrays with stored expectations. Separate tests
pin production rejection for both legacy schema versions.

`scan_stage` now logs, for example, `array capacity 8/64 units observed/configured`. The capacity
check remains before `_prepare_units` and `units.json` publication. The regression forces three
units against capacity two, asserts the exception, checks the log, and proves that `units.json` was
not created. Capacity is also included as `unit_capacity` in both scan and merged stage reports.

The full reference-vs-candidate commands are in `docs/nucleus_competition_review.md`. The harness
was smoke-tested read-only against each completed reference itself, including one 128-voxel
territory-boundary cutout per schema. Those self-checks prove the legacy routes execute; they do not
constitute gates 2 or 3 against a refactored candidate.

## Files Changed

| File | Purpose |
|---|---|
| `lib/abiss/scripts/nucleus_competition.py` | Instrumented scan/flood/merge implementation; v1 adds confirmed-capacity logging/reporting while preserving fail-closed behavior. |
| `lib/abiss/scripts/nucleus_overlay.py` | Strict schema-2.0 production overlay; v1 adds explicit actionable rejection of pre-2.0 publications. |
| `dev/zebrafinch/compare_nucleus_competition.py` | Read-only schema-1.0/1.2 legacy reader and executable gate-2/gate-3 comparison harness. |
| `tests/fixtures/abiss_nucleus_competition/legacy_emitted_labels.json` | Miniature emitted-label oracles derived from the completed win144 and native96 references. |
| `tests/unit/test_abiss_nucleus_competition.py` | Covers legacy replay, strict production rejection, schema-2.0 semantic reading, and capacity fail-closed behavior in addition to code_v0 contracts. |
| `docs/nucleus_competition_review.md` | Plain unvalidated ship verdict, scope amendments, exact comparison/cluster/retry commands, and gate status. |
| `connectomics/runtime/abiss_chunk.py` | Accepted two-line scope amendment: preserve explicit `CHUNKMAP_INPUT`, otherwise default it to `CHUNKMAP_OUTPUT`; removes the 126 MB copy workaround. |
| `dev/zebrafinch/sbatch_nucleus_competition.sh` | Validated serial-baseline launcher retained from code_v0. |
| `dev/zebrafinch/sbatch_nuccomp_scan.sh` | Scan launcher; undeclared in plan v3 but accepted as required by the three-operation split. |
| `dev/zebrafinch/sbatch_nuccomp_flood.sh` | Fixed-capacity flood-array launcher with exact failed-index retry instructions. |
| `dev/zebrafinch/sbatch_nuccomp_merge.sh` | Merge/publication launcher; undeclared in plan v3 but accepted as required by the three-operation split. |
| `dev/zebrafinch/submit_wholevol_sharded.sh` | Dependency chain for scan, 64-element throttled flood array, merge, and downstream agglomeration. |
| `dev/zebrafinch/nucleus_acceptance_report.py` | Durable `zero_repairs` acceptance judgement from code_v0. |
| `artifacts/code_v0.md` | Previous CCC implementation record reviewed by the planner. |
| `artifacts/code_v1.md` | This revised CCC implementation and verification record. |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

## Verification

Ran and passed:

- `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` -- 22 passed in 2.19 seconds.
- `black --check --fast`, `isort --check-only`, and `flake8 --max-line-length=100` on
  `nucleus_competition.py`, `nucleus_overlay.py`, the comparison harness, and the focused test file
  -- passed.
- `mypy --config-file .github/mypy_changed.ini` on the comparison harness, focused test file, and
  `connectomics/runtime/abiss_chunk.py` -- passed with no issues.
- `python -m py_compile` on the two ABISS Python scripts, comparison harness, focused test file, and
  acceptance reporter; `python -m json.tool` on the legacy fixture -- passed.
- `bash -n` on the serial, scan, flood, merge, and whole-volume submission scripts -- passed.
- `git diff --check` in the outer repository and nested `lib/abiss` worktree -- passed.
- `git rev-parse HEAD` -- exactly `c705458ae5b907bb9c32a75c63c85c6aad7edec7`.
- Read-only actual-reference harness smoke checks:
  - native96 schema 1.2 semantic self-comparison: 8 units, 0 bridges; 128-voxel emitted-label
    self-comparison SHA-256 `4aba8a2110e130dd2b72416ad78411873c232ad9c21234f2c89487d25d810051`.
  - win144 schema 1.0 semantic self-comparison: 9 units, 0 bridges, expected
    `72199226020331523:[611,651]` present; 128-voxel emitted-label self-comparison SHA-256
    `817cf47a121eced493dd2124fbd731e21870e66f8a9954eff71c1c3724f54fb6`.

Pending and not claimed:

- Gates 2--8 remain unrun against fresh schema-2.0 candidates. The harness makes gates 2 and 3
  executable, but self-comparing each legacy reference is only a reader smoke check, not evidence
  that the refactor is behavior-preserving.
- In particular, the 7--11 hour serial/array cluster baselines were not launched and efficiency
  acceptance gate 6 did not pass locally. No speedup or production critical-path claim is made.
- Shard-count invariance, all five production failure injections, failed-unit-only retry recovery,
  outside-repair-box equality, clean-nucleus report equality, and `nuc_cuts.data` equality remain
  operator work under the documented commands.
- `ccc-validate.sh` was not rerun by this coder because the prompt did not identify a current CCC
  run folder. Review v0 records that the coordinator validated its run successfully; validation of
  this new artifact remains a coordinator responsibility.

## Review Focus

- Review the strict boundary: no production code imports the legacy reader, and production rejects
  both schema 1.0 and 1.2 before watershed or territory consumption.
- Check legacy fidelity against the actual references: schema 1.0 marker-to-emitted mapping and
  schema 1.2 marker mapping plus single-owner canonicalization must remain literal, with no inferred
  largest-territory winner.
- Check that gate 3 removes only internal-id representation; parent, repair geometry, factor,
  anchor set, pooled counts, and bridge records must compare exactly.
- Run the documented reference-vs-candidate gate-2/3 commands after each fresh cluster publication.
  A command failure is a regression finding, not permission to relax the legacy oracle.
- Confirm the capacity check occurs before unit preparation/publication, reports observed/64
  headroom, and cannot truncate a unit list.
- Treat `connectomics/runtime/abiss_chunk.py`, `sbatch_nuccomp_scan.sh`, and
  `sbatch_nuccomp_merge.sh` as explicit accepted scope amendments rather than hidden plan drift.

## Risks and Unknowns

- The newly executable oracle may reveal emitted-label differences. Schema 1.2 historically emits
  owner-stable labels for both competing territories, while schema 2.0 records an explicit winner
  mapping. No candidate cluster artifact exists locally, so compatibility is deliberately not
  inferred; any gate-2 mismatch must be reviewed before shipment.
- R5 ownership ordering and R6 local-versus-whole-volume scoring remain unresolved ship blockers.
- Every separation claim remains `local_only` until the materialized gate-8 comparison passes.
- Cross-arm results remain confounded by different `ABISS_NUC_MIN_TAGGED` values; no sweep was
  authorized or run.
- Affinity identity remains index SHA-256 plus chunk count and can miss an in-place chunk rewrite
  that preserves both. The clean-nucleus gate compares exact report fields rather than all 58 GB of
  voxels.
- `NUC_MAX_UNITS=64` is now policy, not an estimate. A dataset with 65 or more units will abort and
  require an explicit policy/configuration change; it will never silently truncate.
- Scan sharding and a first-class watershed-reuse flag remain deferred. Only the accepted
  `CHUNKMAP_INPUT` defaulting bug is fixed.
- The worktree contains extensive pre-existing unrelated modifications and ignored/untracked
  experiment files. They were preserved; no attempt was made to clean, stage, or attribute them to
  this CCC increment.

## Changes Since Previous Code Version

- **F1 -- fixed.** Added the comparison-harness-only legacy reader. It loads the existing native96
  schema-1.2 and win144 schema-1.0 manifests read-only, replays their actual legacy emission rules,
  validates completed schema-2.0 candidates, and supplies exact gate-2 and gate-3 commands.
  Production overlay compatibility was not added.
- **F2 -- resolved by coordinator decision.** Kept the two-line
  `connectomics/runtime/abiss_chunk.py` `CHUNKMAP_INPUT` change. It is listed explicitly as outside
  plan v3's declared files, accepted by the planner as a scope amendment, and retained because it
  removes the 126 MB copy workaround without overriding an explicit input.
- **F3 -- fixed.** `docs/nucleus_competition_review.md` now says plainly at the top and in the gate
  table that gates 2--8 are unrun, the refactor is not validated as behavior-preserving, gate 6 has
  not passed, and shipment is blocked pending cluster evidence and R5/R6.
- **F4 -- resolved and documented.** Kept `sbatch_nuccomp_scan.sh` and
  `sbatch_nuccomp_merge.sh`; both are identified as undeclared in plan v3 but naturally required by
  the accepted scan/flood/merge split.
- **F5 -- no code defect to fix.** The prior missing `ccc-validate.sh` invocation was a coordinator
  handoff omission, and review v0 records the coordinator's run as green. This coder did not invent
  or validate against an unspecified run folder.
- **F6 -- fixed by policy decision.** Confirmed `NUC_MAX_UNITS=64`, added observed/capacity logging
  and report fields, and added a regression proving scan aborts above capacity without publishing a
  truncated `units.json`.
- **Requested test 1 -- added.** The miniature gate-2 fixture is sampled from real completed
  win144 and native96 territory boundaries and pins their complete emitted arrays.
- **Requested test 2 -- added.** Both pre-2.0 schemas are explicitly rejected by production with
  republication guidance, while the harness reproduces their stored legacy expectations.
- **Requested test 3 -- added.** A three-unit scan against capacity two fails closed, logs `3/2`,
  confirms the default policy is 64, and writes no unit plan.