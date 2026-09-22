# Code v2

## Overview

Implemented the terminal corrections for review_v1 findings G1–G5 and requested tests 1–5.

Schema-1.2 publications now have an atomic, no-recomputation migration path. Fresh and migrated publications declare nucleus-scoped identity, retain unadjudicated parent residue, and include a complete retirement/consolidation ledger. Zero-repair overlays still canonicalize, and a published-volume realization gate reproduces win144 0/9 versus native96 8/8 units and 15/15 owners.

No publication was migrated, no volume labels were changed, no cluster job was launched, and no commit was created.

## What Changed

- Added an in-place schema-1.2 migration that preserves the original manifest, writes a plan sidecar, leaves legacy `int32` marker-index territories byte-identical, and atomically publishes schema 2.0.
- Changed legacy rejection to print the exact migration command instead of requiring scan/flood/merge recomputation.
- Deleted the duplicated largest-territory/parent-ID convention.
- Added a canonical identity declaration with nucleus scope, retired adjudicated parents, parent-retained residue, and an explicit deterministic mint descriptor.
- Fresh publications now emit the same nucleus-owner label across repaired units and canonicalized segments.
- Added a publication ledger covering competitive parents, every canonicalized single-owner source, consolidation source lists, and the emitted-ID space.
- Removed the zero-repair overlay short circuit and added a populated publication reason.
- Added a realization gate using each run’s own `nucleus_shell_contamination_tol0.json`.
- Corrected the refinement-only claims in the task and lesson documentation.
- Added focused regressions for migration, nucleus-scoped minting, ledger closure, zero-repair integrity, retained residue, and both actual realization oracles.

## Implementation Details

The migration command is:

```bash
python dev/zebrafinch/migrate_nucleus_competition.py \
  --manifest /absolute/path/to/nucleus_competition/manifest.json
```

It supports schema 1.2, infers or accepts the authoritative watershed manifest, verifies every marker-index territory and nucleus mapping, preserves the original as a digest-named backup, and replaces only `manifest.json`. The existing `b3_gate025` parameter file points to this shared manifest and has no fixed plan digest, so migration enables relaunch without repeating nucleus competition.

Identity validation has one canonical owner in `nucleus_overlay.py`. It derives emitted IDs from the declared key template, checks one-to-one nucleus/ID scope across repair mappings and canonicalization, and validates the ledger against the publication contents. It does not check parent-ID absence: marker-free or out-of-box residue retains the parent label by policy.

The measured native96 migration fixture records eight competitively retired parents, 77 canonicalized sources, and 15 consolidations, each with multiple sources and a maximum 7:1 consolidation. Nucleus 373 receives one emitted ID under two different parents.

The realization gate requires each unit’s dominant published segments to be distinct, exceed the declared dominance threshold, and equal the corresponding emitted labels. It reads only the per-run audit derived from nucleus-mask voxels.

## Files Changed

| File | Purpose |
|---|---|
| `lib/abiss/scripts/nucleus_competition.py` | Emits nucleus-scoped labels, publishes identity/ledger/reason metadata, and removes the parent-winner rule. |
| `lib/abiss/scripts/nucleus_overlay.py` | Canonical identity and ledger validation, actionable migration rejection, and zero-repair canonicalization fix. |
| `dev/zebrafinch/migrate_nucleus_competition.py` | New atomic schema-1.2 no-recomputation migration command. |
| `dev/zebrafinch/nucleus_acceptance_report.py` | Published-volume per-unit and per-owner realization gate. |
| `tests/unit/test_abiss_nucleus_competition.py` | G1–G5 regressions and actual win144/native96 oracles. |
| `dev/zebrafinch/lesson_nucleus_competition.md` | Corrected identity, consolidation, residue, and safety documentation. |
| `docs/nucleus_competition_review.md` | Migration/relaunch procedure, realization results, ledger contract, and remaining gate status. |
| `.agent/decoding/abiss_instance/task.md` | Corrected the false parent-winner and refinement-only premises in §1. |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7  
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

The nested `lib/abiss` HEAD also remained unchanged at `452efa5f87f9d3cb241891ee44010d966a33b316`.

## Verification

Ran and passed:

- `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` — 26 passed in 5.65 seconds.
- Actual realization oracle — win144 0/9 units; native96 8/8 units and 15/15 owners.
- `isort --check-only` and `flake8 --max-line-length=100` on all changed Python files.
- `mypy --config-file .github/mypy_changed.ini` on the migration tool, acceptance reporter, and focused tests — no issues.
- `python -m py_compile` on both ABISS scripts, migration/comparison tools, acceptance reporter, and tests.
- Parent and nested `git diff --check`, plus trailing-whitespace checks.
- Parent and nested HEAD verification.
- `squeue -u weidf` before shared-script edits and again before formatting; no `me_*`, `remapagg`, or `nuccomp` consumer was running. One later scheduler query transiently failed, and its immediate retry was clear before further edits.

Black mechanically reformatted four files. A subsequent bounded `black --check` reported all five files unchanged, but the process did not terminate before the 180-second timeout and exited 124. This is not claimed as a passing Black command.

Not run or claimed:

- Full gates 2–8, including cluster label equivalence, shard invariance, failure injection, efficiency, retry recovery, outside-box equality, clean-nucleus equality, and `nuc_cuts.data` equality.
- The 7–11 hour cluster baselines or any speedup measurement.
- The authoritative migration or `b3_gate025` relaunch.
- CCC validation, because this artifact is intentionally supplied only as the final message and has not yet been landed by the coordinator.

## Review Focus

- Confirm schema-1.2 migration changes only publication metadata and sidecars, never territory arrays.
- Check that production rejection names the exact migration command.
- Check nucleus 373’s ID consistency across parents and canonicalization.
- Confirm ledger closure covers both eight competitive parents and all 77 canonicalized sources.
- Confirm parent-retained residue remains unchanged and no parent-absence gate was introduced.
- Check zero-repair execution still runs ownership filtering and canonicalization.
- Treat realization as a materialized-volume subgate, not completion of the remaining downstream gates.

## Risks and Unknowns

- Full schema-3.0 `inputs[].semantics` provenance, required-capability negotiation, and measured native-ID-space scanning remain deferred to a follow-up run.
- Schema-1.0 win144 IDs remain opaque legacy values and have no lossless production migration.
- The separate review_v1 decimal-string audit at every repository write site was outside the binding G1–G5/test-1–5 scope. New emitted identity and ledger IDs are strings, but no repository-wide audit was added.
- The realization gate consumes existing per-run `tol=0` audits; it did not rescan the raw published volumes.
- Canonicalization still intentionally consolidates names. A faulty nucleus mask spanning distinct cells can silently fuse them; the ledger exposes but does not eliminate this risk.
- R5 ownership ordering, R6 local-versus-whole-volume scoring, and the confounded cross-arm attribution remain unresolved.
- The worktree contains extensive pre-existing unrelated modifications and experiment files. None were cleaned, staged, or reverted.

## Changes Since Previous Code Version

- G1: Added and tested the schema-1.2 no-recomputation migration, exact rejection command, original-manifest backup, atomic publication, and unchanged territory hashes.
- G2: Replaced both parent-winner validators with one declared nucleus-identity contract; pinned nucleus 373 across two parents.
- G3: Corrected the refinement claim and added ledger closure for eight split parents, 77 canonicalized sources, and 15 consolidations.
- G4: Removed the zero-repair short circuit; canonicalization and sibling processing now run and the publication records a reason.
- G5: Added the published-volume realization gate and reproduced 0/9 win144 versus 8/8 units and 15/15 owners on native96.
- Test 1: Added the two actual realization oracles.
- Test 2: Added legacy rejection/migration/production-acceptance coverage without territory recomputation.
- Test 3: Added declared-mint recomputation and nucleus/ID equivalence coverage across units and canonicalization.
- Test 4: Added measured ledger-closure and emitted-ID-space coverage.
- Test 5: Added zero-repair canonicalization, sibling-stage, completion, and reason coverage.