# Code v4

## Overview

Implemented H3, H4, H5, and the three requested test updates. Production now accepts schema 3.0 only; schema 2.0 is explicitly withdrawn and rejected.

No commits or cluster jobs were created. Live publications were untouched.

## What Changed

- Renamed the production publication contract from 2.0 to 3.0.
- Added and validated `required_capabilities: []`.
- Preserved the exact schema-1.2 migration command and schema-1.0 opaque-ID rejection.
- Added a distinct schema-2.0 rejection explaining the withdrawn pre-release parent-ID inversion.
- Improved identity errors to report the differing field, expected value, and found value.
- Added `--watershed-manifest` guidance to watershed-identity mismatches.
- Updated read-only comparison and acceptance consumers to understand 3.0 and reject 2.0.
- Extended version-boundary, migration-invariance, and identity-diagnostic coverage.

## Implementation Details

`PUBLICATION_SCHEMA_VERSION = "3.0"` is the canonical version. Fresh publications, unit plans, unit records, stage reports, migration plans, and migrated manifests use it.

Fresh and migrated manifests emit `required_capabilities: []`. Production loading and publisher-side validation reject missing, malformed, or unsupported capability declarations.

Schema handling is now:

- `3.0`: accepted when the complete contract validates.
- `2.0`: rejected as a withdrawn pre-release contract that inverted the parent-ID rule; no migration command is offered.
- `1.2`: retains the precise no-recomputation migration command.
- `1.0`: retains the opaque/no-lossless-migration rejection.

Identity comparison reports nested paths such as `identity.mint.digest`, including expected and found values. Contract semantics remain unchanged: `residue_disposition` is still `parent_retained`, and adjudicated territories cannot emit their parent ID.

## Files Changed

| File | Purpose |
|---|---|
| `lib/abiss/scripts/nucleus_overlay.py` | Enforce schema 3.0 and capabilities; reject withdrawn 2.0; improve identity and watershed diagnostics. |
| `lib/abiss/scripts/nucleus_competition.py` | Emit schema 3.0 and required capabilities from the fresh publication path. |
| `dev/zebrafinch/migrate_nucleus_competition.py` | Emit and validate schema 3.0 migrations. |
| `dev/zebrafinch/compare_nucleus_competition.py` | Read schema 3.0 correctly and reject ambiguous schema 2.0. |
| `dev/zebrafinch/nucleus_acceptance_report.py` | Evaluate schema-3.0 realization records correctly and reject schema 2.0. |
| `tests/unit/test_abiss_nucleus_competition.py` | Cover version boundaries, required capabilities, migration invariance, identity diagnostics, and H4 guidance. |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7  
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Nested `lib/abiss` HEAD remains `452efa5f87f9d3cb241891ee44010d966a33b316`.

## Verification

Passed:

- `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` under conda environment `pytc` — 28 passed in 4.45 seconds.
- `isort --check-only` on all six changed Python files.
- `flake8 --max-line-length=100` on all six changed Python files.
- `mypy --config-file .github/mypy_changed.ini` on all six changed Python files.
- `python -m py_compile` on all six changed Python files.
- `git -C lib/abiss diff --check` on the two nested-repository files.
- Focused `black --check --fast` on overlay, migration, acceptance, and tests.

The aggregate Black check remains nonzero because it would reformat pre-existing code in `nucleus_competition.py` and `compare_nucleus_competition.py`; those files were not broadly reformatted to avoid out-of-scope churn.

Safety verification:

- No `squeue` command was run; the coordinator supplied the zero-consumer clearance.
- No cluster job was launched.
- All migration tests operated on temporary copies.
- The live native96 manifest remains schema `1.2`.
- Its SHA-256 remains `59bbece64bf6f988d45309a95c146e8386bfbb6ebca9b4554c681ff271ed073e`.
- Parent and nested repository HEADs remain unchanged.

## Review Focus

- Confirm schema 2.0 always reaches the dedicated withdrawn-contract rejection before capability or identity validation.
- Confirm schema 1.2 still includes the exact migration command and schema 1.0 remains opaque.
- Confirm every emitted 3.0 artifact carries `required_capabilities: []`.
- Confirm identity diagnostics identify nested field paths with expected and found values.
- Confirm audit consumers cannot misclassify schema 3.0 as a legacy marker-label contract.
- Confirm no label-generation or production overlay logic changed.

## Risks and Unknowns

- Gates 2–8 and cluster baselines remain unrun and are not claimed.
- Win144 schema-1.0 IDs remain opaque and audit-only.
- The broader worktree remains dirty with pre-existing changes.
- The test module remains untracked in the parent repository, `dev/zebrafinch` is gitignored, and `lib/abiss` is a separate dirty repository.
- Full-file Black compliance remains outstanding for two previously noncompliant files.

## Changes Since Previous Code Version

- H3: resolved with field-specific identity rejection diagnostics.
- H4: resolved by naming `--watershed-manifest` in watershed-identity failures.
- H5: resolved by establishing schema 3.0, withdrawing 2.0, adding required capabilities, and updating migration and audit consumers.
- H2’s existing native96 invariance test now verifies schema 3.0 and `required_capabilities: []`.
- H1 remains unchanged and closed.
- No live publication, published label, decode semantics, or cluster state was changed.

CCC run: `.agent/decoding/abiss_instance/`  
Stage completed: `code_v4`  
Next action: `review_v4`