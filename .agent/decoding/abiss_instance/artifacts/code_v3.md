# Code v3

## Overview

Partially implemented H1–H4. H1 and H2 are complete and verified. H3 and H4 remain blocked because the required `squeue -u weidf` safety check could not access Slurm; no `lib/abiss/scripts/*.py` file was edited.

No commits were created. Production decode behavior, identity behavior, and live publications remain unchanged.

## What Changed

- Added skip guards for tests requiring gitignored zebrafinch artifacts.
- Added `PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS=1` support, making missing expected artifacts fatal.
- Added a native96 test proving realization counts are invariant across schema-1.2-to-2.0 migration.
- Performed all migration testing on temporary copies.
- Did not implement H3 or H4 because queue safety could not be confirmed.

## Implementation Details

The artifact guards identify missing directories at collection time and include their exact absolute paths in skip reasons. With the opt-in environment variable set, skips are disabled and the fixture/test assertions fail explicitly on missing data.

The migration-invariance test copies the native96 publication and its `tol=0` audit into `tmp_path`, measures realization before migration, migrates only the copy, measures again, and compares:

- `realized_unit_count`
- `unit_count`
- `realized_owner_count`
- `owner_count`

`residue_disposition` remains `parent_retained`; no labels or live publication files were changed.

## Files Changed

| File | Purpose |
|---|---|
| `tests/unit/test_abiss_nucleus_competition.py` | Add H1 artifact guards, opt-in fatal behavior, and the H2 realization-invariance test. |

## Git Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7  
current_head: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Nested `lib/abiss` HEAD remains `452efa5f87f9d3cb241891ee44010d966a33b316`.

## Verification

Passed:

- `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` — 27 passed in 5.30 seconds.
- Fresh-clone simulation from `/tmp` with artifacts absent — 4 skipped; every reason named the exact missing directory.
- Same simulation with `PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS=1` — exited nonzero with an assertion naming the exact required directory.
- `black --check --fast tests/unit/test_abiss_nucleus_competition.py`
- `isort --check-only tests/unit/test_abiss_nucleus_competition.py`
- `flake8 --max-line-length=100 tests/unit/test_abiss_nucleus_competition.py`
- `mypy --config-file .github/mypy_changed.ini tests/unit/test_abiss_nucleus_competition.py`
- `python -m py_compile tests/unit/test_abiss_nucleus_competition.py`

Safety checks:

- `squeue -u weidf` was attempted twice but repeatedly failed with `Error creating slurm stream socket: Operation not permitted`.
- Consequently, no `lib/abiss/scripts/*.py` edit was made.
- The live native96 manifest remained schema `1.2`, with unchanged SHA-256 `59bbece64bf6f988d45309a95c146e8386bfbb6ebca9b4554c681ff271ed073e`.
- Parent and nested repository HEADs remained unchanged.

Not run or claimed:

- H3 rejection-message test, because its production change was blocked.
- H4 watershed-mismatch message change or test.
- Gates 2–8, cluster baselines, relaunches, or publication migration.

## Review Focus

- Confirm missing artifact tests skip by default and become fatal only when `PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS=1`.
- Confirm skip reasons include the exact absent directories.
- Confirm the H2 test migrates only a `tmp_path` copy and compares all four realization counts.
- Treat H3 and H4 as unresolved findings, not completed work.

## Risks and Unknowns

- H3 and H4 remain unresolved until Slurm can be queried successfully before editing `lib/abiss/scripts/nucleus_overlay.py`.
- Queue state is unknown; editing the shared scripts without confirmation would violate the operator safety constraint.
- The sandbox refused removal of `/tmp/pytc-ccc-code-v3.XlMJ4u`; it contains only a copied test module.
- The broader worktree remains extensively dirty with pre-existing changes.
- No CCC artifact or `.done` file was written into the repository.

## Changes Since Previous Code Version

- H1: Added exact-path skip guards and the coordinator-mandated opt-in fatal mode.
- H2: Added and passed realization-invariance coverage across native96 migration.
- H3: Not implemented because the required queue confirmation was unavailable.
- H4: Not implemented for the same reason.

CCC run: `.agent/decoding/abiss_instance/`  
Stage completed: none  
Next action: blocked