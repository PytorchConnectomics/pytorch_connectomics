# Review v2

## Summary

`code_v2` is the strongest artifact this run has produced. Every one of `review_v1`'s G1–G5 is
implemented, and — the part that matters most — the identity contract is **verified rather than
asserted**. `validate_publication_identity` recomputes each emitted id from the declared mint,
enforces the nucleus↔emitted bijection both ways, forbids an adjudicated territory from emitting
its parent id, and rebuilds the ledger and compares it. That closes the exact hole G2 named: a
migration cannot stamp a declaration the data does not satisfy. I checked the ledger arithmetic
against the live manifest independently and it lands on the measured numbers — 85 qualified
segments = 77 single-owner + 8 multi-owner, 15 consolidated owner labels, maximum fan-in 7.

The migration tool holds up to reading as well as to running: atomic write, a backup that refuses
to be overwritten with different content, explicit refusal of schema 1.0 as opaque, territory
dtype/bbox/factor and both marker domains cross-checked against the manifest, and a second run on
an already-migrated manifest failing loudly instead of double-migrating. Exercised on a copy, it
leaves the `.npz` bytes identical, finishes in under a second, and turns a manifest production
rejects into one production accepts.

It is still **NEEDS_CHANGES**, but for verification gaps rather than defects — I found no
correctness fault in the shipped logic. Both majors are small, and `c3` leaves room for a tightly
scoped `code_v3`.

The first is squarely against this round's stated purpose. You asked for something robust for
other people's deployments; three of the 26 tests hard-depend on gitignored, machine-local
experiment directories with no `skipif`, so a third party running the suite gets **errors, not
skips**. The second is a gap around the very step you are about to take: nothing pins that the
realization gate answers the same before and after migration. I measured that it does — 8/8 units
and 15/15 owners on both sides — so this is a missing test, not a live bug, but it is the property
the `b3_gate025` relaunch rests on.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Parent `HEAD` c705458a and nested `lib/abiss` `HEAD` 452efa5f are both unmoved, and the live
native96 publication was still schema 1.2 when this review finished — every experiment in it ran
against copies under `/tmp`, which were removed. The surface caveat from `review_v0` and
`review_v1` still applies: `lib/abiss` is a nested repository, `dev/` is gitignored, and the test
file is untracked, so this review was performed by reading files and running commands rather than
from a parent-repo diff. Evidence is in `state/review_v2.review.raw.md`.

## Findings

- **[major] H1 — Three tests error rather than skip without local experiment data.**
  `dev/` is gitignored (`.gitignore:159`) and `git ls-files` returns 0 tracked files under
  `wholevol_arm0_native96_nuc_matchguard`, so the directory is absent from a fresh clone.
  `test_realization_gate_reproduces_the_two_published_run_oracles` calls `realization_gate` on both
  live run dirs directly, and the `migrated_native96_manifest` fixture does an unguarded
  `shutil.copytree(NATIVE96_PUBLICATION, ...)`, which two further tests depend on. Guard all three
  with `pytest.mark.skipif(not <path>.is_dir(), ...)` naming what is missing and why. Keep the
  live-data assertions — they are the strongest evidence in the suite — but let them skip. Nothing
  reads disk at import time, so collection itself is safe; only these three tests are affected.
- **[major] H2 — Nothing pins realization-invariance across migration.** The fixture migrates into
  `tmp_path` but never runs the realization gate on the result, and the realization test runs only
  against the un-migrated live runs. Since migrating the live publication is the immediate next
  operational step, add a test that runs the gate before and after migrating a copy and asserts
  the answers are identical. I measured this today and it holds — units (8,8) and owners (15,15)
  on both sides, so `_repair_owner_labels`' 2.0 branch is equivalent to its legacy branch on real
  data — but nothing prevents a later change from breaking it silently.
- **[minor] H3 — The identity declaration is presently decorative, and its rejection is unhelpful.**
  `validate_publication_identity:120` tests the whole identity dict for exact equality against the
  single supported declaration and raises "lacks the supported identity declaration" without naming
  the disagreeing field. A publication declaring `residue_disposition: "minted"` — a legal value
  the design contemplates and `review_v1` Q-2 weighed — is rejected wholesale. That is defensible
  while capability negotiation is deferred, but the message must name field, expected and found,
  or a third party cannot tell a rejected-by-policy artifact from a corrupt one. Field-level
  handling becomes mandatory the day residue minting is adopted.
- **[minor] H4 — Two different watershed-resolution paths.** The migration infers the watershed
  manifest from the legacy manifest's `base_watershed`, while the overlay derives it from
  `global_params`. If they disagree, migration bakes in a fingerprint that fails only later, at job
  time, as "built from another watershed identity". `--watershed-manifest` is the escape hatch;
  name it in that error message.

Verified positively, so `code_v3` does not redo them: G2's rule is gone from both former sites;
G4's ordering is correct (`canonicalize` at `:446` now precedes `if not repairs:` at `:447`); the
ledger records all 85 retirements and 15 consolidations with honest
`scope: "adjudicated_voxels"` and `residue_disposition: "parent_retained"`, which is the truthful
description given that the parent id survives outside the adjudicated box; the realization gate
handles both schemas; and the suite is at 26 passing. Codex was also accurate about its own
limits, including declining to claim a `black --check` that exited 124 on a timeout.

## Tests to Add

1. **Skip guards** for the three live-data tests (H1), asserting the skip reason names the missing
   directory.
2. **Realization-invariance across migration** (H2): gate a copy, migrate it, gate it again,
   assert equality of `realized_unit_count`, `unit_count`, `realized_owner_count`, `owner_count`.
3. **A rejection-message test** (H3): an identity block differing in exactly one field produces an
   error naming that field with expected and found values.

## Questions

- **Q-1.** Confirm `code_v3` is scoped to H1–H4 only — four small, local changes with no
  production-logic edit. That keeps the terminal round reviewable and leaves the shipped behaviour
  exactly where `code_v2` put it.
- **Q-2.** H1's guards mean a third-party CI run silently covers less. Should the suite instead
  fail loudly when the data is absent but an explicit `PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS=1` is set,
  so our own CI cannot quietly stop exercising the strongest tests?
- **Q-3.** Unchanged from `review_v1` and still open: gates 2–8 and the cluster baselines remain
  unrun and unclaimed, and win144's schema-1.0 ids stay opaque with no lossless migration, so that
  arm remains audit-only and barred as a comparison reference.

## Verdict

VERDICT: NEEDS_CHANGES
