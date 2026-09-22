# Review v3

## Summary

`code_v3` closed H1 and H2 properly and stopped honestly on H3 and H4.

H1 is done better than I asked. I exercised the guard directly rather than trusting it: with
`PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS` unset the skip condition is `True` and the reason names the
exact missing directory; with it set to `1` the condition is `False`, so the test runs and fails on
the absent data. That is precisely the two-sided behaviour Q-2 asked for — a third party skips, and
our own CI can be made to fail rather than quietly under-cover. H2 adds
`test_native96_realization_is_invariant_across_migration`, pinning the property I had only measured
by hand. The suite is at 27.

H3 and H4 were not implemented, and the reason is infrastructure rather than coder judgement:
`squeue -u weidf` failed twice inside the codex sandbox with
`Error creating slurm stream socket: Operation not permitted`, and my own `code_v3` prompt made a
successful queue check a precondition for editing `lib/abiss/scripts/*.py`. The coder stopped
instead of editing blind, which is the constraint working exactly as intended — the same discipline
whose absence cost 840 node-hours in G1. Worth noting the check *succeeded* during `code_v2`, so
the sandbox's access to the Slurm socket is intermittent, not permanently unavailable.

One of those two blocks was my mistake: **H4 touches only
`dev/zebrafinch/migrate_nucleus_competition.py`, which is not under `lib/abiss/scripts/`**, so the
precondition should never have gated it. H3 does touch `nucleus_overlay.py` and was correctly
gated. I have since run the check myself — 0 live consumers, as it was before both prior launches,
and nothing has been submitted since — so `code_v4` can be given that result as a stated fact
instead of a command it cannot run.

This is **NEEDS_CHANGES**, carrying H3 and H4 forward and adding **H5**, which is the more
consequential of the three and came out of the operator's question about the final schema version.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Parent `HEAD` c705458a and nested `lib/abiss` `HEAD` 452efa5f unmoved. `code_v3` edited no file
under `lib/abiss/scripts/`, and the live native96 manifest is still schema 1.2 with the SHA-256 it
records. Evidence in `state/review_v3.review.raw.md`.

## Findings

- **[major] H5 — `schema_version: "2.0"` denotes two mutually incompatible contracts.** The
  `code_v0`/`code_v1` form of 2.0 had no `identity` block, no `ledger`, and *required*
  `emitted.count(parent_id) == 1`. The `code_v2` form of 2.0 *requires* `identity` and `ledger` and
  *forbids* a territory emitting its parent id. An artifact valid under either is rejected by the
  other, under the same version string, with the governing rule inverted. The hard-reject version
  policy this run adopted is unenforceable if the number does not move when the contract does — the
  exact failure the policy exists to prevent, one layer up.
  **Use 3.0**, which is what the decision document names in three places. `2.1` would be a lie,
  because a MINOR must be additive and this inverted a rule. The deferred surface
  (`inputs[].semantics`, capability negotiation, `native_id_space`, `id_encoding`) is additive and
  lands as 3.1+, so the shipped subset stays policy-consistent.
  **The fix is free exactly now**: I surveyed every nucleus manifest on disk — five publications,
  three at 1.2 and two at 1.0, and **no 2.0 artifact anywhere**. That stops being true the moment
  native96 is migrated or a fresh nuccomp publishes. `2.0` should become a recognized-but-rejected
  version whose message says it was a pre-release contract that inverted the parent-id rule, so the
  window this run created cannot silently mislead anyone who built against it.
  Also add **`required_capabilities: []` to 3.0 itself** — the policy lets a consumer accept a newer
  minor only if it implements every listed capability, which is unusable unless the base version
  carries the field.
- **[major] H3 — carried forward, unchanged.** `validate_publication_identity` compares the whole
  identity dict for exact equality and raises "lacks the supported identity declaration" without
  naming the disagreeing field. Severity is raised from minor to major by H5: once `2.0` and `3.0`
  both exist in the world, an unhelpful identity rejection is how a third party fails to tell a
  policy rejection from a corrupt artifact. Name the field, the expected value and the found value.
- **[minor] H4 — carried forward, unchanged.** The migration infers the watershed manifest from the
  legacy `base_watershed`, while the overlay derives it from `global_params`; a disagreement
  surfaces only at job time as "built from another watershed identity". Name `--watershed-manifest`
  in that error. This was never gated by the queue precondition and should have been done.

Recorded so `code_v4` does not redo them: H1 and H2 are closed and independently verified; the skip
reason names the exact path; the suite is at 27 passing; no `lib/abiss/scripts/` file was touched by
`code_v3`; live publications untouched.

## Tests to Add

1. **Version-boundary coverage** (H5): a 3.0 manifest is accepted; a 2.0 manifest — including one
   carrying a valid `identity` block — is rejected with a message naming 2.0 as a withdrawn
   pre-release contract; a 1.2 manifest still yields the exact migrate command.
2. **Migration emits 3.0** (H5): migrating a 1.2 copy produces `schema_version: "3.0"` with
   `required_capabilities: []` present, and the realization gate remains invariant across it — the
   H2 test should be extended rather than duplicated.
3. **Identity rejection names the field** (H3): an identity differing in exactly one field produces
   an error naming that field with expected and found values.

## Questions

- **Q-1.** Confirm `3.0` as the final version and `2.0` as withdrawn-and-rejected. This is the last
  moment it costs nothing.
- **Q-2.** `code_v4` needs `c4`. Scope is H3, H4, H5 and the three tests above — no production
  decode logic, no change to what the contract *means*, only what it is *called* and how it
  rejects. Confirm.
- **Q-3.** Unchanged and still open: gates 2–8 and the cluster baselines are unrun; win144's 1.0
  ids remain opaque, so that arm stays audit-only.

## Verdict

VERDICT: NEEDS_CHANGES
