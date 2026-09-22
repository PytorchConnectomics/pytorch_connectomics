# Review v4

## Summary

`code_v4` resolves H3, H4 and H5, and I verified each by exercising it rather than by reading the
claim. The suite is at 28.

The version work is right on every rung. `PUBLICATION_SCHEMA_VERSION = "3.0"`; a 3.0 publication is
accepted; **2.0 is rejected as "a withdrawn pre-release contract that inverted the parent-id rule"
and — the detail that matters — is not offered the 1.2 migrate command**, because a 2.0 artifact is
not a 1.2 artifact and running that command on one would be wrong. 1.2 still gets the exact command
with a resolved path, 1.0 still gets the opaque/no-lossless-migration message, and an unknown
version falls through to the same honest refusal. Migration now emits 3.0 with
`required_capabilities: []`, and removing that field is itself a rejection, so the negotiation
mechanism 3.1+ will depend on cannot be quietly dropped.

Critically, the rename changed only the name: migrating a real 1.2 publication on a copy still
leaves `territory_*.npz` byte-identical and still yields realization (8,8) units and (15,15) owners
on both sides of the migration, and the result is accepted by the production overlay.

H3 is now genuinely useful — an identity mismatch names the field, the expected value and the found
value, e.g. `identity.residue_disposition: expected 'parent_retained', found 'minted'`. That is
what lets a third party distinguish a policy rejection from a corrupt artifact, which is the whole
point of declaring identity as fields rather than as a version number. H4's watershed failure now
names `--watershed-manifest`.

No new findings. Everything still open was declared rather than discovered, so this is
**APPROVE_WITH_MINOR_COMMENTS** and the run is complete.

One correction of my own: my first H3 probe wrote the mutated manifest to `/tmp`, which broke the
`plan_digest` check before identity was ever evaluated and made the diagnostics look absent. That
was a fault in my harness — `plan_file` resolves relative to the manifest's own directory. Redone
in place, the diagnostics are correct.

## Diff Baseline

run_start_ref: c705458ae5b907bb9c32a75c63c85c6aad7edec7

Parent `HEAD` c705458a and nested `lib/abiss` `HEAD` 452efa5f unmoved. The live native96 manifest
is still schema 1.2 with SHA-256 `59bbece6...73e`; every experiment ran on `/tmp` copies, since
removed. `code_v4` ran no `squeue` — the coordinator supplied the zero-consumer clearance — and
launched no cluster job. Evidence in `state/review_v4.review.raw.md`.

## Findings

- **[minor] I1 — Aggregate `black --check` is still nonzero.** It would reformat pre-existing code
  in `nucleus_competition.py` and `compare_nucleus_competition.py`. Codex declined the out-of-scope
  churn and disclosed it plainly, which is the right call inside a scoped round; focused `black`
  on the four in-scope files passed. Worth a separate formatting-only pass so the signal stops
  being noisy, not worth reopening this run.
- **[minor] I2 — The deferred 3.1+ surface is now the run's main debt**, and it is real debt rather
  than a nice-to-have: `inputs[].semantics` is what would have prevented the months-long
  win144/win96 path-versus-content misattribution, since an affinity URI differing only by a
  `_win48x96x96` suffix carries no machine-readable statement of the inference window. It is
  additive, so it lands as 3.1 without breaking 3.0 consumers.

Verified positively and not to be redone: the 3.0/2.0/1.2/1.0/unknown ladder; migration emitting
3.0 with capabilities; realization invariance across migration; byte-identical territories;
production acceptance of a migrated publication; field-level identity diagnostics;
`--watershed-manifest` named in the watershed failure; H1's two-sided skip guard; H2's invariance
test.

## Tests to Add

None for this round. The three `review_v3` asked for are present and pass, and I re-derived their
assertions independently.

For whoever picks up 3.1: a test that a 3.1 manifest listing an unimplemented capability is
rejected by a 3.0-only consumer, which is the first thing the `required_capabilities` mechanism
should be made to prove.

## Questions

- **Q-1.** The run is complete. The remaining work is operational, not implementation: migrating
  the live native96 publication (still schema 1.2), then the `b3_gate025` relaunch, the two
  `abiss_tuning/fixed_tiers/.../z4_nucleus_matchguard{,_win96}` consumers, and the
  `ABISS_NUC_MIN_TAGGED=50` sweep. The migrate command is in `docs/nucleus_competition_review.md`
  and in the rejection message itself.
- **Q-2.** Still open from earlier rounds and unaffected by this one: gates 2–8 and the 7–11 h
  cluster baselines are unrun and unclaimed, so the refactor is not demonstrated to be
  behaviour-preserving at cluster scale; and win144's schema-1.0 ids remain opaque, so that arm
  stays audit-only and barred as a comparison reference.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
