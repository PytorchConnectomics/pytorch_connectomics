# review_v2 raw evidence log (planner-owned, in-session)

Planner is `claude`; the artifact under review is Codex's `code_v2`, so this is a genuine
cross-model review. Nothing was mutated: parent HEAD c705458a and nested lib/abiss HEAD 452efa5f
unmoved, and the live native96 publication is still schema 1.2 after the review finished.

## Reproduced

    pytest tests/unit/test_abiss_nucleus_competition.py -q   -> 26 passed (22 at code_v1)
    G2  "parent watershed id exactly once"                   -> absent from BOTH former sites
    G4  nucleus_overlay.py:446 canonicalize_qualified_segments
        now precedes :447 `if not repairs:`                  -> ordering fix confirmed

## Ledger arithmetic recomputed from the live manifest

    qualified segments 85 = 77 single-owner + 8 multi-owner
    distinct owner labels from the 77 single-owner sources : 15
    labels absorbing more than one source                  : 15   max fan-in 7

Exactly the numbers review_v1 G3 asserted. build_publication_ledger records all 85 retirements
(77 `owner_canonicalization` + 8 `competitive_split`) and 15 consolidations.

## Identity is VERIFIED, not asserted — the central design question

`validate_publication_identity` (nucleus_overlay.py:116) recomputes every emitted id from the
declared mint via `minted_nucleus_id`, enforces the nucleus<->emitted bijection in both
directions, forbids an adjudicated territory from emitting the parent id, and rebuilds the ledger
and compares it. So migration cannot stamp a declaration the data does not satisfy — the
declaration is checked. This is the concrete answer to review_v1 G2's rule.

## Migration reviewed by reading, then exercised

Read dev/zebrafinch/migrate_nucleus_competition.py in full. Properties confirmed by reading:
atomic write (x-mode + fsync + os.replace); `_preserve_once` refuses to overwrite a backup with
different content; explicit refusal of schema 1.0 ("schema 1.0 ids are opaque"); territory dtype,
bbox, factor and both marker domains cross-checked against the manifest; path-escape check;
re-running on an already-migrated manifest fails loudly rather than double-migrating.

Exercised on a COPY at /tmp (never the live publication):
    territory_*.npz aggregate md5 identical before and after
    completes in under a second (no flood recomputation)
    production load_validated_manifest REJECTS the original 1.2 manifest, message names the
      exact migrate command with the real path
    production load_validated_manifest ACCEPTS the migrated copy (schema 2.0, 8 repairs)

## H2 — realization invariance across migration, measured

Built a throwaway run dir at /tmp holding the tol=0 audit plus a copy of the publication, ran the
gate, migrated in place, ran it again:

    BEFORE migration (schema 1.2): units (8, 8)   owners (15, 15)
    AFTER  migration (schema 2.0): units (8, 8)   owners (15, 15)
    REALIZATION INVARIANT ACROSS MIGRATION: True

So the 2.0 branch of `_repair_owner_labels` is equivalent to the legacy branch on real data, and
migrating the live publication will not change the gate's answer. No test pins this.

## H1 — live-data test dependency, quantified

    git check-ignore -v dev/zebrafinch/wholevol_arm0_native96_nuc_matchguard
      -> .gitignore:159:dev/
    git ls-files <that run dir> | wc -l  -> 0     (absent from a fresh clone)

Three of 26 tests depend on it, with NO skipif:
    test_realization_gate_reproduces_the_two_published_run_oracles   (direct, lines 705-706)
    test_native96_mint_is_nucleus_scoped_across_units_and_canonicalization  (via fixture)
    test_native96_publication_ledger_closes_every_measured_retirement       (via fixture)

The fixture `migrated_native96_manifest` (line ~127) does an unguarded
`shutil.copytree(NATIVE96_PUBLICATION, ...)`. A third party gets errors, not skips. No disk read
happens at import/collection time, so only these three are affected, not the whole module.

Correcting my own first pass: a grep of test bodies found only 1 dependent test; the other two
hide the dependency in a fixture. The count is 3.

## H3 / H4 — read-only observations

`validate_publication_identity:120` compares the whole identity dict for exact equality against
the single supported declaration, and raises "lacks the supported identity declaration" without
naming the disagreeing field. `migrate_nucleus_competition._infer_watershed_manifest` derives the
watershed from the legacy manifest's `base_watershed`, whereas the overlay derives it from
`global_params`; a mismatch surfaces only at job time.

READY: no
