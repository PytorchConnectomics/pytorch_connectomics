# review_v4 raw evidence log (planner-owned, in-session)

All three carried findings verified resolved by direct exercise, not by reading claims.

## Suite

    pytest tests/unit/test_abiss_nucleus_competition.py -q  -> 28 passed
    (22 at code_v1, 26 at code_v2, 27 at code_v3, 28 now)

## H5 -- version ladder, exercised against a real migrated publication

    PUBLICATION_SCHEMA_VERSION = "3.0"   (nucleus_overlay.py:13)

    3.0  ACCEPTED in place
    2.0  rejected | offers-migrate-cmd=False
         "schema '2.0' is a withdrawn pre-release contract that inverted the parent-id rule
          and is not accepted by the production overlay"
    1.2  rejected | offers-migrate-cmd=True   (exact command with resolved path)
    1.0  rejected | "no lossless production migration; use the read-only comparison harness"
    9.9  rejected | same unknown-version message

Correct on every rung: 2.0 does NOT offer the 1.2 migrate command, because a 2.0 artifact is not
a 1.2 artifact.

## H5 -- migration end to end, on a /tmp copy

    migrated schema_version   : 3.0
    required_capabilities     : []
    realization before/after  : (8, 8, 15, 15) -> (8, 8, 15, 15)
    territory files unchanged : True
    production accepts 3.0    : YES

    required_capabilities removed -> "nucleus competition manifest lacks required_capabilities"

## H3 -- identity diagnostics now name field, expected and found

Mutated in place (my first attempt wrote the manifest to /tmp, which broke the plan_digest check
before identity was ever reached -- a fault in my harness, not the code; redone in place):

    identity.residue_disposition -> "unsupported nucleus competition identity field
        identity.residue_disposition: expected 'parent_retained', found 'minted'"
    identity.scope               -> "... identity.scope: expected 'nucleus', found 'territory'"
    identity.parent_disposition  -> "... identity.parent_disposition: expected 'retired',
        found 'winner_inherits'"

## H4 -- watershed failure names the escape hatch

    "nucleus competition manifest was built from another watershed identity; schema-1.2 migration
     must select the authoritative manifest with --watershed-manifest"

## Safety

    parent HEAD  c705458a  unmoved
    lib/abiss    452efa5f  unmoved
    live native96 manifest still schema 1.2, sha256 59bbece64bf6f988d45309a95c146e8386bfbb6ebca9b4554c681ff271ed073e
    all experiments on /tmp copies, removed afterwards
    coordinator supplied the zero-consumer clearance; code_v4 ran no squeue and launched no job

## Residual, all previously declared -- no new findings

- Aggregate `black --check` still nonzero: it would reformat pre-existing code in
  nucleus_competition.py and compare_nucleus_competition.py. Codex deliberately declined the
  out-of-scope churn and said so. Focused black on the four in-scope files passed.
- Gates 2-8 and the 7-11 h cluster baselines remain unrun and unclaimed.
- win144 schema 1.0 stays opaque with no lossless migration; audit-only.
- Deferred 3.1+ surface: inputs[].semantics, capability negotiation, native_id_space, id_encoding.

READY: yes
