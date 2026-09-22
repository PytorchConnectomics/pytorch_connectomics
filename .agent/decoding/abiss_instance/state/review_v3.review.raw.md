# review_v3 raw evidence log (planner-owned, in-session)

## Reproduced

    pytest tests/unit/test_abiss_nucleus_competition.py -q  -> 27 passed (26 at code_v2)
    new test at line 742: test_native96_realization_is_invariant_across_migration
    4 skip guards applied: lines 667, 698, 726, 741

## H1 verified BOTH ways, by exercising the helper directly

    _skipif_missing_zebrafinch_artifacts(Path("/nonexistent/zebrafinch_run"))
      PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS unset -> condition True  (skip)
      PYTC_REQUIRE_ZEBRAFINCH_ARTIFACTS=1     -> condition False (runs, then fails on missing data)
    reason in both cases: "missing gitignored zebrafinch artifact directory: /nonexistent/zebrafinch_run"

Exactly the semantics the coordinator ruled for on review_v2 Q-2: third parties skip, our own CI
can be made to fail rather than silently under-cover. The reason names the exact path.

## H2 satisfied

`test_native96_realization_is_invariant_across_migration` pins the property the coordinator had
only measured by hand. It is guarded by the NATIVE96_RUN skip.

## H3 / H4 NOT implemented -- infrastructure, not coder judgement

code_v3 reports, and its log confirms:

    squeue -u weidf  ->  "Error creating slurm stream socket: Operation not permitted"
    attempted twice, failed both times

The coordinator's own code_v3 prompt made a successful `squeue` a precondition for editing
`lib/abiss/scripts/*.py`, so the coder correctly stopped rather than editing blind. This is the
constraint working as designed. Note the same check SUCCEEDED during code_v2, so it is the codex
sandbox's access to the Slurm socket that is unreliable, not a permanent limitation.

Coordinator error to correct in code_v4: H4 touches only
`dev/zebrafinch/migrate_nucleus_competition.py`, which is NOT under `lib/abiss/scripts/`, so the
precondition should never have gated it. H3 does touch `nucleus_overlay.py` and was correctly
gated.

The coordinator ran the check itself instead:

    squeue -u weidf | grep -E 'me_L|remapagg|nuccomp|mgrag|abshard'  -> 0 live consumers
    (also 0 immediately before the code_v2 and code_v3 launches; nothing has been submitted since)

## H5 -- new finding, discovered by the coordinator while answering the operator's version question

`schema_version: "2.0"` has denoted two mutually incompatible contracts inside this run:

    code_v0/v1-era 2.0 : no identity block, no ledger, and REQUIRED
                         emitted.count(parent_id) == 1
    code_v2-era   2.0 : identity block and ledger REQUIRED, and a territory emitting the parent
                         id is FORBIDDEN (validate_publication_identity, "retired repair parents
                         cannot be emitted by adjudicated territory")

An artifact valid under one is rejected by the other, under the same version string, with the
governing rule inverted. The version policy this run adopted -- MAJOR mismatch is a hard reject --
is unenforceable if the number does not move when the contract does.

The decision document names the target version three times, e.g. "Schema 3.0. Every field below is
required" and "a `migrate` tool writes a new 3.0 manifest". Codex extended 2.0 because neither
review_v1 nor the coordinator's code_v2 prompt named a version; that omission is the coordinator's.

Cost of fixing now, measured -- every nucleus manifest on disk:

    1.2  dev/zebrafinch/abiss_tuning/fixed_tiers/work/z4_nucleus_matchguard/.../manifest.json
    1.2  dev/zebrafinch/abiss_tuning/fixed_tiers/work/z4_nucleus_matchguard_win96/.../manifest.json
    1.0  dev/zebrafinch/wholevol_arm2mix_r10_nuc_competitive_v2/.../manifest.json
    1.2  dev/zebrafinch/wholevol_arm0_native96_nuc_matchguard/.../manifest.json
    1.0  dev/zebrafinch/wholevol_arm096_nuc_competitive_v2/.../manifest.json

NO schema-2.0 artifact exists anywhere. The rename is free today and stops being free the moment
native96 is migrated or a fresh nuccomp publishes.

2.1 would be wrong: a MINOR must be additive and this inverted a rule. The deferred contract
surface (inputs[].semantics, required_capabilities, native_id_space, id_encoding) is additive, so
it lands as 3.1+ and the shipped subset stays consistent with the policy.

`required_capabilities: []` must exist in 3.0 itself, because the policy says a consumer accepts a
newer minor only if it implements every listed capability -- unusable if the base version lacks
the field.

READY: no
