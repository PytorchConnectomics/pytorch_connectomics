# Plan v2 Review

## Summary

Codex reviewed `plan_v2.md` against `task.md`, `plan_v1.md` and its own `plan_v1_review`, and
returned **READY: no** with **8 major and 4 minor** findings. Raw transcript:
`state/plan_v2_review.review.raw.md` (6,191 bytes). Mutation guard clean, CLI exit 0.

Reviewer's accounting of the ten open items: **#1, #10, #13, #14 closed**; **#6, #7, #8, #9
partially closed**; **#4 and #11 materially open**. Two new findings were added (#11 retry
workflow, #12 ID representation versus the comparison gate).

The direction of travel is convergence: round 1 raised eleven majors about whether the plan made
sense at all, round 3 raises eight about arithmetic, formulas, fixtures and comparison semantics.
The reviewer also accepts the two contested judgements from `plan_v2` — selecting the flood array
is "defensible even at ~1.1× if it truly provides resumable per-unit execution", and the overlay
mapping plus voxel-level gate "do close the substance of finding #1".

**Two findings are objectively correct and the coordinator verified them independently rather than
taking them on trust:**

- **The speedup arithmetic in `plan_v2` was wrong.** For `f=0.7`, `1/(1-f+f/9)` is **2.65×**, not
  the 3.9× stated. Recomputed: `f=0.1 → 1.10×`, `f=0.4 → 1.55×`, `f=0.7 → 2.65×`. In a plan whose
  entire justification is expected speedup, an inflated headline figure is exactly the error that
  must not survive.
- **The critical-path model was wrong too.** Array critical time is
  `nonflood + max(per-unit flood) + scheduler/merge overhead`, not `nonflood + total_flood/9`.
  Units are not balanced, and native96 has **eight** units, not nine.

Finding #12 is the sharpest structural catch: `plan_v2` §B.1 changes the winning territory's id
inside `terr_*.npz` from the parent id to a preallocated high id, which contradicts verification 3's
"manifest differs only by additive keys". The plan cannot both change the id representation and
claim additive-only differences.

## Findings

All 8 major and 4 minor findings are preserved verbatim in
`state/plan_v2_review.review.raw.md`.

**Closed, no further action:** #1 (overlay mapping and voxel-level gate — with a tightening: the
mapping's domain must equal the exact territory-id set, exactly one emitted parent id per unit,
remaining emitted ids unique), #10 (audit-then-test, with each validation required to abort and
publish no canonical manifest), #13 (zero-repair judged after completion), #14 (wording).

**Open, to be closed by `plan_v3`:**

1. [major] **#4 fingerprints** — the affinity/cost input is omitted although flooding consumes it;
   `WS_PATH` is a location, not an identity; "watershed manifest identity" is undefined; the
   nucleus input may be a store rather than one hashable file; code identity must cover Python and
   native.
2. [major] **#6 selection rule** — instrumentation separates "scan instance geometry" from "map
   nuclei→watershed ids", but the rule lumps both into `s`. If geometry dominates, sharding the
   mapping addresses the wrong work. Needs separate fractions and a design for whichever dominates.
3. [major] **#7 efficiency plan** — the flood prediction is numerically wrong (above), and
   "bounded by shard count" is a ceiling rather than an expected scan-sharding speedup.
4. [major] **#8 already-clean nuclei** — comparing contamination reports for nuclei absent from
   `units.json` does not prove voxel-for-voxel invariance; name the exact fields or voxel regions
   and require exact equality or a defined label-equivalence relation.
5. [major] **#9 R5 and R6** — R5 names invariance dimensions but no owning code/test target,
   fixture or ownership oracle; R6 gives no method, input, expected result or verification step.
   Both remain promises rather than planned work.
6. [major] **#11 efficiency acceptance** — no correct prediction and no pass/fail threshold,
   including what happens if the array is *slower* than serial.
7. [major] **new: retry workflow undefined** — nothing specifies how a failed array element is
   requeued, how the merge dependency is repaired after an `afterok` failure, or how completed
   unit artifacts are reused without rerunning scan. Without this the robustness justification for
   the flood array is unestablished. Requires a failure-injection gate: fail unit 8, rerun only
   that unit, prove prior artifacts unchanged, then merge successfully.
8. [major] **new: ID representation versus comparison gate** — see #12 above; define whether
   manifest fields hold internal territory ids or emitted ids, and replace additive-only comparison
   with exact semantic comparison after applying the translation.

## Questions

The reviewer answered both of `plan_v2`'s open questions. The coordinator adopts both, and notes
that **Q-2 reverses a decision made in `plan_v2`**:

- **Q-1 — yes.** The coordinator or operator may execute the 7–11 h cluster baselines after
  `code_v0` supplies exact commands, frozen fingerprints, fresh output namespaces and expected
  artifacts. `code_v0` may complete implementation and focused tests, but **must not claim the
  efficiency gate passed** before both parameter-specific baselines and array runs are recorded.
- **Q-2 — use a fully separate run-scoped staging directory and publish only at the end.** Keep the
  old canonical manifest as the last successful publication and atomically replace it after
  validation. `plan_v2`'s quarantine-before-execution is worse: it creates a window with no valid
  canonical result and can disrupt concurrent readers. `plan_v3` must adopt staging and drop
  quarantine.

`p3-c2` allows `plan_v3`, which is the terminal plan version under the raised rounds.

## Verdict

VERDICT: NEEDS_CHANGES
