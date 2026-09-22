# Plan v1

## Summary

Same goal as v0 — make `nuccomp` faster and harder to break — but the shape changes in response
to the review. v0 tried to instrument *and* re-architect in one increment while the evidence
needed to choose the architecture did not exist yet. The reviewer was right that this is not
executable, and right that three of the mechanisms were self-contradictory.

**This increment is behaviour-preserving by construction.** It adds measurement, observability
and tests; it does not change execution structure. That makes the regression gate exact rather
than negotiated: territory arrays must be **byte-identical** and the manifest identical except
for documented additive keys. The parallelization is fully designed here, with a numeric decision
threshold, and is explicitly **deferred to the next increment** pending the phase table — which is
what the reviewer asked for in Q-1.

It also delivers the `task.md` §7 written deliverables (R1–R7 answers, the watershed-reuse
recommendation, a ship verdict). v0 dropped them; they are owed regardless of whether code lands.

## Scope

In scope:

- **A. Instrumentation of the existing serial path.** Phase timers, per-unit records, defined
  resource semantics, and a machine-readable `stage_report.json`. No structural change.
- **C. Robustness.** Loud zero-repair, the five fail-closed tests (not four), and a B2 regression
  test with a real oracle.
- **D. Written deliverables** answering `task.md` §7: R1–R7, the watershed-reuse recommendation,
  and a ship-readiness verdict, committed as `docs/nucleus_competition_review.md` in the run
  folder.

Designed but **not implemented** in this increment:

- **B. Sharded execution.** Full design in `## Proposed Changes` §B including the ID rule, the
  array-bounds mechanism and the stale-artifact controls, gated on the §A measurement by an
  explicit threshold. Implementing it now would mean choosing between a ~9x-capped flood array
  and an unscoped scan shard without evidence.

Out of scope, unchanged from v0: watershed reuse as a first-class flag (separate change to
`abiss_chunk.py` payload precedence), the `ABISS_NUC_MIN_TAGGED` sweep (an experiment), and any
change to the competition algorithm or cannot-link.

Acceptance work is **no longer wholly deferred** (finding 8): the unchanged-material check and
the `nuc_cuts.data` rejected-edge counts are regression gates for a behaviour-preserving refactor
and are in the verification plan below.

## Proposed Changes

### A. Instrumentation of the serial path (the whole of this increment's runtime change)

1. Phase timers in `lib/abiss/scripts/nucleus_competition.py` around: nucleus mask load, scan
   instance geometry, map nuclei→watershed ids, contact/bridge classification, **each flood**,
   and manifest write. Each phase logs a timestamped line and appends to
   `nucleus_competition/stage_report.json`.
2. Per-flood record: unit index, parent id, anchor pair, box extent in voxels, voxels visited,
   territories produced with their voxel counts, wall seconds.
3. **Resource semantics defined** (finding 12): report `ru_maxrss` *with its unit stated*
   (KiB on Linux) and label it `peak_rss_kib_process`, explicitly the single-process peak of the
   serial stage. The report carries a `topology: "serial"` field so that when sharding lands, the
   per-task / max-task / concurrent-aggregate fields can be added without ambiguity about what an
   older report meant.
4. `stage_report.json` includes `schema_version`, the run name, and the `param` fingerprint, so a
   phase table can never be silently attributed to the wrong run or parameter set.

### B. Sharded execution — designed, gated, not implemented

**Decision threshold** (finding 6). From the §A phase table on a full run:

```text
floods >= 40% of stage wall     -> implement the per-unit flood array (B.2). Expected ceiling
                                   is bounded by unit count (8-9 today), so best case ~1/(0.6+0.4/9)
                                   ~= 1.6x overall; state the expected number before building.
scan+map >= 60% of stage wall   -> the flood array is not worth it alone; design scan sharding
                                   by nucleus range or by watershed chunk, which is where the
                                   remaining 10 hours live.
neither dominant               -> report and re-plan; do not build either.
```

**B.1 Territory-ID rule** (finding 1 — the defect that sank v0). The specification says the
largest territory retains the parent watershed id and the others take deterministic ids above
`2^60` derived from the parent/anchor pair. Scan cannot know which territory is largest, so v0's
"scan preallocates, flood consumes" was incoherent. Resolution:

```text
scan   preallocates a deterministic >=2^60 id for EVERY territory, including the one that will
       later win, keyed on (parent, anchor). No size knowledge required.
flood  writes each territory under its preallocated id and reports its voxel count.
merge  selects the winner by (largest voxel count, then lowest anchor id -- the existing
       tie-break) and records `parent_id_assigned_to: <territory>` in the manifest.
       The overlay applies that mapping; no voxel rewrite is needed.
```

Determinism then rests only on the floods themselves being deterministic, which is testable
directly (B.4) rather than assumed.

**B.2 Array bounds without dynamic submission** (finding 2). `submit_wholevol_sharded.sh` cannot
know the unit count at submit time. Rather than submitting from inside the scan job, use a
**fixed-capacity array** `0..NUC_MAX_UNITS-1` (default 64): task *i* reads `units.json`, exits 0
immediately if `i >= len(units)`, else floods unit *i*. This keeps the static `afterok` chain,
costs a few one-second no-op tasks, and **handles the zero-unit case naturally** — every task
no-ops, merge writes a zero-repair manifest. Scan **fails closed** if the unit count exceeds
capacity rather than silently truncating.

**B.3 Stale-artifact controls** (finding 4). `units.json` is immutable and carries
`schema_version`, a config/input fingerprint (`param` digest + `WS_PATH` + nucleus mask path and
size) and a `plan_digest`. Every per-unit record and `terr_*.npz` embeds the `plan_digest` of the
`units.json` it was produced from. Merge accepts a record only if its digest matches, requires
**exactly** the expected artifact set, writes the manifest via a temp file and atomic rename, and
**removes or refuses to overwrite** a pre-existing manifest so a stale one cannot survive a failed
merge. This is the same failure family as the `flood_contact_units.py` incident where a resumed
run emitted `[]` over completed records.

**B.4 Determinism tests**: flood the same unit twice and assert byte-identical `terr_*.npz`; run
the array at throttle 1 and at N and assert the merged manifest is identical; assert no per-unit
path reads mutable global state.

### C. Robustness

1. **Loud zero-repair** (R2). Warning line, `"zero_repairs": true` in manifest and stage report,
   surfaced by `submit_wholevol_sharded.sh`. Not an error — the no-op is legitimate and already
   asserted correct by `test_no_repair_manifest_preserves_base_segmentation_and_nuclei`.
2. **Five fail-closed tests** (finding 10 — v0 miscounted). The claimed conditions are: missing
   unanimous seeds; overlapping repair scopes for one parent; generated-ID collision; incompatible
   watershed manifest; **invalid nucleus coordinates**. `test_missing_competition_manifest_fails_closed`
   covers a *missing* manifest, which is not the same as an *incompatible* one. Add all five, and
   state reachability for each: seeds/coords are input-driven and reachable; overlapping scopes
   and ID collisions are currently unreachable in the serial path but become reachable under
   sharding, which is precisely why they need tests before B lands.
3. **B2 test with an oracle** (finding 5). v0's "both paths agree" would pass if both returned the
   *unsplit* result. The new test constructs a cutout with a known competitive split, drives the
   real `apply_nucleus_competition` through both `cut_chunk_agg.py` and `cut_chunk_remap.py`, and
   asserts (a) the competitive labels are present in the remap path's output — not merely equal to
   the other path — and (b) two distinct owners remain distinct after the remap. Keep the existing
   stubbed test as a cheap guard.

### D. Written deliverables

`docs/nucleus_competition_review.md` in the run folder, answering `task.md` §7: R1–R7 with a
position on each (including R3 containment / `separation_claim = local_only`, R5 ownership
determinism, R6 the `z4_y6_x1` null), the watershed-reuse recommendation, and a ship verdict
constrained by R4's parameter non-equivalence. These are written analysis, not code.

## Files and Areas

| path | change |
|---|---|
| `lib/abiss/scripts/nucleus_competition.py` (747) | phase timers, per-flood records, `stage_report.json`, zero-repair flag |
| `dev/zebrafinch/sbatch_nucleus_competition.sh` (36) | pass through report path; keep `set -u`-after-activation order |
| `dev/zebrafinch/submit_wholevol_sharded.sh` (115) | surface `zero_repairs` |
| `tests/unit/test_abiss_nucleus_competition.py` (442) | five fail-closed tests, oracle-bearing B2 test |
| `.agent/decoding/abiss_instance/docs/nucleus_competition_review.md` (new) | §7 deliverables |

Not modified: `NucExtractor.hpp`, `mean_aggl.cpp`, `Types.h`, `nucleus_overlay.py`,
`cut_chunk_agg.py`, `cut_chunk_remap.py` (the last two are exercised by the new test).

## Verification Plan

The gate is exact **because this increment is behaviour-preserving** (finding 3). No serializer
change, no reordering, no structural change — so the honest claim is byte identity, not
"normalized semantic equality". If any of that stops being true, the plan must be revised rather
than the gate loosened.

1. `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` — all existing tests pass
   unchanged, plus the new five fail-closed tests and the oracle B2 test.
2. **Territory arrays byte-identical.** Re-run the instrumented stage on
   `wholevol_arm0_native96_nuc_matchguard` with its own original `param`
   (`ABISS_NUC_MIN_TAGGED=1024`) and on `wholevol_arm096_nuc_competitive_v2` with its own
   (`=50`); every `terr_*.npz` must be byte-identical to the reference. Per the reviewer's Q-3,
   each run is compared **only against itself**; the two are never compared to each other.
3. **Manifest identical except additive keys.** Diff must show only the documented new keys
   (`zero_repairs`, report pointers). Expect 8 contact units / 0 bridges on native96 and 9 on
   win144 including `parent 72199226020331523 anchors [611, 651]`.
4. **Instrumented serial baseline** (finding 11). The "before" phase table cannot come from the
   uninstrumented completed runs, so §A's first full run *is* the baseline. Record it in the run
   folder. Future speedup claims compare **end-to-end critical-path elapsed time**, never summed
   array-task time.
5. **Zero-repair loudness**: synthetic no-multi-nucleus run emits the warning, sets
   `zero_repairs: true`, and leaves the segmentation identical to the no-op case.
6. **Downstream regression gates** (finding 8, replacing v0's blanket deferral): after the
   instrumented stage, (a) material outside every repair box is unchanged versus the reference
   segmentation, and (b) `nuc_cuts.data` rejected-edge counts per hierarchy level match the
   reference. The 611/651 materialized check remains, but is no longer claimed to establish
   "no downstream drift" by itself.
7. `bash /home/weidf/.claude/skills/ccc/scripts/ccc-validate.sh .agent/decoding/abiss_instance/`
   stays green, and `git rev-parse HEAD` still equals `run_start_ref`
   (`c705458ae5b907bb9c32a75c63c85c6aad7edec7`) — no commits during the run.

## Risks and Questions

- **R-a. This increment yields no speedup by itself.** It buys the evidence to choose one, plus
  robustness. If the operator needs wall-clock relief immediately, that is an argument for
  accepting a larger, riskier increment — say so now rather than after §A lands.
- **R-b. The phase table may show no dominant phase**, leaving B unjustified in either form. Then
  the honest outcome is a re-plan, not a build.
- **R-c. Byte-identity may fail for benign reasons** — nondeterministic dict ordering already
  present, or a timestamp leaking into an artifact. If so, the cause must be found and excluded
  explicitly, never waved through as "close enough".
- **R-d. `NUC_MAX_UNITS` capacity is a new failure mode** when B lands: too low silently caps
  repairs. Mitigated by scan failing closed above capacity, which must be tested.
- **Q-1.** Is `docs/nucleus_competition_review.md` inside the run folder the right home for the §7
  deliverables, or should they go to `dev/zebrafinch/lesson_nucleus_competition.md` where the
  acceptance checklist already lives?
- **Q-2.** Should §A's baseline run be a fresh whole-volume execution (~11 h on native96, real
  cluster cost) or is re-running the stage alone against the existing watershed acceptable? The
  plan assumes the latter — the stage is idempotent given a completed watershed.

## Changes Since Previous Plan Version

Every finding from `plan_v0_review.md` is addressed below; none are dropped.

1. **[major] Territory-ID preallocation contradiction** → resolved in B.1: scan preallocates ids
   for *all* territories including the eventual winner; merge assigns the parent id by
   (voxel count, lowest anchor id) and records the mapping. No size knowledge needed at scan time.
2. **[major] Dynamic array bounds / zero units** → resolved in B.2 with a fixed-capacity array and
   no-op tasks above the unit count; zero units is now the natural case, and over-capacity fails
   closed.
3. **[major] Bit-identity inconsistent and blind to `terr_*.npz`** → this increment is
   behaviour-preserving, so the gate is genuine byte identity **of the territory arrays** plus
   manifest equality modulo documented additive keys (verification 2 and 3).
4. **[major] Stale-artifact hazards** → B.3 adds `plan_digest` stamping, an exact expected
   artifact set, atomic manifest write, and refusal to leave a stale manifest visible.
5. **[major] B2 test lacks an oracle** → C.3 asserts the competitive labels are *present* after
   the real remap path and that distinct owners stay distinct, instead of comparing two paths that
   could both be wrong.
6. **[major] Section B conditional with no threshold** → B now carries a numeric decision
   threshold and is explicitly deferred, per the reviewer's Q-1.
7. **[major] Scope omitted `task.md` §7 and silently became implementation** → §D adds the R1–R7
   answers, watershed-reuse recommendation and ship verdict as deliverables. The coordinator's
   reading is recorded: the CCC task string authorizes implementation *and* §7 remains owed.
8. **[major] Deferring all acceptance work too broad** → verification 6 pulls the
   unchanged-material and rejected-edge checks in as regression gates.
9. **[major] R3/R5/R6 unaddressed; R4 must constrain the verdict** → §D covers all four in the
   written review.
10. **[major] R7 accounting wrong** → C.2 lists all five conditions, notes that the existing test
    covers *missing* rather than *incompatible* manifest, and states reachability for each.
11. **[major] Efficiency acceptance incomplete** → verification 4 makes §A's own run the baseline
    and mandates end-to-end critical-path comparison; B carries expected speedup before building.
12. **[minor] `ru_maxrss` semantics** → A.3 names the field `peak_rss_kib_process`, states the
    unit, and adds a `topology` field so serial and sharded reports are not conflated.
