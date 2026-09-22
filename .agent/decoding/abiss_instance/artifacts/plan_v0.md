# Plan v0

## Summary

Make `nuccomp` (the competitive nucleus growth stage) faster and harder to break, without
changing its output. The stage is **88% of post-watershed wall-clock** (440.6 of 500.6 min on
win144; 11h18m42s on native96) yet runs single-node and serial to repair 8-9 units, while every
neighbouring ABISS stage shards across up to 1520 cores.

The plan is deliberately ordered **measure → parallelize → harden**, and its central safety
requirement is a **bit-identical manifest** against the two completed reference runs. This is a
refactor: if the output changes at all, the change is wrong.

The one thing this plan refuses to do is optimize on a guess. `task.md` §5.1 states, and labels
as an assumption, that the global scan rather than the per-unit floods dominates those 11 hours.
Nobody knows, because the stage emits three log lines and no timestamps. So instrumentation
lands first and is not merely diagnostic — it is the acceptance evidence for the speedup.

## Scope

In scope:

- **A. Instrumentation** of `lib/abiss/scripts/nucleus_competition.py`: per-phase and per-unit
  timings, peak RSS, and a machine-readable stage report written beside the manifest.
- **B. Sharded execution** of the stage: `scan → flood (array over contact units) → merge`,
  chained with `--dependency=afterok` like every other ABISS stage, replacing the single job.
- **C. Robustness**: loud failure when a nucleus run performs zero repairs; tests for the four
  fail-closed conditions currently only claimed; and a B2 regression test that does not stub the
  code path B2 broke.

Explicitly out of scope, with reasons:

- **Watershed reuse as a first-class flag** (`task.md` §5.3). High value for `NUC_*` sweeps and
  proven manually this week, but it changes `abiss_chunk.py`'s payload precedence
  (`CHUNKMAP_INPUT` is assigned from `CHUNKMAP_OUTPUT` inside a `payload.update()` that overrides
  `param_overrides`, at `abiss_chunk.py:663-664`). That is a separate incremental change to a
  shared execution path; bundling it would make the bit-identity gate below ambiguous.
- **`ABISS_NUC_MIN_TAGGED` sweep** (R4). That is an experiment, not a refactor.
- **The five unrun acceptance checks** (`task.md` §4). Two are already in flight as jobs
  2872415-2872420; they gate *shipping the feature*, not *this refactor*.
- Any change to the competition algorithm, its parameters, or the cannot-link.

## Proposed Changes

### A. Instrumentation (do this first; it is the acceptance evidence)

1. Add a lightweight phase timer to `nucleus_competition.py` covering, at minimum: nucleus mask
   load, `scan instance geometry`, `map N nuclei to watershed ids`, contact/bridge
   classification, each flood, and manifest write.
2. Emit each phase as a timestamped log line **and** accumulate into
   `nucleus_competition/stage_report.json` beside the manifest: phase name, wall seconds, and
   for floods the unit id, parent id, box extent in voxels, and voxels visited.
3. Record peak RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) at stage end. `MaxRSS` was not
   captured by `sacct` for either completed run, so the 8 CPU / 128 GB request is unvalidated.
4. No behavioural change. The manifest must be byte-identical with instrumentation on.

### B. Sharded execution

Split the stage into three ops that mirror the existing `sbatch_abiss_shard.sh` pattern:

1. **`nuccomp_scan`** — loads the nucleus mask, maps nuclei to watershed ids, classifies contact
   units versus bridges, and writes a *plan file* (`nucleus_competition/units.json`) listing each
   contact unit with its parent watershed id, anchor nuclei, repair box, and the deterministic
   territory ids it will produce. Shardable by nucleus range if instrumentation shows this phase
   dominates; if it does not, it stays a single job and the array in step 2 is the whole win.
2. **`nuccomp_flood`** — a Slurm **array over contact units** (8-9 tasks today). Each task floods
   exactly one unit inside its own bounded box and writes `terr_<unit>.npz` plus a per-unit JSON
   record. The units are independent by construction: each floods within one parent watershed
   object, and overlapping repair scopes for one parent are already a fail-closed error.
3. **`nuccomp_merge`** — assembles `manifest.json` from the per-unit records **in sorted unit
   order** (never in completion order), re-runs the global cross-unit invariants that a single
   process previously got for free (generated-ID collision check across all units; no two units
   claiming overlapping scopes of one parent), and **fails closed if any unit record is missing**.

Determinism requirements, which are the crux:

- Territory ids are derived from the parent/anchor pair and collision-checked; sharding must not
  move that derivation into per-shard state. Ids are computed in `nuccomp_scan` and written to
  `units.json`, so a flood task *consumes* its ids rather than allocating them.
- The manifest key order and every list order must be sorted, not arrival-ordered.
- `nuccomp_merge` must reject a partial fan-in rather than emit a manifest describing fewer
  repairs than `units.json` declared — that is exactly the "silent empty artifact scoring as a
  clean null" failure this repo has hit before.

Rebuild note: this touches only Python under `lib/abiss/scripts/`, not the compiled `ws2`/`agg`
binaries, so the watershed-reuse equivalence argument in `task.md` §5.3 is unaffected.

### C. Robustness

1. **Loud zero-repair** (R2). A run configured with `NUC_PATH` that finds zero multi-nucleus
   watershed ids currently writes `repairs: []` and produces a segmentation indistinguishable
   from plain ABISS, silently — the r10 arm did exactly this. Emit a clearly-marked warning line,
   set an explicit `"zero_repairs": true` flag in the manifest and stage report, and have
   `submit_wholevol_sharded.sh` surface it. Do **not** make it a hard error: zero repairs is a
   legitimate outcome (`test_no_repair_manifest_preserves_base_segmentation_and_nuclei` asserts
   the no-op is correct). This is an observability fix, not a policy change.
2. **Fail-closed tests** (R7). Five conditions are claimed to abort the stage; only
   `test_missing_competition_manifest_fails_closed` exists. Add tests for the other four:
   missing unanimous seeds, overlapping repair scopes for one parent, generated-ID collision, and
   an incompatible watershed manifest. Sharding makes two of these *more* important, because they
   move from implicit single-process guarantees into `nuccomp_merge`.
3. **Strengthen the B2 regression test** (R1).
   `test_final_aggregation_remap_replays_competition_overlay` monkeypatches
   `overlay.apply_nucleus_competition` — but B2 was two separately-correct paths disagreeing
   about *which volume* they operated on, so stubbing the overlay assumes the wiring that broke.
   Add a test that drives both `cut_chunk_agg.py` and `cut_chunk_remap.py` over the same synthetic
   cutout with the real overlay and asserts the two produce the same labels. Keep the existing
   test.

## Files and Areas

| path | change |
|---|---|
| `lib/abiss/scripts/nucleus_competition.py` (747) | phase timers, stage report, split into scan/flood/merge entry points, sorted-order manifest assembly |
| `lib/abiss/scripts/nucleus_overlay.py` (279) | unchanged unless the merge refactor needs a shared loader |
| `dev/zebrafinch/sbatch_nucleus_competition.sh` (36) | becomes the scan/merge wrapper, or is replaced by three thin wrappers |
| `dev/zebrafinch/sbatch_nuccomp_flood.sh` (new) | Slurm array over contact units |
| `dev/zebrafinch/submit_wholevol_sharded.sh` (115) | `submit_nucleus_competition()` submits the 3-op chain and surfaces zero-repair |
| `tests/unit/test_abiss_nucleus_competition.py` (442) | fail-closed tests, unstubbed B2 test, shard-merge determinism tests |

Untouched on purpose: `NucExtractor.hpp`, `mean_aggl.cpp`, `Types.h`, `cut_chunk_agg.py`,
`cut_chunk_remap.py` (the last two are read by the new test but not modified).

## Verification Plan

The gate is **bit-identity against known-good reference runs**, not a plausibility argument.

1. **Unit tests**: `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` — all 442
   lines' worth of existing tests must pass unchanged, plus the new ones.
2. **Manifest bit-identity (the primary gate).** Re-run the refactored stage against both
   completed references and diff the manifest, ignoring only the new instrumentation keys:
   - `wholevol_arm0_native96_nuc_matchguard` — expect **8 multi-nucleus watershed ids → 8 contact
     units, 0 bridges**, and the same territory ids for the same parent/anchor pairs;
   - `wholevol_arm096_nuc_competitive_v2` — expect **9 contact units**, including the known
     `flood 2/9 parent 72199226020331523 anchors [611, 651]`.
   Any difference in unit count, parent ids, anchor pairs, boxes, or territory ids fails the change.
3. **Shard-count invariance**: run `nuccomp_flood` with the array throttled to 1, then to N, and
   assert the merged manifests are byte-identical. This is the determinism claim, tested rather
   than asserted.
4. **Partial-fan-in failure**: delete one per-unit record and assert `nuccomp_merge` exits
   non-zero and writes no manifest.
5. **Speedup, measured**: report the stage report's phase table before and after, and total
   stage wall-clock from `sacct`, against the 440.6 min (win144) / 11h18m42s (native96) baselines.
   A change that does not move the dominant phase is not done, whatever the code looks like.
6. **Zero-repair loudness**: a synthetic run with no multi-nucleus objects emits the warning and
   sets `zero_repairs: true`, while its segmentation output stays identical to the no-op case.
7. **No downstream drift**: after a refactored `nuccomp`, `me_L0` consumes the territories and the
   materialized 611/651 pair remains in distinct dominant segments.

## Risks and Questions

- **R-a. The speedup may be small if the scan dominates and does not shard cleanly.** The array in
  B.2 caps at ~9x on the flood phase alone. If instrumentation shows floods are 10% of runtime,
  B.2 buys ~1.1x and the real work is sharding the scan. **This is why A lands first**; the coder
  should report the phase table before committing to B's shape, and may reasonably propose a
  different split.
- **R-b. Sharding can break determinism in ways tests miss.** Mitigated by computing ids in scan,
  sorting all merge output, and testing shard-count invariance — but a reviewer should check
  whether any per-unit code path reads mutable global state.
- **R-c. Cross-unit invariants lose their free ride.** A single process could check ID collisions
  and overlapping scopes implicitly; the merge stage must now do it explicitly, and a missed one
  is a silent correctness hole rather than a crash.
- **R-d. More jobs, more partial-failure surface.** The B1 incident (a 4-second env failure taking
  down a 10-hour chain) argues for fewer moving parts, not more. The merge stage's fail-closed
  behaviour is the mitigation, and the new wrappers must use the `set -u`-after-activation order.
- **Q-1.** Should `nuccomp_scan` shard by nucleus range in this change, or stay single and be
  sharded in a follow-up once the phase table justifies it? Plan assumes the latter unless
  instrumentation says otherwise.
- **Q-2.** Is `units.json` an acceptable new intermediate artifact, or should the plan file live
  inside the existing manifest schema with a `planned`/`completed` distinction?
- **Q-3.** The two reference runs used different `ABISS_NUC_MIN_TAGGED` (50 vs 1024). Bit-identity
  must therefore be checked per-run against its own parameters, not across runs. Confirm the coder
  reproduces each with its original `param`.

## Changes Since Previous Plan Version

Initial plan.
