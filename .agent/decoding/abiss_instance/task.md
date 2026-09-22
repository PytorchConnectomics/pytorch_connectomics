# Task: ABISS competitive nucleus growth — robustness and efficiency review

**Reviewer: GPT-5.6.** This is a *review and design* task, not an implementation task. The
feature works and has a measured positive effect on two independent affinity substrates. The
question is whether it is **robust** enough to ship and **efficient** enough to tune, and one
stage currently consumes 88% of the pipeline's post-watershed wall-clock.

Everything below was measured on the dates given. Where a claim is an assumption rather than a
measurement, it says so — please do not treat inference as evidence, and please challenge any
of it that the code contradicts.

---

## 1. What the feature is

Two operators, applied in this order, replacing an earlier merge-veto-only design:

```text
competitive seeded watershed = CREATES the missing soma boundary
nucleus cannot-link          = PRESERVES that boundary during agglomeration
```

The motivating failure: an external nucleus instance mask shows two somata that ABISS has fused.
A merge veto cannot help, because at the region-graph level the fused pair is **one node** — in
the worst measured case a single watershed supervoxel held 28,576,381 nucleus-tagged voxels
inside a 243-million-voxel object spanning 88 atomic chunks. Three earlier in-pipeline variants
(nucleus-first merge ordering, conflict-clause veto, global per-supervoxel nucleus table)
therefore changed the result by exactly zero. Competition supplies the missing identity prior;
the cannot-link then stops agglomeration undoing it.

Pipeline position:

```text
affinity + keep mask
  -> ABISS watershed L0..L5          (nucleus-BLIND; see §5 efficiency item 3)
  -> global watershed remap
  -> nucleus contact detection + competitive growth   <-- `nuccomp`, the stage under review
  -> sparse territory overlay while building atomic RAGs
  -> nucleus-aware mean-edge agglomeration L0..L5
  -> final agglomeration remap
```

### Implementation inventory

| file | lines | role |
|---|---:|---|
| `lib/abiss/scripts/nucleus_competition.py` | 747 | the `nuccomp` stage: scan, contact detection, flood, manifest |
| `lib/abiss/scripts/nucleus_overlay.py` | 279 | applies territories to a watershed cutout |
| `lib/abiss/scripts/cut_chunk_agg.py` | 149 | overlays territories while building the atomic RAG |
| `lib/abiss/scripts/cut_chunk_remap.py` | 41 | final remap path — **the site of bug B2 below** |
| `lib/abiss/src/seg/NucExtractor.hpp` | 108 | per-supervoxel `NONE`/`PROPER(id)`/`CONFLICT` records |
| `lib/abiss/src/seg/Types.h` | — | record layout |
| `lib/abiss/src/agg/mean_aggl.cpp` | 1357 | `nuc_can_merge` veto inside mean-edge agglomeration |
| `dev/zebrafinch/sbatch_nucleus_competition.sh` | 36 | the stage's Slurm wrapper |
| `dev/zebrafinch/submit_wholevol_sharded.sh` | 115 | `submit_nucleus_competition()` gates on `NUC_PATH` |

Tests: `tests/unit/test_abiss_nucleus_competition.py` (442), `lib/abiss/tests/test_nuc_algebra.cpp`
(98), `lib/abiss/tests/test_nuc_agg.py` (85).

### Governing parameters (from a materialized run's `param`)

```text
NUC_MIN_SHARE               0.02     a nucleus qualifies in a watershed object only if that
                                     object holds >=2% of that nucleus's own total mask mass
NUC_CONTACT_UM              8.0      signed surface gap below which two nuclei form a contact unit
NUC_COMPETITION_MARGIN_ZYX  [1024,1024,512]   repair-box margin
NUC_COMPETITION_FACTOR      4        pooling factor for the flood
NUC_RATIO                   [4,8,8]  one 80 nm source voxel -> 256 mip-0 tags
ABISS_NUC_DOMINANCE         0.6      share of tagged mass one identity must own for PROPER
ABISS_NUC_MIN_TAGGED        50 (win144)  /  1024 (native96)   <-- differs between arms, see R4
```

Cost function is `1 - min(channel affinity)`, 6-connected, and a seed survives pooling only when
the pooled block unanimously belongs to one nucleus. Every adjudicated territory receives a
deterministic nucleus-scoped id above `2^60`; the unadjudicated residue deliberately retains the
parent watershed id. The flood geometry stays inside its parent, but publication is **not** a
refinement-only operation: owner canonicalization gives every qualified single-owner watershed
segment for one nucleus the same id. In the measured native96 publication, 77 source segments
collapse to 15 labels (up to 7:1). If one nucleus mask incorrectly spans two genuinely different
cells, that naming consolidation silently fuses them, outside the reach of a region-graph
cannot-link.

---

## 2. What is measured (evidence, not claims)

All NERL is the standard now recorded in `dev/zebrafinch/MANUAL.md` § "Evaluation standard":
funlib-matched, **voxel units**, `test_50_skeletons.h5` (50 skeletons / 500,845 nodes), bbox
`[0,0,0,10664,10912,5700]`, reported at **both** merge thresholds. FFN reference: `mt50 = 0.538003`,
`mt0 = 0.525766`.

| arm | contact units | mt=50 | mt=0 | multi-owner node frac |
|---|---:|---:|---:|---:|
| win144 naive | — | 0.444376 | 0.304262 | 0.262920 |
| win144 nucleus | 9 | 0.468030 | 0.318718 | 0.227891 |
| native96 naive | — | 0.469513 | 0.267679 | 0.305430 |
| native96 nucleus | 8 | 0.481614 | 0.287184 | 0.263465 |
| arm2mix r10 nucleus | **0** | 0.014443 | — | 0.041 |

Nucleus deltas: win144 **+0.023654** (mt50) / **+0.014456** (mt0); native96 **+0.012100** (mt50) /
**+0.019505** (mt0). The effect is positive in all four cells and **larger at mt=0 on native96** —
the direction that matters, since it means merged mass is genuinely removed rather than shrunk
below mt=50's tolerance. `arm0_96` throughout the notes is really the **window-144** affinity;
see the corrected global note at the top of `dev/zebrafinch/lessons.md`.

Two independent contextual results the review should weigh:

- **Contamination dominates mt=0.** On win144, the *entire* 0.140 mt50→mt0 penalty is **158 stray
  nodes** in 123 merge events of 1–2 nodes each (median 1). Zeroing exactly those 158 nodes moves
  mt0 to 0.444376, i.e. exactly the mt50 score. Those nodes are 38× enriched for low affinity
  (57% have min-channel affinity < 0.1 vs 1.5% of clean nodes), consistent with
  `WS_LOW_THRESHOLD = 0.00001` letting background into segments. But an affinity threshold
  captures only ~44% of that headroom; the remaining contaminating nodes are high-affinity.
- **A local null.** On chunk `z4_y6_x1`, plain ABISS, +CC3D nucleus veto, and +competition+veto all
  scored **exactly** 0.7788734050 (mt50), while competition demonstrably separated nuclei 611/651
  inside watershed object `72057662824513538`. Never reconciled with the whole-volume gain.

---

## 3. Bug history — both found in production, not in tests

**B1 — `set -u` before conda activation.** `sbatch_nucleus_competition.sh` ran
`set -euo pipefail` before `source .../activate pytc`; conda's
`activate-binutils_linux-64.sh` reads `ADDR2LINE` unbound, so the stage died in **4 seconds**
after a 10-hour verified watershed, taking its whole agglomeration tail with it (dependencies
became unsatisfiable). Fixed by moving `set -u` after activation, matching
`sbatch_abiss_shard.sh`. Note the failure was invisible in every internal diagnostic — the job
simply exited 1 with two lines of log.

**B2 — competitive labels lost during the final remap** (`MANUAL.md`, incident 2026-08-16).
Territories were applied by `cut_chunk_agg.py` when building the RAG, but the final aggregation
remap independently ran `cut_chunk_remap.py`, reloaded the **unsplit** watershed from `WS_PATH`,
and applied the hierarchy's remap table to the original labels. The new labels existed in the
graph and not in the volume, and the writer silently collapsed the split back to the parent's
representative. Nuclei 611/651 shared a segment while `load_conflict_collisions` was zero,
owner records were distinct, and a nucleus-rejected edge had been written. **"A clean internal
collision count is necessary but insufficient."**

B2 is the important one for this review: it is a *graph-versus-volume divergence*, the class of
bug where every check you would naturally write still passes.

---

## 4. Open acceptance gates — what is NOT verified

`dev/zebrafinch/lesson_nucleus_competition.md` lists seven checks required before a run is
believed. **Only item 6 (NERL) is done.** Currently in flight (jobs 2872415–2872420):

- item 2, `nucleus_shell_contamination.py --tol 0.0` on the *materialized* whole volume — this is
  also B2's mandated acceptance test (`fused_source_pairs == 0` **and** 611/651 in distinct
  dominant segments **in the output volume**, not a chunk gate);
- item 7, per-skeleton regressions, because a positive mean can hide over-splitting.

Still unrun and unscheduled:

- item 3 — fused final segments, fused nucleus pairs, neurons participating, contact vs bridge
  reported separately, per-nucleus dominant fraction and number of segments for 90% mass;
- item 4 — proof that already-clean nuclei and material outside every repair box are **unchanged**;
- item 5 — nucleus-rejected RAG edge counts from `nuc_cuts.data` across hierarchy levels.

Indirect evidence B2 is not recurring: if competitive labels were being collapsed, the nucleus and
naive arms would score *identically* (as all three arms did on `z4_y6_x1`). They differ by
+0.012100 NERL and 0.042 multi-owner fraction, so the split does reach the volume. That is
evidence, not the required verification.

---

## 5. Efficiency — the headline problem

Measured stage wall-clock, win144 nucleus chain (`sacct`, jobs 2856392–2856399):

| stage | wall-clock | concurrency |
|---|---:|---|
| **nuccomp** | **440.6 min (7h20m)** | **1 node, 8 CPUs, serial** |
| me_L0 | 17.8 min | 80 shards × 19 cpus |
| me_L1..L5 | 28.3 min | 24→1 shards |
| remapagg | 13.8 min | 80 shards × 19 cpus |
| **total** | **500.6 min** | — |

**`nuccomp` is 88% of post-watershed wall-clock**, and on native96 it took **11h18m42s**
(2026-08-16T20:38:41 → 2026-08-17T07:57:23) to repair **8** units. Everything around it runs at
up to 1520-way concurrency; this stage runs alone.

Efficiency questions, in the order I would attack them:

1. **The stage has no phase timing at all.** Its log emits `scan instance geometry`, then
   `map 465 nuclei to watershed ids`, then one line per flood — with no timestamps. So 11 hours
   cannot be attributed between the global scan and the per-unit floods. *Assumption, not
   measurement:* the global scan dominates, because the floods are bounded boxes with factor-4
   pooling over 8 units while the scan maps 465 nuclei against the whole watershed. **First
   deliverable should be instrumentation, not optimization** — please confirm or refute this
   ordering from the code before proposing a fix.
2. **The 8–9 flood units are independent by construction** (each floods inside its own parent
   watershed object within a bounded repair box; overlapping repair scopes for one parent are a
   fail-closed error). A Slurm array of one task per unit, plus a trivial manifest merge, looks
   like the obvious win — *if* item 1 shows floods are actually a meaningful share. If the scan
   dominates, parallelize the scan by chunk instead and this is the wrong fix.
3. **The watershed is nucleus-blind, so a `NUC_*` parameter sweep should never rebuild it.**
   Verified in production this week: the naive control arm was re-submitted to consume
   matchguard's completed watershed via `WS_PATH`, because `atomic/composite/remap_chunk_ws.sh`
   read only `WS_{LOW,HIGH,SIZE,DUST}_THRESHOLD`, and every nucleus-touching script
   (`cut_chunk_agg.py`, `cut_chunk_remap.py`, `nucleus_competition.py`, `nucleus_overlay.py`) is
   agg-stage or later. That saved ~3.5 h at 1520-way concurrency. **Today a sweep over
   `NUC_MIN_SHARE` / `NUC_CONTACT_UM` / `ABISS_NUC_MIN_TAGGED` re-runs the full watershed per
   cell.** Is there a clean, safe way to make watershed reuse a first-class option rather than a
   hand-edited `WS_PATH` override? Note the constraint discovered while doing it:
   `CHUNKMAP_INPUT` cannot be redirected because `abiss_chunk.py:663-664` assigns it from
   `CHUNKMAP_OUTPUT` inside a `payload.update()` that overrides `param_overrides`, so the reuse
   had to seed its own `ws/chunkmap` with a 126 MB copy.
4. Is the 8 CPU / 128 GB request right? `MaxRSS` was not captured for either run, so the memory
   headroom is unknown.

---

## 6. Robustness questions

- **R1 — B2's regression test EXISTS; the question is whether it is strong enough.**
  `tests/unit/test_abiss_nucleus_competition.py:408`
  `test_final_aggregation_remap_replays_competition_overlay` monkeypatches
  `overlay.apply_nucleus_competition` and drives `cut_chunk_remap.py`, which is exactly the test
  the prevention list demanded. Companions cover more of that list: `:351`
  `test_one_stable_label_is_shared_by_every_segment_of_an_owner` (the shared-protected-label
  requirement), `:136` `test_seeded_flood_refines_only_the_parent_component` (the refinement
  invariant), `:275` `test_subshare_filter_only_changes_protected_owners`.
  **So do not go hunting for a missing test.** The real question: these are unit tests with a
  monkeypatched overlay and synthetic cutouts. B2 escaped because two *separately correct* code
  paths disagreed about which volume they operated on. Does a test that stubs
  `apply_nucleus_competition` actually rule out a recurrence, or does it assume the very wiring
  that broke? What is the smallest test that would have caught B2 without stubbing?
- **R2 — silent no-op.** The r10 arm found **0** multi-nucleus watershed ids, wrote a manifest with
  `repairs: []`, and produced a segmentation indistinguishable from plain ABISS — with no warning
  anywhere. Note the *correctness* of that path is tested (`:314`
  `test_no_repair_manifest_preserves_base_segmentation_and_nuclei` asserts zero repairs is a
  no-op), so this is purely an observability gap: a run that intends a nucleus intervention and
  performs none should say so loudly. Where should that assertion live — the stage, the launcher,
  or the acceptance report?
- **R3 — bounded-ROI containment.** Voxels outside a repair box keep the original id, so a
  "separated" claim is local to the box. A 2026-08-10 design review required proving global
  containment or downgrading to `separation_claim = local_only`; that was never resolved, and
  the margin is `[1024,1024,512]`. Does the manifest record enough to decide this after the fact?
- **R4 — parameter non-equivalence.** `ABISS_NUC_MIN_TAGGED` is 50 on win144 and 1024 on native96
  (a 20× stricter gate, justified in the yaml as "four source voxels prevents a resampling-edge
  speck from becoming an identity"). The two arms are therefore not the same configuration, which
  makes "the" effect size ambiguous — native96's mt50 delta is about half win144's. No sweep or
  recorded rationale ties the value to an outcome.
- **R5 — permissive untagged growth.** `NONE` objects may attach to a protected identity, which is
  needed to recover neurites, but if one untagged object bridges two identities the first accepted
  attachment wins and the second is blocked. Exclusion is preserved; ownership may be wrong. Is
  that ordering deterministic across runs and shard counts?
- **R6 — the `z4_y6_x1` null.** Three arms scoring bit-identically while competition provably
  fired needs an explanation before the whole-volume gain is trusted. Most likely local per-chunk
  scoring cannot see cross-chunk merges — but that is an assumption.
- **R7 — fail-closed coverage is partial.** The design claims five abort conditions: missing
  unanimous seeds, overlapping repair scopes for one parent, generated-ID collisions, an
  incompatible watershed manifest, and invalid nucleus coordinates. Exactly one is tested —
  `tests/unit/test_abiss_nucleus_competition.py:394`
  `test_missing_competition_manifest_fails_closed`. Which of the remaining four are reachable in
  practice, and which deserve a test rather than a claim? Silent fallback to unconstrained
  agglomeration would invalidate an experiment without failing it.

---

## 7. Deliverables

1. **A robustness review** with findings ranked by (probability of silently corrupting a result) ×
   (cost to detect). Explicitly answer R1–R7. Say plainly which are already handled — a short
   honest list beats an invented one.
2. **An efficiency plan for `nuccomp`**, ordered, starting from whatever item 5.1's
   instrumentation question actually implies. Include expected speedup and the risk each change
   poses to determinism — territory ids are deterministic and collision-checked today, and any
   parallelization must keep that property.
3. **A recommendation on watershed reuse** (5.3) as a supported mode, including whether the
   `CHUNKMAP_INPUT` constraint can be fixed properly rather than worked around.
4. **A verdict on ship-readiness** given §4's open gates, and the minimum set of them you would
   require. Disagreeing with my prioritization is useful.

## 8. Constraints and traps

- **Never quote a NERL without its convention and merge threshold.** `--nm` reads ~0.03 lower than
  the voxel convention and is not comparable to the FFN reference. Report both mt=50 and mt=0.
- The affinity store filename encodes chunk size and halo but **never the inference window**,
  which is how the win144/win96 misattribution survived for months. Genuine window-96 output is
  the sibling store with an explicit `_win48x96x96` suffix.
- Changing either the RAG-construction path or the remap path **changes the ABISS runtime hash and
  invalidates existing gates** (per B2's prevention list).
- `dev/` has no git history, so there is no diff to recover a lost edit from.
- The ws binaries are older than the nucleus sources (`build/ws2` 08-03 vs `src/seg/*` 08-13) and
  that is expected: the watershed is nucleus-blind. Do not "fix" it by rebuilding ws2 without
  understanding that a rebuild invalidates the watershed-reuse equivalence argument in 5.3.
