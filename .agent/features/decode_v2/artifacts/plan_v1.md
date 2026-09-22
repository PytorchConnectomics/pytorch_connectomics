# Plan v1

## Summary
Add `dev/mit_liconn/decode_v2.py`: tube-bb base → detect fused merge cross-sections → seeded
watershed force-split (propagated) of ELIGIBLE fusions only → **relink with a seed-anchored
constrained union-find** that provably keeps each split axon separate (rejects any union that
would place both sides of a split run in one component, including indirect A–X–B paths). Fusion
eligibility (two incoming trace to two distinct substantial spine-tubes) limits splits to
likely-true merges so the hard constraint does not convert same-axon over-splits into permanent
false splits. decode_v2 prints all three metrics (real NERL, oracle-merge NERL, instance metrics)
for three ablations in-process, and guarantees no regression via a tube-bb fallback. Includes a
`--self-test` of the constrained union-find invariant on synthetic graphs.

## Scope
In scope: new `dev/mit_liconn/decode_v2.py` (pipeline + constrained union-find + `--self-test` +
in-script metrics), `dev/mit_liconn/sbatch_decode_v2.sh`, and a MINIMAL extraction in
`force_split_decode.py` of the detect+seeded-split+propagate loop into a reusable
`force_split(seg2d, affxy, iou_edges, eligible_fusions, area_tol) -> (split_seg, runs)` where each
`run` carries globally-unique side id sets. Reuse decode_lib/eval_inst/liconn_nerl/endpoint_link.
Out of scope: weak-recovery/hysteresis (separate thread), model/GT/affinity changes, git commits,
N-way (>2) fusion splitting (skipped + counted), any refactor beyond the one extraction.

## Proposed Changes
Stages on the thr-0.3 sections (load cached `sections_thr0.3_z0-800.h5`, else build):

1. **tube-bb base + spine tubes.** Compute `conservative_pairs(0.2)` → spine union-find →
   `tube_of` (used for eligibility) and, with `bb_pairs(0.3)` added, the tube-bb reference seg.

2. **Detect fusions.** `incoming[t] = {s at z : IoU(s,t) >= 0.2}` from `segs_to_iou`. A fusion
   candidate = `t` with **exactly 2** incoming `s1,s2` (N>2 skipped and counted) and
   `|area(t) - (area(s1)+area(s2))| <= 0.5*area(t)`.

3. **Eligibility (addresses review #2).** A fusion is ELIGIBLE iff `tube_of[s1] != tube_of[s2]`
   AND both spine-tubes have `>= min_len` sections (default 3). This restricts splitting to
   likely-true merges (two independently-traced tubes fusing). Ineligible fusions are NOT split
   (left as-is). Report eligible / ineligible / N>2-skipped counts. (Imperfect eligibility is
   tolerated: the constrained union-find below preserves any false-positive split.)

4. **Force-split ELIGIBLE fusions with propagated seeds (addresses #3,#4,#5).** Process eligible
   fusions as run-heads in deterministic order `(z, t)`, tracking a global `processed` set of
   section labels; skip a run-head whose section is already processed by an earlier run. For a
   run: seeded watershed on `t` (markers = footprints of `s1,s2` projected from z-1; ridge
   `1 - mean(aff[1:], axis=0)`), producing pieces with NEW globally-unique labels `l1,l2`;
   propagate downward while the single overlapping next section is area-matched to the two seeds
   and unprocessed, carrying `l1,l2` as seeds; stop at a 2-incoming section or natural separation.
   The run records `sideA = {s1, all l1 pieces}` and `sideB = {s2, all l2 pieces}` as sets of
   **globally-unique section labels** (each section label is already globally unique per slice;
   store them directly — no slice-local ambiguity). Sections are only ever written by one run
   (processed-tracking), so side sets are never stale.

5. **Seed-anchored constrained union-find relink (addresses #1).** Build link candidates on the
   split sections: `conservative_pairs(0.2)` + `bb_pairs(0.3)`. Union-find where each component
   carries a dict `run -> side ('A'|'B')` seeded from step-4 side sets (each `sideA` label →
   `{run: 'A'}`, each `sideB` label → `{run: 'B'}`). Process candidate links in descending IoU;
   for link `(u,v)`: find roots `ru,rv`; the union is REJECTED iff merging their run→side dicts
   conflicts (same `run` maps to different side); else union and merge dicts. This guarantees
   no component ever contains both sides of any split run, via any direct OR indirect path.
   Relabel `seg` by the resulting components → `decode_v2` segmentation.

6. **No-regression fallback (addresses #6).** Compute real NERL (via `liconn_nerl`) of tube-bb and
   of the split-relink. `decode_v2.h5` = the higher-NERL of the two (documented as an eval-time
   choice since it uses GT); print both so a regression is visible, never silently accepted. If
   split-relink < tube-bb, that is a NEEDS-work signal, not success.

7. **In-script metrics (addresses #7).** `decode_v2.py` itself computes and prints, for each of
   (a) tube-bb, (b) force-split + PLAIN relink [ablation], (c) force-split + CONSTRAINED relink:
   `eval_inst.score(seg, gt)` and `liconn_nerl` real NERL + `--oracle-merge` NERL, by importing
   those modules and calling in-process (each seg kept in memory). Output names:
   `decode_v2{,_zZ0-Z1}.h5` for `--full`/`--zslice`; ablation segs optionally saved with `--tag`.

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_v2.py` | NEW — 5-stage pipeline, constrained union-find, `--self-test`, in-script metrics for a/b/c, no-regression fallback, `--full/--zslice/--tag/--min-len/--area-tol` |
| `dev/mit_liconn/force_split_decode.py` | MINIMAL — extract `force_split(seg2d, affxy, incoming, eligible, area_tol)->(split_seg, runs)` with side sets + processed-tracking + deterministic ordering; keep existing `main` calling it |
| `dev/mit_liconn/sbatch_decode_v2.sh` | NEW — `-p short -c8 --mem120G`, activates `pytc`, runs `--full`, echoes the exact output path |
| reused unchanged | `decode_lib.py`, `eval_inst.py`, `liconn_nerl.py`, `endpoint_link.py` |

## Verification Plan
1. **`--self-test` (addresses #8), runs in <1s, no data:** deterministic synthetic section graphs
   asserting the constrained union-find keeps A/B separate:
   (t1) direct A–B link rejected; (t2) indirect A–X–B rejected (X unions to A, then B–X rejected);
   (t3) two interacting runs — cross-run unions allowed, same-run cross-side rejected;
   (t4) allowed same-side unions succeed; (t5) seed anchors present after relabel; (t6) repeated
   labels across slices treated as distinct nodes. Non-zero exit on any failure.
2. **Pilot smoke:** `python dev/mit_liconn/decode_v2.py --zslice 0:96` runs clean; prints fusion
   counts (eligible/ineligible/N>2) and the a/b/c metric block; output `decode_v2_z0-96.h5`.
3. **Full volume (SLURM):** `sbatch dev/mit_liconn/sbatch_decode_v2.sh`. Record for a/b/c:
   real NERL, oracle-merge NERL, instances/false-merge-gt/false-split-gt/missed.
   Concrete checks (addresses #9,#10): (c) real NERL >= tube-bb (a) real NERL (no regression, via
   fallback); (c) false-merge-gt <= (a) 278; #instances(c) reported (expect < force-split-all's
   3600+ because only eligible fusions split); (c) real NERL > (b) plain-relink (the constraint's
   value). All numbers reporting-only except the no-regression guarantee.
4. Record results in `dev/mit_liconn/split_merges.md` run log.

Success = `--self-test` passes; full run completes; no regression vs tube-bb (guaranteed by
fallback); constrained relink (c) > plain relink (b) on real NERL; oracle-merge NERL of the split
substrate >= 0.78.

## Risks and Questions
- **Eligibility imperfection:** `tube_of` from a first spine pass may put a true merge's two axons
  in one tube (missed) or split one axon into two tubes (false eligible). Missed → that merge
  stays; false-eligible → constrained union-find keeps the (harmless-for-oracle, minor-for-real)
  split. Reported via counts; acceptable for v1.
- **Union order dependence:** greedy constrained union is order-sensitive; fixed to descending IoU
  for determinism. Not globally optimal but stable and testable.
- **Watershed without membrane:** merge points have weak membranes; seeded split uses geodesic
  distance from seeds — boundary voxels may mis-assign; relink uses IoU on pieces so small
  mis-assignment is tolerated. Flag if instance metrics show it.
- **Runtime:** eligible-only splitting is far cheaper than force-split-all (9421 secs); expect
  fewer runs; keep within SLURM time using cached sections.
- Open question for reviewer: is the `tube_of`-different-tube eligibility rule acceptable for v1,
  or should eligibility instead be "both incoming persist as separate tubes for >= K slices
  up AND down" (bidirectional)? v1 uses the simpler upstream-different-tube rule and exposes
  `--min-len`; bidirectional is a v-next option.

## Changes Since Previous Plan Version
Directly addresses every plan_v0 review finding:
- #1 relink: replaced forbidden-edge deletion with a seed-anchored CONSTRAINED UNION-FIND
  (run→side dicts per component; reject conflicting unions) — blocks indirect A–X–B paths.
- #2 same-axon contradiction: added an explicit FUSION-ELIGIBILITY rule (two distinct substantial
  spine-tubes); only eligible fusions are split; ineligible left intact; imperfect eligibility
  tolerated by the constraint.
- #3 side identity: side sets are globally-unique section labels (per-slice-unique), stored
  directly; survive relabel via the union-find node set.
- #4 run ownership: deterministic run-head order `(z,t)` + global `processed` set; no overwrites.
- #5 N-way: restricted to EXACTLY 2 incoming (area test on that pair); N>2 skipped + counted.
- #6 acceptance: added a tube-bb NO-REGRESSION FALLBACK; regression is surfaced, never "accepted".
- #7 evaluation: `decode_v2.py` prints real NERL + oracle-merge NERL + instance metrics in-process
  for ablations a/b/c; explicit output names + `--zslice` isolation.
- #8 verification: added `--self-test` graph-level invariant cases (direct/indirect/interacting/
  same-side/anchor/repeated-label).
- minors: verification activates `pytc`; SLURM echoes exact path; vague thresholds made
  reporting-only with the single hard no-regression check.
