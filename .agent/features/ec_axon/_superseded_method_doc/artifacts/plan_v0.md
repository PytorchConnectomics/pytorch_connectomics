# Plan v0

## Summary
Author one self-contained `method.md` that reframes the axon EC decode around a **single idea** and a
**small set of durable signals**, and delete the overfit scaffolding (M1–M8 taxonomy, per-case gates,
per-example seg-ids, superseded sub-method zoo). The reframe: *a tracklet is a tube of 2D cross-sections;
convert false merges into splits first (cheap, oracle-safe), then re-link the splits (recover length)* —
one merge operator reused at growing scale, differing only in the split signal. Fold the ~8 named
sub-methods (force_split, Prob-1, Prob-2, area_step, parallel_split, branch_split, round-C) into the
**split-by-signal → merge-with-veto** frame so the reader sees the principle, not the case list.

## Scope
- IN: create `.agent/features/ec_axon/method.md` (markdown only), ~150–250 lines. Synthesize the four
  sources into a clean design: motivation, core principle, the 4-stage pipeline, the signal table, the
  metric, results, and honest limits.
- OUT: no code changes; no edits to source docs; no new seg-ids/thresholds; do not reproduce the M1–M8
  taxonomy or the failed-attempt log verbatim. No claims not supported by the sources.

## Proposed Changes
Target section outline for `method.md` (the coder should follow this shape, not copy prose):
1. **Problem & principle** (~15 lines): affinity → axon instances; errors = false merges vs false splits;
   in connectomics a split < a merge, so **raise the false-merge-free ceiling first (split), then recover
   base under it (merge)**. Tube = stack of 2D cross-sections; all splitting is per-slice 2D watershed with
   a propagated seed (never 3D/cc3d).
2. **The four stages** (one short paragraph each, growing scale — keep the v0/v1/v2/v3 names):
   - v0 base: conservative 2D-waterz + mutual-best-buddy linking (intentionally over-split).
   - v1 split: cut false merges at a signal (IoU change-point; close-ended tunnel-merge). Raises the om ceiling.
   - v2 merge: re-link z-adjacent pieces by **shape (IoU), not affinity**, mutual best-buddy, after
     **cross-section completion** (absorb 1-slice fragments that hold half a cross-section). The base-recovery win.
   - v3 weak: bridge strong tracklets across weak-affinity gaps (trajectory + shape), with a parallel/collinear veto.
3. **Signals** (a 4–6 row table): shifted-IoU (shape continuity → the merge/continue decider); 1-sided vs
   2-sided IoU (branch vs continuation); z-affinity (a floor, not the selector); trajectory collinearity;
   cross-section completion. Emphasize the durable lesson: **select by shape, not affinity — strong-aff
   seams are already merged, so residual splits live where affinity is weak.**
4. **Metric** (~10 lines): length-weighted skeleton ERL, base + oracle-merge om; the **merge-tolerance**
   fairness point (a 2-node graze must not halve a neuron's ERL → thr=10); rank limiters by length²-weighted
   ERL deficit, not raw counts.
5. **Results** table (fair yardstick) + one-line takeaway (beats v0 and waterz on both metrics).
6. **Limits** (honest, ~10 lines): the sep=0 fused-tube floor (blended centroid → IoU/affinity/trajectory
   all fail; needs membrane/orthogonal signal); 2-1-2 (both-anchored) is fixable, 2-to-1 (one disappears)
   is confounded; the om gap is diffuse, not a few big fixes.

## Files and Areas
- CREATE: `.agent/features/ec_axon/method.md`.
- READ (context): `dev/mit_liconn/PIPELINE.md`; `../pytc-agent/.../{tracklet.md, decode_3round_method.md,
  findings.md L59–L74}`. (The coder will be given the salient content in-prompt; it need not open the repo.)

## Verification Plan
- Structural: the six sections above all present; ≤ ~250 lines; a signals table exists.
- Bloat check: no per-example seg-ids (grep for the old ones — 40761, 66909, 78607, 42582 — should be
  ABSENT), no M1–M8 taxonomy block, no failed-attempt log, no unexplained magic thresholds.
- Fidelity: every quantitative claim (numbers, the thr=10 point, the split<merge rationale, the sep=0 limit)
  traces to the sources; nothing invented.
- Read test: a senior reader unfamiliar with the code can state the method, its signals, and its limits.

## Risks and Questions
- R1 Over-trimming: dropping *too* much (e.g. why per-slice 2D watershed, why mutual-best-buddy) would make
  it "too simple." Keep the *reasons*, cut the *cases*.
- R2 Stale numbers: sources mix yardsticks (911-skel vs 943; thr=1 vs thr=10). Use the **fair thr=10 / 943-GT**
  numbers from PIPELINE.md as canonical and say so; don't blend tables.
- Q1 Should `method.md` keep the file/function code-map (orchestrator → helpers)? Recommendation: a single
  compact "code map" line, not the full per-file listing (that's reference, not method).

## Changes Since Previous Plan Version
Initial plan.
