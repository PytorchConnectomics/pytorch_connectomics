# Plan v2

## Summary
`dev/mit_liconn/decode_v2.py`: tube-bb base → detect exactly-2-incoming area-matched fusions →
force-split ELIGIBLE fusions (upstream-independent tubes) with propagated seeds, tracking global
node identity, ownership, mandatory same-side continuity, and terminal assignment → relink with a
seed-anchored constrained union-find that guarantees BOTH separation (no component holds both sides
of a run) AND connectivity (each piece is unioned to its seed). Emits candidate + ablation +
SELECTED (no-regression) artifacts, prints real NERL / oracle-merge NERL / instance metrics
in-script, and ships an end-to-end `--self-test`. Every plan_v1 review finding is pinned below.

## Scope
Unchanged from v1: new `decode_v2.py` + `sbatch_decode_v2.sh`; one minimal extraction in
`force_split_decode.py`. Out of scope: weak-recovery, model/GT/affinity changes, git commits,
N>2 fusion splitting (skipped+counted), any other refactor.

## Proposed Changes
**Global node identity (fix #1).** `waterz_2d_spacefill` assigns per-slice-offset labels, so every
section label is already volume-unique. decode_v2 ASSERTS this at load (no label appears in two
slices; abort with a clear error otherwise) and uses the integer label as the single global node id
everywhere (edges, `owner`, run side sets, relabel). New split pieces get fresh ids `>= current
max+1`. No `(z,label)` tuples needed once the invariant holds; a `z_of[label]` map is kept for
geometry only.

**Stage 1 — base + spine tubes.** `segs_to_iou` → IoU edges. `conservative_pairs(0.2)` → spine
union-find `tube_of`; `+bb_pairs(0.3)` → tube-bb reference seg `A`.

**Stage 2 — fusion detection.** `incoming[t] = {s : IoU(s,t) >= 0.2}`. Candidate = `t` with EXACTLY
2 incoming `s1,s2` and `|area(t)-(area(s1)+area(s2))| <= area_tol*area(t)`. N>2 skipped+counted.

**Stage 3 — upstream-only eligibility (fix #2).** For candidate `t` (at z), trace UPSTREAM from
`s1` and `s2` separately: follow spine links toward decreasing z, WITHOUT traversing `t` or any
link incident to `t`, collecting up to `min_len` sections each. ELIGIBLE iff the two upstream
traces are node-DISJOINT and each reaches `>= min_len` sections (default 3). This tests two
independently-traced incoming tubes, excluding the ambiguous transition. Report
eligible/ineligible/N>2 counts.

**Stage 4 — force-split eligible fusions (fix #3,#4).** Extract into
`force_split(seg2d, affxy, incoming, eligible_runheads, area_tol) -> (split_seg, runs)`:
- Deterministic run-head order `(z, t)`. Global `owner: label->run_id`.
- Start a run only if BOTH `s1,s2` and `t` are UNOWNED; else deterministic conflict-SKIP (counted)
  — this prevents stale seeds (an owned section can never be another run's seed).
- Seeded watershed on `t` (markers = `s1,s2` footprints from z-1; ridge `1-mean(aff[1:],0)`) →
  fresh pieces `l1,l2`; mark `t,l1,l2` owned. Propagate down while the single overlapping next
  section `nt` is UNOWNED, area-matched to the two current seeds, and not itself a 2-incoming
  fusion; split `nt` → `l1',l2'`, mark owned, continue.
- On stop, the last still-fused-but-unsplit or the naturally-separated TERMINAL sections are
  assigned to the side (A/B) they most overlap and added to that side set.
- Each run returns: `sideA`, `sideB` (label sets incl. the seed anchors, all pieces, terminals);
  `mand_links` = must-union pairs threading each side (seed→piece→piece…) so pieces STAY connected
  to their seed tube.

**Stage 5 — constrained union-find relink (fix #4).**
- Nodes = all labels in `split_seg`. Component state: a dict `run_id -> 'A'|'B'`.
- Seed component states from side sets: each `sideA[r]` label → `{r:'A'}`, `sideB[r]` → `{r:'B'}`.
- FORCED unions first: apply every `mand_links` pair (guarantees each piece connects to its seed;
  these never conflict by construction).
- Then candidate links = `conservative_pairs(0.2)+bb_pairs(0.3)` on `split_seg`, processed in
  order `(-IoU, min(u,v), max(u,v))` (fix minor tie-break). For `(u,v)`: roots `ru,rv`; REJECT if
  merging their run→side dicts conflicts (same `run_id`→different side); else union+merge dicts.
- Relabel `split_seg` by final components → candidate seg `C`.

**Stage 6 — metrics + no-regression selection (fix #5,#6).** In-script (import eval_inst,
liconn_nerl), for a=tube-bb, b=force-split+PLAIN relink (no constraint), c=`C`:
- print `eval_inst.score`, real NERL, oracle-merge NERL.
- Also evaluate oracle-merge NERL on the UNLINKED `split_seg` substrate and assert it `>= 0.78`
  (binds the ceiling check to the correct artifact, fix #6).
- Save `decode_v2_a_tubebb{tag}.h5`, `_b_plain`, `_c_constrained`.
- SELECTED `decode_v2{tag}.h5` = argmax(real NERL) over {a, c}. Guarantee: selected NERL
  `>= recomputed(a)` AND `>= 0.593`. Candidate (c) NERL and selected NERL both printed; if c<a the
  banner says "candidate regressed; selected=tube-bb (no-regression)". GT-based selection is
  labelled EVAL-ONLY (a GT-free selector is future work).

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_v2.py` | NEW — stages 1-6, constrained UF, `--self-test`, in-script metrics, selection, CLI |
| `dev/mit_liconn/force_split_decode.py` | MINIMAL — add `force_split(...)` (single signature above) returning split_seg + runs(sideA/sideB/mand_links/terminal) + owner-based ordering; keep `main` calling it |
| `dev/mit_liconn/sbatch_decode_v2.sh` | NEW — `-p short -c8 --mem120G`; `source .../activate pytc`; run `--full`; echo exact output paths |
| reused unchanged | decode_lib, eval_inst, liconn_nerl, endpoint_link |

## Verification Plan
1. **`--self-test` (no data, <1s)**, asserts and non-zero-exits on failure:
   - union-find: direct A–B rejected; indirect A–X–B rejected; two runs — cross-run allowed,
     same-run cross-side rejected; same-side allowed; tie-break determinism.
   - split path on synthetic 3D section graphs: (i) stale-seed — two interacting runs sharing a
     section → second run conflict-skipped, no stale reference; (ii) consecutive/overlapping runs;
     (iii) terminal assignment adds terminal to correct side; (iv) repeated cross-slice label →
     assertion fires; (v) end-to-end split→relink→CONNECTIVITY: every piece's final component
     contains its seed anchor (fix #4) AND no component holds both sides of a run.
2. **Pilot (instance-only):** `source activate pytc; python dev/mit_liconn/decode_v2.py --zslice 0:96`
   — `--zslice` reports INSTANCE metrics only (NERL needs the full-volume skeleton graph; stated in
   help); prints fusion eligible/ineligible/N>2 counts; writes `decode_v2_z0-96.h5`.
3. **Full (SLURM):** `sbatch dev/mit_liconn/sbatch_decode_v2.sh` → a/b/c real NERL, oracle-merge
   NERL, instances/FM-gt/FS-gt/missed; unlinked-substrate oracle-merge NERL. Hard checks: selected
   real NERL `>= max(recomputed a, 0.593)`; unlinked substrate oracle-merge `>= 0.78`. Reporting-
   only: c vs b (constraint value), FM-gt(c) vs 278, instance count(c).
4. Results appended to `dev/mit_liconn/split_merges.md`.

Success = `--self-test` passes (incl. connectivity + stale-seed cases); full run completes;
selected artifact non-regressing (guaranteed); substrate oracle-merge `>= 0.78`; c > b on real NERL.

## Risks and Questions
- Conflict-skip may drop some interacting merges (counted); acceptable — correctness over coverage.
- Upstream-only eligibility can still mis-classify (one axon traced as two tubes → false-eligible);
  the constrained UF preserves such a split (minor real-decode split cost, none for oracle-merge).
- GT-based selection is eval-only; a GT-free selector (e.g. calibrated confidence) is out of scope.
- Slice-mode gives instance metrics only by design; full-volume NERL is the primary gate.

## Changes Since Previous Plan Version
Pins every plan_v1 review finding:
- #1 node identity: assert global-unique section labels (true by `waterz_2d_spacefill` construction)
  and use the label as the single global node id everywhere; abort if the invariant fails.
- #2 eligibility: replaced whole-graph `tube_of` with an UPSTREAM-ONLY disjoint-trace test that
  excludes the fusion transition; `--min-len`.
- #3 stale seeds: global `owner` map + run only if seeds+t unowned else deterministic conflict-skip
  (counted); owned sections can never be another run's seed.
- #4 continuity + terminal: `mand_links` FORCED same-side unions guarantee each piece connects to
  its seed; terminals assigned to their best-overlap side; `--self-test` asserts connectivity.
- #5 fallback: separate candidate (c) vs SELECTED = argmax NERL{a,c}; check selected vs recomputed
  base AND absolute 0.593; GT-based selection flagged eval-only.
- #6 oracle threshold: bound `>=0.78` to the UNLINKED force-split substrate, evaluated separately.
- #7 verification: `--self-test` now covers stale seeds, overlapping runs, terminal assignment,
  repeated-label assertion, and end-to-end split→relink connectivity.
- #8 CLI/slice: `--tag` (str, appended to all output filenames), exact ablation filenames,
  `--zslice` = instance-metrics-only (documented), `--full` = NERL+oracle-merge+instance.
- minors: `(-IoU, min-id, max-id)` tie-break; commands activate `pytc`; single settled `force_split`
  signature (`incoming`, not `iou_edges`).
