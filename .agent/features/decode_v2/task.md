# Task: decode_v2 — tube-bb + error detection + force-split via waterz propagated seeds + relink

Build an integrated LiCONN axon decode pipeline (`decode_v2`) that takes the model affinity and
produces an instance segmentation, combining the components validated in `dev/mit_liconn/`:

1. **tube-bb (base decode):** thr-0.3 2D `waterz_2d_spacefill` sections (fill 0) → link with
   `conservative_pairs(min_iou=0.2)` + `bb_pairs(0.3)` (best-buddy), NO z-contact dominance,
   NO EDT fill. This is today's winning decode (`dev/mit_liconn/tube_decode.py --stages spine,bb`),
   full-volume NERL 0.593 vs waterz 0.484.
2. **error detection (false merges):** on the RAW section overlaps, detect fused merge
   cross-sections — a section `t` at z+1 that receives ≥2 area-matched incoming sections
   `{s_i}` from z (N-to-1; axons don't branch, so area(t)≈Σarea(s_i) ⇒ a merge). Reference:
   `dev/mit_liconn/detect_parallel.py` (found 48 true 2-GT merge cross-sections; 539 total
   area-matched fusions, most same-axon over-splits).
3. **force-split via waterz propagated seeds:** split each fused section with a seeded
   watershed — markers = the incoming `s_i` footprints projected onto `t`, ridge = (1 − in-plane
   affinity); PROPAGATE the two identities down the merge run until the axons separate.
   Reference: `dev/mit_liconn/force_split_decode.py`. Do NOT membrane-gate the split (merge
   points have weak membranes — that's why they fused); the seeds drive it.
4. **relink them:** assign each split piece to its seed's tube and re-run the linking so the two
   axons stay separate; the pieces should connect up/down to their own axon without re-merging.

## Evaluation
- Primary: `dev/mit_liconn/liconn_nerl.py --eval <seg>` (NERL vs GT skeletons, res z25×y9×x9 nm).
- Ceiling probe: `--oracle-merge` (relabel each fragment to majority GT then NERL) — decouples
  split from re-link. Known: sections 0.760, force-split sections 0.780, coverage cap ~0.95.
- Instance metrics: `dev/mit_liconn/eval_inst.py` (false-merge/-split/missed, voxel P/R/F1).
- Baselines to beat: tube-bb real NERL 0.593 (merges 278 / splits 62); waterz 0.484.

## Key facts / constraints (measured this session — see dev/mit_liconn/{lesson,split_merges}.md)
- Sections at thr 0.3 are GT-pure; merges are a LINKING problem, not a model/section problem.
- Merges split into: LINK-merges (~93, sections pure → cut links; oracle-merge recovers them,
  +0.133) and IMPURE-section merges (~21/48 fused cross-sections → force-split, +0.020).
- Force-split ALONE (with today's re-link) LOWERS real NERL (0.593→0.553) by over-splitting —
  the re-link re-merges. So the **relink step is load-bearing**: it must keep force-split pieces
  separate (e.g. forbid re-linking the two pieces of a split, or only split fusions whose two
  incoming belong to independently-traced tubes).
- Coverage floor: 16.3% of GT skeleton nodes have no section (faint axons); weak-recovery
  (lower aff_bg 0.66→0.4) is a separate follow-up, out of scope here.

## Deliverable
A runnable `dev/mit_liconn/decode_v2.py` (or a clearly-named module) with `--full` + `--tag`,
that produces `outputs/mit_liconn/DL288B_crop1/decode_v2.h5` and prints NERL + instance metrics.
Reuse the existing helpers (decode_lib, force_split_decode, eval_inst, liconn_nerl) — do not
re-implement waterz/linking. Small, surgical, matches existing dev/mit_liconn style.

## Success criteria
- Runs end-to-end on the full 800³ volume via SLURM (short partition, ~120G).
- Real NERL ≥ tube-bb 0.593 (goal: exceed it by keeping force-split gains, i.e. the relink must
  not give back the merges). Report NERL, oracle-merge NERL, and instance metrics.
- If the relink can't yet beat 0.593, the pipeline must at least (a) not regress and (b) expose
  the relink as the tunable knob, with the oracle-merge NERL (≥0.78) documented as the ceiling.

## Environment
env `pytc` (`source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`); data in
`datasets/mit-liconn/`; cached sections `outputs/mit_liconn/DL288B_crop1/sections_thr0.3_z0-800.h5`.
