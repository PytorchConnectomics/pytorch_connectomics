# Plan v0

## Summary

The two design docs (`agglomeration_design_fable5.md`, `agglomeration_design_gpt5-5.md`)
converge: keep `decode_v1` as the uniform GT-free base, and the **#1 remaining
structural lever is a global big-branch *endpoint-continuation* linker** that
attacks the ~47% cross-chunk "tiling" term which per-face IoU stitching provably
cannot realize (L44: 0.9²⁹≈0.05, ~4% of the 0.906 ceiling). Everything is
committed under a **nucleus-firewalled union-find** and judged by whole-volume
`test_50_skeletons` NERL — **realized base must rise, merge-oracle must stay flat**.

We implement the design **one evaluable step at a time**. The step ladder (each a
separate CCC run) is:

1. **This run — substrate + global big-branch endpoint linker on a GT-bearing
   block.** Regenerate the deleted `decode_v1` base for a small contiguous block,
   reproduce its baseline NERL as the control, then extract face-crossing big
   branches across *all* block faces, link them by mutual-best boundary-IoU +
   tangent/caliber/perp-exit under the nucleus firewall, relabel, and measure
   block-cropped test-50 NERL (base↑, oracle-flat).
2. Nucleus firewall + soma-root prior integration (soma watershed as identity
   scaffold; reject cross-soma links, boost same-basin).
3. Targeted local corrections (cc-cut bridge / local bg-fill floor / multi-plane
   crack), surgical not global.
4. Component-wise weak-coverage recovery (owner-count rule).
5. Scale the linker to the whole 600-box volume.

**This CCC run delivers Step 1 only.** Its success criterion is a measured
NERL table on the evaluation block showing the endpoint linker raises realized
base NERL over the un-linked decode_v1 baseline **without lowering the oracle**.

## Scope

**In scope (code_v0):**

- A block definition: a contiguous set of ≥4 adjacent GT-bearing chunks that
  includes the `z2_y5_x4 ↔ z2_y5_x5` pair used by the `global_link.py` demo (L47)
  so at least one internal face carries GT crossings. Codex picks the exact block
  from `chunk_gt_skel/` availability and adjacency; record it.
- **Substrate regeneration:** run the existing producers (`decode_v1_chunk.py`
  for GT chunks, `decode_v1_produce.py` for any GT-less block chunk) to
  materialize `{key}_decode_v1.h5` for the block and re-create the
  `decode_v1.chunks/` symlinks. Reproduce and record the per-chunk baseline
  oracle NERL; it must match the existing `dv1eval/{chunk}.csv` /
  `balance_cc3d_vs_dv1_600.csv` rows within tolerance (this proves the substrate
  is faithfully rebuilt before anything new is trusted).
- **Global big-branch endpoint linker** — a new script (build on
  `big_branch_extract.py` + `global_link.py`, do not rewrite them):
  - extract face-crossing big branches for every block chunk
    (`big_branch_extract.py` → `crossings/{key}.npz`), adding **nucleus
    membership** per branch node (which `yl_cb` marker, if any, its base segments
    contain);
  - for every shared face between adjacent block chunks, propose candidate edges
    only between face-crossing big branches; score with **mutual-best
    boundary-IoU** (primary) + tangent continuation + caliber compatibility +
    perpendicular-exit, minus nucleus conflict;
  - commit in descending structural confidence through a **nucleus-firewalled
    union-find** (`can_link`: reject if the union would hold ≥2 distinct nuclei;
    require endpoint-continuation + mutual-best + IoU/tangent thresholds); defer
    (do not silently drop) rejected/ambiguous edges to a certificate log;
  - relabel the block into a single global-id volume by union-find components.
- **Evaluation** (block-level, reusing the settled contract):
  - block-cropped `test_50_skeletons.h5` NERL via `oracle_cc3d_chunked`
    (`sample_variant` + `nerl` + `branch_merge`) on the assembled block volume:
    report **base** and **merge-oracle** for (a) un-linked decode_v1 and (b) the
    linked result;
  - GT-validated link precision/recall on the internal faces (as
    `global_link.py` already computes) as a secondary cue-quality check — but the
    **decisive metric is realized block NERL gain with oracle stability**, not
    sparse-GT edge precision (L47).

**Out of scope (later runs / not this change):**

- Changing the `decode_v1` recipe or its bg-fill (L59/L60: uniform base is fixed).
- Whole 600-box run (Step 5) — this run proves the linker on a block first
  (de-risk before scale; the tiling wall is the main uncertainty).
- Soma-root prior, targeted cc-cut/bg-fill/multi-plane corrections, weak-coverage
  recovery (Steps 2–4).
- Any git commit; any touch of the 24 pre-existing unrelated dirty files.

## Proposed Changes

1. `dev/zebrafinch/graph_link_block.py` (new): the block linker + evaluator.
   Args: `--block "<chunk keys or z0-z1,y..,x.. range>"` (or `--center z2_y5_x5
   --radius 1`), `--min-size 1e6`, `--min-ov 50`, `--tau-iou`, `--tau-tangent`,
   `--out`. Steps: regenerate-or-load decode_v1 per block chunk → assemble block
   volume → `big_branch_extract` per chunk (with nucleus tag) → cross-face
   candidate edges → nucleus-firewalled union-find → relabel → NERL(base,oracle)
   before/after → write linked block h5 + a certificate/edge log + a metrics row.
   Prints a `RESULT:` line (baseline base/oracle, linked base/oracle, Δ).
2. Small additive helper in/near `big_branch_extract.py` for **nucleus
   membership per branch** if not already derivable (read `yl_cb_80nm.h5`, map
   markers to block base segments). Prefer a new function over editing the
   existing extractor's output contract; if the `.npz` schema must grow, keep it
   backward-compatible (new keys only).
3. A tiny driver/notes file `dev/zebrafinch/graph_link_block.README.md` (or a
   docstring) recording the chosen block, the regeneration commands, and the
   baseline-reproduction numbers, so the step is reproducible.
4. No changes to `decode_v1_chunk.py`, `decode_v1_produce.py`,
   `local_nerl_all_chunks.py`, `oracle_cc3d_chunked.py`, or the `em_erl` package.

## Files and Areas

| Path | Change |
|---|---|
| `dev/zebrafinch/graph_link_block.py` | new — block substrate + endpoint linker + NERL eval |
| `dev/zebrafinch/big_branch_extract.py` | additive only if needed (nucleus tag helper, new npz keys) |
| `dev/zebrafinch/graph_link_block.README.md` | new — chosen block, regen commands, baseline numbers |
| (reused, unchanged) | `global_link.py`, `decode_v1_chunk.py`, `decode_v1_produce.py`, `local_nerl_all_chunks.py` (`chunk_nerl`, `crop_split`), `oracle_cc3d_chunked.py` (`sample_variant`/`nerl`/`branch_merge`), `em_erl` |
| (data, read-only) | `AFF_DIR …grid1008_halo72.h5.chunks/`, cc3d `…cc3d_t066_tissue.chunks/`, `test_50_skeletons.h5`, `chunk_gt_skel/`, `yl_cb_80nm.h5`, index JSON |

## Verification Plan

Success = a measured NERL table on the evaluation block. Concretely:

1. **Substrate control.** After regenerating decode_v1 for the block, run the
   existing per-chunk eval and confirm the baseline oracle matches
   `dv1eval/{chunk}.csv` / `balance_cc3d_vs_dv1_600.csv` for those chunks (within
   ~±0.01, allowing RNG/rounding). If it does not match, stop — the substrate is
   not faithfully rebuilt and no linker result is trustworthy.
2. **Linker merge-safety (the hard gate).** Block-assembled merge-oracle NERL of
   the linked result **must be ≥** the un-linked decode_v1 block oracle (oracle
   flat = zero new false merges; L33/L50). A drop means the linker introduced a
   cross-neuron merge → fail.
3. **Linker realized gain (the win).** Block-assembled **base** NERL of the
   linked result **> un-linked** decode_v1 base (the cross-chunk joins recovered
   backbone length). Report the delta.
4. **Cue quality (secondary).** `global_link.py`-style GT precision/recall of the
   committed links on internal GT-crossing faces; expect high recall on the true
   crossings and note that "precision" against sparse GT is a lower bound (L47).
5. **Firewall sanity.** No linked component contains ≥2 distinct `yl_cb` nuclei
   (assert); count deferred/conflict edges.

Run commands (recorded in the README):
`python dev/zebrafinch/decode_v1_chunk.py --chunk <k>` (per block GT chunk),
`python dev/zebrafinch/graph_link_block.py --center z2_y5_x5 --radius 1 --out …`,
then inspect the printed `RESULT:` table.

Reviewer (code) focus: (a) the nucleus firewall is correct and can never merge
two nuclei; (b) the block NERL eval assembles cross-chunk labels consistently
(no double-counting, offsets correct per `chunk_offset`); (c) oracle is computed
with `branch_merge` exactly as the existing evaluators; (d) no edits to the
decode_v1 base or the em_erl core.

## Risks and Questions

- **Substrate regeneration cost.** decode_v1 is ~15–25 min/chunk; a 2×2×2 block
  is ~8 chunks. Mitigation: pick the smallest block that still has an internal
  GT-crossing face (could be as few as 2–4 chunks); reuse the ~8 existing
  `results/decode_*_final.h5` hero chunks where they coincide with the block.
- **Small-block tiling signal.** On a tiny block a neuron crosses few internal
  faces, so the realized gain may be modest even if the linker is correct; the
  point of Step 1 is to prove *merge-safety + a positive gain + the harness*, not
  to realize the full 47% (that needs Step 5 scale). State this so a small Δbase
  is not misread as failure.
- **Nucleus markers source.** Assume `yl_cb_80nm.h5` (the soma markers
  `soma_recon_wholevol.py` uses); `test_50_nuclei.txt` may be an alternative.
  Codex should confirm which maps cleanly onto the block base segments.
- **Assembly correctness.** Cross-chunk relabel must use the index-JSON offsets
  and the same crop/break config (`RES=[10,10,10]`, `--break/length-nm 1000`,
  dedup GT) as the existing evaluators, or the NERL will be silently wrong
  (L27/L46 phantom traps). Reuse `crop_split` / `oracle_cc3d_chunked`, do not
  reimplement.
- **Question for plan review:** is a block-level `oracle_cc3d_chunked` assembly
  the right eval primitive, or should we instead extend `oracle_stitch_decode_v1.py`
  (currently blocked on the deleted chunks) to accept a linker remap? Either is
  acceptable; the plan prefers a self-contained block eval to avoid coupling to
  the whole-volume path this run does not need.

## Changes Since Previous Plan Version

Initial plan.
