# Plan v1

## Summary

Both design docs converge and both the user correction and the Codex plan review
push the same revision, so plan_v1 changes the substrate assumption and the
acceptance metric:

- **The `decode_v1` base is NOT deleted.** All 600 `{key}_decode_v1.h5` (dataset
  `main`, `(1008,1008,1008)` uint32) exist in `dev/zebrafinch/results/`; the
  `decode_v1.chunks/` symlinks are merely stale (they point at `../{key}…` but the
  files moved to `../results/{key}…`). **No regeneration** — just re-point the
  symlinks.
- **Acceptance = whole-volume `test_50_skeletons` NERL**, not block-cropped. With
  the full base present this is cheap, and it is what the task requires.
- **Architecture: the linker emits a union-find *remap/LUT* keyed by
  `(chunk_key, local_label) → global_id`, and evaluation streams the existing
  per-chunk scorer over that remap — no dense 1008³×600 assembly.**

This CCC run still delivers **Step 1: a whole-volume global big-branch
endpoint-continuation linker on the existing `decode_v1` base, under a
nucleus-firewalled union-find, judged by whole-volume test-50 NERL — realized
base must rise, merge-oracle must stay flat (equal within a tight tolerance).**

Step ladder (each a separate CCC run) is unchanged except Step 2 is renamed
(the nucleus firewall is mandatory already in Step 1):

1. **This run** — whole-volume big-branch endpoint linker + nucleus firewall +
   remap-based whole-volume NERL eval.
2. Soma-root / identity-prior integration (soma watershed as scaffold; boost
   same-basin links, defer contested-basin) — the firewall is already in Step 1.
3. Targeted local corrections (cc-cut / local bg-fill floor / multi-plane crack).
4. Component-wise weak-coverage recovery (owner-count rule).
5. Tuning / operating-point sweep + certificate audit at full scale.

## Scope

**In scope (code_v0):**

1. **Substrate (mechanical, no regeneration).** Re-point
   `dev/zebrafinch/decode_v1.chunks/chunk_{key}.h5 → ../results/{key}_decode_v1.h5`
   for all 600 keys (and the output-tree `…decode_v1.chunks/` if a scorer needs
   it). Sanity-check by re-scoring 2–3 chunks with the existing per-chunk
   evaluator and confirming base/oracle/missing/n_skel/gt_len match the recorded
   `dv1eval/{chunk}.csv` rows within a justified deterministic tolerance
   (decode_v1 is deterministic; expect exact or near-exact).
2. **Big-branch node extraction over all 600 chunks.** Run/extend
   `big_branch_extract.py` on each `results/{key}_decode_v1.h5` →
   `crossings/{key}.npz`, one row per (segment, face) crossing with: global
   centroid, cross-section footprint + area, PCA inward tangent (unit, oriented
   into the chunk), elongation, `perp` cleanness, local caliber proxy, and
   **nucleus membership** (the set of `yl_cb` marker IDs contained in that
   segment). Reuse `faces/{key}.h5` boundary planes where they already provide
   the footprint to avoid re-reading full volumes.
3. **Cross-face endpoint-continuation candidate generation (not overlap-gated).**
   For each internal shared face between adjacent chunks, propose candidate edges
   between oppositely-facing big-branch crossings whose footprints fall within a
   physical search radius `r_s` (default ≈ one caliber, ~200 nm) of each other on
   the face. Score (all features normalized to [0,1]):
   `s = wI·IoU_face + wT·tangent + wC·caliber + wP·perp − wA·ambiguity`
   with `tangent = max(0, −cos∠(t_A, t_B))` (the two inward tangents should be
   anti-parallel across the face = colinear continuation; **polarity-aware**),
   `caliber = min(a,b)/max(a,b)` on cross-section area, `IoU_face` = boundary
   footprint overlap on the shared plane, `perp` = both crossings are clean exits
   (not grazes), `ambiguity` penalizes a second comparable partner. **Boundary
   overlap is one weighted cue, not a hard `min-ov` gate** — a strong-tangent,
   caliber-matched, low-overlap thin continuation must remain eligible (this is
   what distinguishes an endpoint linker from per-face IoU stitching, L44/L47).
   Defaults: `wI,wT,wC,wP,wA` and hard floors `tau_iou_min≈0`, `tau_tangent≈0.5`,
   `tau_score` set on validation; deterministic tie-break by
   `(−score, chunk_key, local_label)`.
4. **Nucleus-firewalled union-find (marker-SET semantics).** Node id =
   `(chunk_key, local_label)`; each root stores a *set* of `yl_cb` marker IDs.
   `can_link(C,D,e)` requires: `|markers(C) ∪ markers(D)| ≤ 1`; edge kind is
   endpoint-continuation; mutual-best on the face; `score ≥ tau_score`;
   caliber/tangent floors pass. Commit in descending score. **Reject (record as
   conflict), never merge, any edge whose union would hold ≥2 markers** — this
   also blocks the transitive `nucleus A → unmarked X → nucleus B` path (once
   A∪X holds A's marker, linking X→B is refused). Pre-existing decode_v1 segments
   that already contain ≥2 markers are **quarantined** (flagged, excluded from
   linking) and reported; do not silently absorb them. Deferred/ambiguous edges
   are **logged to a certificate table, not dropped.**
5. **Remap output.** Emit `graph_link_remap.npz`:
   `keys[(chunk_key, local_label)] → global_id`, plus a certificate table
   (columns: face, u=(key,label), v=(key,label), IoU, tangent, caliber, perp,
   score, mutual_rank, commit_order, markers_u, markers_v, decision,
   reject_reason). Background label 0 is preserved and never remapped.
6. **Whole-volume evaluation via the remap.** Extend `oracle_stitch_decode_v1.py`
   (now runnable — symlinks fixed) to accept an optional `--remap
   graph_link_remap.npz` and apply it while streaming each chunk through
   `oracle_cc3d_chunked.sample_variant`. Report, on the 600-box-cropped
   `test_50_skeletons.h5`, **base** and **merge-oracle** NERL for (a) decode_v1
   baseline and (b) linked. Reuse the exact existing config (`RES=[10,10,10]`,
   `branch_merge` oracle, `merge_threshold=1`, break/length 1000, canonical
   crop). No new NERL code.

**Out of scope (later runs):** changing the decode_v1 recipe or bg-fill; the
soma-root prior, cc-cut/bg-fill/multi-plane corrections, weak-coverage recovery;
any git commit; touching the 24 pre-existing unrelated dirty files.

## Proposed Changes

1. `dev/zebrafinch/graph_link_whole.py` (new): the whole-volume linker. Reads
   `crossings/*.npz` + `faces/*.h5`, builds cross-face candidates, runs the
   firewalled union-find, writes `graph_link_remap.npz` + `graph_link_certs.csv`.
   Flags: `--markers yl_cb_80nm.h5`, `--r-search`, `--wI/--wT/--wC/--wP/--wA`,
   `--tau-tangent`, `--tau-score`, `--out`. Prints an assert that no global_id
   holds ≥2 markers + counts (accepted/deferred/conflict/quarantined).
2. `dev/zebrafinch/big_branch_extract.py`: **additive only** — add the nucleus
   membership tag and caliber proxy to the `.npz` (new keys, backward-compatible);
   add a batch/`--all` or a thin driver to run all 600 (or a separate
   `run_big_branch_all.py`). Do not change existing keys.
3. `dev/zebrafinch/oracle_stitch_decode_v1.py`: **additive** `--remap` option that
   applies a `(chunk_key, local_label)→global_id` LUT during streaming; default
   behavior (no `--remap`) unchanged.
4. `dev/zebrafinch/fix_decode_v1_symlinks.py` (or documented one-liner): re-point
   the 600 `decode_v1.chunks/` symlinks to `../results/`.
5. `dev/zebrafinch/graph_link_whole.README.md`: chosen defaults, the symlink fix,
   the baseline-reproduction numbers, and the acceptance table.
6. No changes to `local_nerl_all_chunks.py`, `oracle_cc3d_chunked.py`, `em_erl`,
   `decode_v1_chunk.py`, `global_link.py` (kept as the 2-chunk reference).

## Files and Areas

| Path | Change |
|---|---|
| `dev/zebrafinch/graph_link_whole.py` | new — candidate gen + firewalled UF + remap/cert output |
| `dev/zebrafinch/big_branch_extract.py` | additive — nucleus tag + caliber (new npz keys); batch driver |
| `dev/zebrafinch/oracle_stitch_decode_v1.py` | additive — `--remap` streaming apply |
| `dev/zebrafinch/fix_decode_v1_symlinks.py` | new — re-point 600 symlinks to results/ |
| `dev/zebrafinch/graph_link_whole.README.md` | new — defaults, symlink fix, baseline + acceptance numbers |
| (reused, unchanged) | `faces/*.h5`, `crossings/*.npz`, `global_link.py` (ref), `local_nerl_all_chunks.py`, `oracle_cc3d_chunked.py`, `em_erl` |
| (data, read-only) | `results/*_decode_v1.h5` (600), `test_50_skeletons.h5`, `chunk_gt_skel/`, `yl_cb_80nm.h5`, index JSON |

## Verification Plan

Success = a whole-volume NERL table (decode_v1 baseline vs linked).

1. **Substrate control.** After the symlink fix, re-score 2–3 chunks with the
   existing evaluator; base/oracle/missing/n_skel/gt_len must match the recorded
   `dv1eval/{chunk}.csv` (deterministic → expect exact/near-exact). Mismatch ⇒
   stop (wrong substrate).
2. **Firewall safety (hard, must pass).** Assert no `global_id` in the remap
   contains ≥2 distinct `yl_cb` markers; report quarantined multi-marker base
   segments and conflict-edge count. Include a unit check of the transitive
   `A→unmarked→B` rejection.
3. **Oracle stays flat (hard gate).** Whole-volume merge-oracle of the linked
   result must equal the decode_v1 baseline oracle **within a tight tolerance**
   (state it, e.g. |Δ| ≤ 0.002). A drop = a new false merge → fail. An
   *increase* is anomalous for a pure join → investigate sampling/namespace, do
   not auto-pass.
4. **Realized gain (the win).** Whole-volume **base** NERL of the linked result
   `>` decode_v1 baseline base; report Δ and the decode_v1 oracle ceiling (~0.906,
   L43) as context. This is the tiling lever the per-face stitchers could not
   realize (~4% of ceiling, L44) — a clear positive Δbase at flat oracle is the
   Step-1 win.
5. **Cue quality (secondary).** On internal GT-crossing faces, the
   `global_link.py`-style precision/recall of committed links; note precision vs
   sparse GT is a lower bound (L47) — the decisive metric is #3/#4.

Commands (recorded in the README): `python fix_decode_v1_symlinks.py`;
`python big_branch_extract.py --all` (or the driver); `python graph_link_whole.py
--markers …/yl_cb_80nm.h5 --out results/graph_link_remap.npz`; `python
oracle_stitch_decode_v1.py --remap results/graph_link_remap.npz`.

Reviewer (code) focus: (a) firewall correctness incl. transitive and
pre-existing-multi-marker cases; (b) remap application in the streamed scorer is
consistent, namespaced by `(chunk_key, local_label)`, preserves bg 0, no
double-counting; (c) oracle computed with `branch_merge` exactly as the existing
evaluator; (d) marker coordinate mapping (`yl_cb_80nm.h5` axis order / resolution
/ origin → 1008 grid + chunk offset) is correct; (e) no edits to decode_v1, the
em_erl core, or unrelated dirty files.

## Risks and Questions

- **Marker coordinate mapping.** `yl_cb_80nm.h5` is at 80 nm; base is 10 nm on the
  1008 grid. The 8× scale + axis order + origin must be converted correctly or the
  firewall attaches markers to the wrong segments. Confirm against
  `soma_recon_wholevol.py`, which already consumes `yl_cb_80nm.h5`. (Open question
  for review: is `yl_cb_80nm.h5` or `yl_cb_80nm_neuron.h5` the right marker set?)
- **I/O over 600 chunks.** big-branch extraction reads thin boundary slabs (not
  full volumes) and can reuse `faces/`; still, 600 × slab reads — parallelize
  (the existing `parallel_faces.py` pattern) and cache `crossings/`.
- **Endpoint vs face-overlap balance.** Making overlap a soft cue risks admitting
  a wrong low-overlap continuation; mitigated by the tangent polarity floor +
  caliber + mutual-best + firewall, and validated by the oracle-flat gate. If the
  oracle drops, tighten `tau_tangent`/`tau_score` — the operating point is set by
  the NERL curve, not guessed.
- **Whole-volume oracle sensitivity.** A single cross-neuron link among the 600
  chunks can lower the oracle measurably; the certificate table lets us find and
  cut the offending edge without discarding the run.
- **Scope size.** This is a larger increment than a block, but the eval machinery
  exists (oracle_stitch, big_branch_extract, faces/) and the new code is bounded
  to candidate-gen + firewalled UF + remap + a `--remap` hook. If code review
  finds it too large, the fallback is to gate acceptance on a predeclared
  in-bounds sub-block first (smoke test) before the 600-box run — but the
  whole-volume number remains the acceptance metric.

## Changes Since Previous Plan Version

- **Substrate:** removed decode_v1 regeneration; the 600 `results/*_decode_v1.h5`
  already exist — Step 1 now just re-points the stale `decode_v1.chunks/`
  symlinks (per user correction + verified on disk).
- **Acceptance metric:** switched from block-cropped NERL to **whole-volume**
  test-50 NERL (Codex [major] #1; block is now smoke-test only).
- **Architecture:** linker emits a `(chunk_key, local_label)→global_id`
  **remap/LUT** consumed by a streamed scorer (extend `oracle_stitch_decode_v1.py`
  `--remap`); no dense 1008³×600 assembly (Codex [major] #2/#8).
- **Algorithm:** replaced the overlap-gated matcher with **polarity-aware
  endpoint-continuation candidate generation** — search radius, tangent formula,
  caliber, perp, soft-overlap; boundary IoU is a weighted cue, not a hard gate
  (Codex [major] #5).
- **Firewall:** per-root **marker-ID set**, union iff `|set| ≤ 1`, explicit
  transitive `A→unmarked→B` rejection, quarantine of pre-existing multi-marker
  segments; fixed marker source `yl_cb_80nm.h5` with a stated coordinate mapping
  (Codex [major] #6).
- **Oracle gate:** changed from `≥` to **equality within a tight tolerance**, with
  investigation on any deviation (Codex [major] #4).
- **Certificate schema** defined; **Step 2 renamed** to soma-root/identity-prior
  integration (Codex [minor] #1/#2).
- Substrate fidelity check strengthened to base/oracle/missing/n_skel/gt_len vs
  `dv1eval` (Codex [major] #7), and hero-final reuse dropped (moot).
