# Plan v2

## Summary

Final plan for **Step 1: a whole-volume global big-branch endpoint-continuation
linker on the existing 600-chunk `decode_v1` base, under a nucleus-firewalled
union-find, accepted by whole-volume `test_50_skeletons` NERL — realized base up,
merge-oracle flat within numerical roundoff.** plan_v2 pins the execution
contracts Codex flagged in plan_v1: a *total* collision-free global-ID namespace,
a no-edge control, a single concrete marker-sampling stage, a corrected quarantine
invariant, a predeclared operating point plus a bounded 1-D calibration, and one
oracle tolerance.

Substrate (from the user correction, verified): the 600 `{key}_decode_v1.h5`
(dataset `main`, `(1008,1008,1008)` uint32) exist in `dev/zebrafinch/results/`;
the `decode_v1.chunks/` symlinks are stale (point at `../` not `../results/`). Fix
= re-point symlinks; no regeneration.

Step ladder (each a later CCC run): 2 soma-root/identity-prior integration; 3
targeted local corrections; 4 component-wise weak-coverage recovery; 5 full-scale
operating-point sweep + certificate audit.

## Scope

### S0 — Substrate (mechanical)
Re-point `dev/zebrafinch/decode_v1.chunks/chunk_{key}.h5 →
../results/{key}_decode_v1.h5` for all 600 keys; assert every target resolves.

### S1 — Total global-ID namespace (the remap is a total function)
Enumerate **every distinct foreground `(chunk_key, local_label)`** across all 600
chunks into a contiguous node index at load (dict). Union-find over **all** these
nodes (most remain singletons). Final `global_id` = compressed contiguous root
index, dtype `uint32` (assert node count < 2³¹). The remap
`R: (chunk_key, local_label) → global_id` is therefore **total** (defined for
untouched labels too, each mapping to its own unique id), so equal local labels in
different chunks never collide. Background label `0 → 0` always, never remapped.
The evaluator applies `R` **exactly once** per sampled voxel: voxel → local_label
→ `(key,label)` → `global_id`, for both linked-base NERL and `branch_merge`.

### S2 — Marker sampling (ONE stage, fixed source, reused mapping)
Source = `/projects/weilab/dataset/zebrafinch/yl_cb_80nm_neuron.h5` (per user: the
**corrected neuron nuclei**; verified: key `main`, `(1425,1365,1333)` uint16, 465
non-zero distinct soma IDs — labeled, not binary; same grid/shape as the
`soma_recon`-consumed `yl_cb_80nm.h5`). Map each marker voxel from 80 nm to the
10 nm 1008-grid by **reusing the exact conversion in `soma_recon_wholevol.py`**
(`POOL=[8,16,16]`, `CBSCALE=[4,8,8]`, `caff=(g*CBSCALE)//POOL`; snap-off-fg-seed
handling as there) — do **not** re-derive it (L14 alignment trap). For each marker
id, sample the decode_v1 base label at its grid location → record membership
`M: (chunk_key, local_label) → set(marker_ids)`. Write once to
`results/marker_membership.npz`. Report: the number of in-box markers in
`yl_cb_80nm_neuron.h5` and the number that land on a foreground segment (the
latter should be close to the former; a large shortfall ⇒ mapping error → stop),
plus markers that miss fg (snapped or logged). This single stage owns membership;
the linker consumes `marker_membership.npz` (extraction does not write
membership).

### S3 — Big-branch node extraction (600 chunks)
Run `big_branch_extract.py` per `results/{key}_decode_v1.h5` →
`crossings/{key}.npz`, one row per (segment, face): global centroid, cross-section
footprint + area, unit inward PCA tangent, elongation, `perp` cleanness, caliber
proxy. Reuse `faces/{key}.h5` boundary planes for the footprint where possible.
`big_branch_extract.py` changes are **additive only** (add a caliber-proxy key if
absent; new npz keys, existing keys unchanged). Batch via a thin driver
(`parallel_faces.py` pattern) and cache `crossings/`.

### S4 — Endpoint-continuation candidate generation (soft-overlap, polarity-aware)
For each internal shared face between adjacent chunks, candidate edges between
oppositely-facing big-branch crossings whose footprint centroids fall within
physical search radius `r_s = 200 nm` on the face. Features, all normalized to
`[0,1]`:
- `IoU_face` = footprint intersection/union on the aligned shared plane;
- `tangent = max(0, −cos∠(t_A, t_B))` (inward tangents anti-parallel ⇒ colinear
  continuation; polarity-aware);
- `caliber = min(area_A, area_B) / max(area_A, area_B)`;
- `perp = perp_A · perp_B` (both are clean exits, not grazes);
- `ambiguity = best_competing_pair_score / this_pair_score` for either endpoint.

Predeclared defaults (fixed): `wI=1.0, wT=2.0, wC=1.0, wP=1.0, wA=1.0`;
`score = (wI·IoU + wT·tangent + wC·caliber + wP·perp)/(wI+wT+wC+wP) − wA·ambiguity`.
Hard floors: **`tau_iou_min = 0` exactly** (no overlap gate — low-overlap strong
continuations stay eligible), `tau_tangent = 0.5` (reject anti-parallelism worse
than 60°). Candidate is kept only if it is **mutual-best** on the face for both
endpoints. Deterministic ordering / tie-break: sort by
`(−score, chunk_key_u, label_u, chunk_key_v, label_v)`.

### S5 — Nucleus-firewalled union-find (marker-set semantics; quarantine)
Node = `(chunk_key, local_label)`; each root stores a `set` of marker ids from
`M`. **Quarantine:** any base segment whose `M` set has ≥2 distinct markers is
quarantined = its own singleton, accepts no edges, is never modified; count and
report these. **`can_link(C,D,e)`** requires: neither `C` nor `D` quarantined;
edge is endpoint-continuation, mutual-best, `score ≥ tau_score`, tangent/caliber
floors pass; and **`|markers(C) ∪ markers(D)| ≤ 1`**. Commit in descending score.
Rejected/ambiguous edges are **logged to the certificate table, not dropped.** The
`|·|≤1` rule automatically blocks the transitive `nucleus A → unmarked X →
nucleus B` path (once A∪X carries A's marker, X→B is refused).

### S6 — Outputs
`results/graph_link_remap.npz` (the **total** `R` over all fg labels + the accepted
union list) and `results/graph_link_certs.csv` with columns: face, u=(key,label),
v=(key,label), IoU, tangent, caliber, perp, ambiguity, score, mutual_rank,
commit_order, **markers_u (current root set), markers_v (current root set)**,
decision, reject_reason.

### S7 — Whole-volume evaluation via the remap
Extend `oracle_stitch_decode_v1.py` with `--remap results/graph_link_remap.npz`
that applies `R` while streaming each chunk through
`oracle_cc3d_chunked.sample_variant`; default (no `--remap`) unchanged. Report on
the 600-box-cropped `test_50_skeletons.h5`, using the exact existing config
(`RES=[10,10,10]`, `branch_merge` oracle, `merge_threshold=1`, break/length 1000,
canonical crop): **base** and **merge-oracle** for (a) decode_v1 baseline and (b)
linked. No new NERL code.

**Out of scope:** decode_v1 recipe/bg-fill; soma-root prior; cc-cut/bg-fill/
multi-plane corrections; weak-coverage recovery; any git commit; the 24
pre-existing dirty files.

## Proposed Changes

1. `dev/zebrafinch/fix_decode_v1_symlinks.py` (new) — re-point + assert 600
   symlinks.
2. `dev/zebrafinch/sample_markers.py` (new) — S2 marker-membership stage; reuses
   the `soma_recon_wholevol.py` mapping; writes `results/marker_membership.npz`.
3. `dev/zebrafinch/big_branch_extract.py` — additive caliber key + a `--all`/driver
   (no existing-key changes).
4. `dev/zebrafinch/graph_link_whole.py` (new) — S1/S4/S5/S6: total namespace,
   candidate gen, firewalled UF, remap + certificates. Flags: `--markers
   results/marker_membership.npz`, `--r-search 200`, `--wI/--wT/--wC/--wP/--wA`,
   `--tau-tangent 0.5`, `--tau-score`, `--out`.
5. `dev/zebrafinch/oracle_stitch_decode_v1.py` — additive `--remap` streaming apply
   (default unchanged).
6. `dev/zebrafinch/graph_link_whole.README.md` (new) — symlink fix, marker check,
   no-edge control result, the `tau_score` sweep table, and the acceptance table.
7. No changes to `local_nerl_all_chunks.py`, `oracle_cc3d_chunked.py`, `em_erl`,
   `decode_v1_chunk.py`, `global_link.py`.

## Files and Areas

| Path | Change |
|---|---|
| `dev/zebrafinch/fix_decode_v1_symlinks.py` | new — S0 |
| `dev/zebrafinch/sample_markers.py` | new — S2 single marker stage |
| `dev/zebrafinch/graph_link_whole.py` | new — namespace + candidate gen + firewalled UF + remap/cert |
| `dev/zebrafinch/big_branch_extract.py` | additive — caliber key + batch driver |
| `dev/zebrafinch/oracle_stitch_decode_v1.py` | additive — `--remap` |
| `dev/zebrafinch/graph_link_whole.README.md` | new — controls, sweep, acceptance |
| (reused) | `faces/*.h5`, `soma_recon_wholevol.py` (mapping ref), `oracle_cc3d_chunked.py`, `local_nerl_all_chunks.py`, `em_erl` |
| (data, read-only) | `results/*_decode_v1.h5`, `test_50_skeletons.h5`, `yl_cb_80nm_neuron.h5`, index JSON |

## Verification Plan

Acceptance = a whole-volume NERL table (decode_v1 baseline vs linked) + passing
controls. Single roundoff tolerance **`eps = 1e-4`** on NERL.

1. **Substrate control.** All 600 symlink targets resolve; re-score 2–3 chunks —
   base/oracle/missing/n_skel/gt_len match `dv1eval/{chunk}.csv` (deterministic ⇒
   exact/near-exact). Mismatch ⇒ stop.
2. **Marker control.** #markers landing on fg ≈ #in-box markers in
   `yl_cb_80nm_neuron.h5` (compute both); a large shortfall ⇒ coord mapping wrong
   ⇒ stop.
3. **No-edge namespace control (hard).** Build `R` with **zero** accepted unions
   (pure namespacing) and run the `--remap` scorer: base, oracle, missing,
   n_skel, gt_len must equal the existing no-remap run within `eps`. This
   validates the total namespace + the modified scorer independent of any link.
   Fail ⇒ stop (namespacing/scorer bug), before trusting any link.
4. **Firewall safety (hard).** No *accepted union* yields a component with ≥2
   markers; quarantined roots unchanged and edge-free; report quarantined count
   and conflict-edge count; include a unit test of the transitive
   `A→unmarked→B` rejection.
5. **Oracle gate (hard), single tolerance `eps`.** On the linked result vs
   baseline: `|Δoracle| ≤ eps` ⇒ flat/pass; `Δoracle < −eps` ⇒ **fail** (false
   merge); `Δoracle > +eps` ⇒ **fail pending investigation** (a pure join cannot
   raise the oracle ⇒ namespace/scoring bug).
6. **Realized gain (the win).** `Δbase > +eps` at flat oracle; report Δ and the
   decode_v1 oracle ceiling (~0.906, L43) as context.
7. **Operating point (reproducible).** Report the fixed defaults and the bounded
   1-D `tau_score` calibration: sweep `tau_score ∈ {0.4,0.5,0.6,0.7,0.8}`; for
   each, build `R`, compute whole-volume base+oracle; **select the smallest
   `tau_score` whose `Δoracle` is within `eps` (max recall subject to
   merge-safety).** Record the full sweep table — this is deterministic, not free
   tuning.
8. **Cue quality (secondary).** `global_link.py`-style precision/recall on
   internal GT-crossing faces; precision vs sparse GT is a lower bound (L47).

Scope note on the metric (do not overstate): oracle-flat detects merges that
separate **distinct sampled test-50 owners**; markerless / off-GT false merges are
invisible to this sparse metric — this is not global merge safety, it is the
requested sparse-GT acceptance signal. The certificate table lets us locate and
cut any single offending edge without discarding the run.

Reviewer (code) focus: (a) `R` is total, collision-free, applied exactly once, bg
0 preserved; (b) no-edge control reproduces baseline within `eps`; (c) marker
mapping reuses `soma_recon_wholevol.py` correctly and yields ~448 fg markers; (d)
firewall (set union ≤1, quarantine, transitive) correct and tested; (e) oracle via
`branch_merge` exactly as the existing evaluator; (f) no edits to decode_v1, the
em_erl core, or unrelated dirty files.

## Risks and Questions

- **Coordinate mapping** is the highest-risk item; mitigated by reusing the
  proven `soma_recon_wholevol.py` conversion and gating on the in-box-marker
  control (S2/#2) — if the fg-landing count falls well short of the in-box count,
  stop before linking.
- **I/O over 600 chunks:** extraction reads thin boundary slabs and reuses
  `faces/`; parallelize; cache `crossings/`.
- **Whole-volume oracle sensitivity:** one bad cross-neuron link can lower the
  oracle measurably; the certificate table + the `tau_score` sweep localize and
  gate it. If even `tau_score=0.8` drops the oracle, the linker cue is unsafe and
  Step 1 fails honestly (report it) rather than shipping a merged result.
- **Scope:** larger than a block, but the eval machinery exists and new code is
  bounded to S1/S2/S4/S5/S6 + a `--remap` hook. If code review deems it too large,
  a predeclared in-bounds sub-block is a smoke test only; the whole-volume number
  remains acceptance.
- **Marker source (resolved by user):** `yl_cb_80nm_neuron.h5` (corrected neuron
  nuclei). The `soma_recon`-consumed `yl_cb_80nm.h5` is the fallback; swapping is
  a one-line source change (same grid/shape).

## Changes Since Previous Plan Version

- **Total namespace (Codex [major] #1):** `R` is now a total, collision-free
  `(chunk_key, local_label)→global_id` map over *all* foreground labels via a
  load-time enumeration + UF, dtype `uint32`, bg 0 fixed, applied exactly once.
- **No-edge control (Codex [major] #2):** added as a hard gate (S1/#3) —
  zero-union remap must reproduce the no-remap baseline within `eps`.
- **Marker contract (Codex [major] #3):** pinned source `yl_cb_80nm_neuron.h5`
  (per user: corrected neuron nuclei; labeled uint16, 465 distinct IDs), one stage
  `sample_markers.py` reusing the `soma_recon_wholevol.py` mapping, in-box-marker
  control; membership no longer split between extractor and linker.
- **Quarantine invariant (Codex [major] #4):** reworded — quarantined roots
  unchanged/edge-free; assert *no accepted union* creates/absorbs a multi-marker
  component (not a global "no id has ≥2 markers").
- **Operating point (Codex [major] #5):** predeclared feature formulas + fixed
  weights + `tau_iou_min=0` exactly + `tau_tangent=0.5`, plus a bounded,
  deterministic 1-D `tau_score` sweep/selection (max recall s.t. oracle-flat) —
  no deferral to Step 5.
- **Oracle gate (Codex [major] #6):** single tolerance `eps=1e-4`; negative-beyond
  fails, positive-beyond fails pending investigation.
- **Merge-safety wording (Codex [minor]):** scoped to distinct sampled test-50
  owners, not global merge safety.
- **Auditability (Codex [minor]):** validate all 600 symlink targets + crossing
  schema; certificates record **current-root** marker sets and commit order.
