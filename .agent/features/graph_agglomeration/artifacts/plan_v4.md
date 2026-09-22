# Plan v4

## Summary

Final plan for **Step 1: a whole-volume big-branch endpoint-continuation linker on
the existing 600-chunk `decode_v1` base, under a nucleus-firewalled union-find,
accepted by whole-volume `test_50_skeletons` NERL — realized base up, merge-oracle
flat within roundoff.** plan_v4 keeps plan_v3's resolved contracts (total
collision-free namespace, exactly-once remap, single marker stage, quarantine +
transitive firewall, `tau_iou_min=0`, single `eps=1e-4` oracle gate, non-circular
raw score, ambiguity-defer, no caliber floor, numeric marker gate, crossing-schema
assert) and pins the three final plan_v3-review fixes: the **best-other-partner**
ambiguity competitor, the **union-consistent** namespace assertions (distinct-id
only under the zero-union control), and **one union attempt per undirected pair**.

Substrate (verified): 600 `{key}_decode_v1.h5` (dataset `main`, `(1008,1008,1008)`
uint32) in `dev/zebrafinch/results/`; `decode_v1.chunks/` symlinks stale → re-point,
no regeneration. Markers: `yl_cb_80nm_neuron.h5` (corrected neuron nuclei).

Step ladder (later CCC runs): 2 soma-root/identity-prior; 3 targeted local
corrections; 4 component-wise weak-coverage recovery; 5 full-scale sweep + audit.

## Scope

### S0 — Substrate (mechanical)
Re-point `dev/zebrafinch/decode_v1.chunks/chunk_{key}.h5 →
../results/{key}_decode_v1.h5` for all 600 keys; **assert every target resolves**.

### S1 — Total global-ID namespace (remap is a total function)
Enumerate **every distinct foreground `(chunk_key, local_label)`** across all 600
chunks into a contiguous node index at load. Union-find over **all** nodes (most
stay singletons). `global_id` = compressed contiguous root index, `uint32` (assert
count < 2³¹). The remap `R: (chunk_key, local_label) → global_id` is **total**
(untouched labels each get a unique id); background `0 → 0`, never remapped.
Evaluator applies `R` **exactly once** per sampled voxel for both linked-base NERL
and `branch_merge`. **Structural asserts** (hard, independent of GT), scoped
correctly to unions: (a) *zero-union control only* — node-wise injectivity, i.e.
each enumerated node maps to its own distinct nonzero id; (b) *linked output* —
total nonzero mapping (every foreground node has a nonzero id), `R(u) == R(v) ⇔
find(u) == find(v)`, and distinct ids across distinct roots (unions intentionally
share an id, so node-wise injectivity does **not** hold post-union); (c) background
fixed at 0; (d) remap lookup covers every foreground `(key,label)` the evaluator
encounters (no missing key).

### S2 — Marker sampling (ONE stage, pinned source, reused mapping)
Source = `/projects/weilab/dataset/zebrafinch/yl_cb_80nm_neuron.h5` (per user:
corrected neuron nuclei; key `main`, `(1425,1365,1333)` uint16, 465 non-zero
distinct IDs — labeled, not binary). **No automatic fallback to `yl_cb_80nm.h5`
for this run.** Map each marker voxel 80 nm → 10 nm 1008-grid by **reusing the
exact conversion in `soma_recon_wholevol.py`** (`POOL=[8,16,16]`, `CBSCALE=[4,8,8]`,
`caff=(g*CBSCALE)//POOL`, snap-off-fg handling) — do not re-derive (L14). For each
marker id, sample the decode_v1 base label → membership
`M: (chunk_key, local_label) → set(marker_ids)`, written once to
`results/marker_membership.npz`. **Numeric gate:** compute `n_inbox` (markers whose
grid location lands inside the 600-box) and `n_onfg` (of those, landing on a
foreground base segment); require `n_onfg / n_inbox ≥ 0.95`, else the mapping is
wrong → stop. Log markers that miss fg.

### S3 — Big-branch node extraction (600 chunks)
Run `big_branch_extract.py` per `results/{key}_decode_v1.h5` →
`crossings/{key}.npz`: per (segment, face) row — global centroid, cross-section
footprint + area, unit inward PCA tangent, elongation, `perp` cleanness, caliber
proxy. Reuse `faces/{key}.h5` footprints where possible. Changes to
`big_branch_extract.py` are **additive only** (caliber-proxy key if absent; new
keys, existing unchanged). Batch via a thin driver. **Hard schema assert before
linking:** all 600 `crossings/*.npz` exist and carry the required keys with
consistent row lengths.

### S4 — Candidate generation + raw score (non-circular)
For each internal shared face, generate candidate edges between oppositely-facing
big-branch crossings whose footprint centroids are within `r_s = 200 nm` on the
face. Features in `[0,1]`: `IoU_face`, `tangent = max(0, −cos∠(t_A,t_B))`,
`caliber = min(area_A,area_B)/max(area_A,area_B)`, `perp = perp_A·perp_B`.

**Raw score (no ambiguity term — this removes the circularity):**
`raw(e) = (wI·IoU + wT·tangent + wC·caliber + wP·perp) / (wI+wT+wC+wP)`, in `[0,1]`,
with fixed defaults `wI=1.0, wT=2.0, wC=1.0, wP=1.0`.

**Ranking / gates** all use `raw`: hard floors `tau_iou_min = 0` exactly (no
overlap gate), `tau_tangent = 0.5` (reject worse than 60° anti-parallel);
**there is no caliber floor** — caliber enters only through `raw`. A candidate is
**mutual-best** iff it is the top-`raw` partner for *both* endpoints on that face.
Deterministic ordering / tie-break: `(−raw, chunk_key_u, label_u, chunk_key_v,
label_v)`.

**Ambiguity as a separate DEFER gate (not inside `raw`):** for edge `e=(a,b)`, let
`comp_a = max({raw(a,x) | x ≠ b}, default=0)` — the **best other partner** of `a`
on the face (not the second-highest; when `e` is `a`'s top partner this equals the
runner-up) — and `comp_b = max({raw(b,y) | y ≠ a}, default=0)` symmetrically.
`ambiguity(e) = max(comp_a, comp_b) / max(raw(e), 1e-6)`, clipped to `[0,1]`. An
edge with `ambiguity(e) > tau_amb = 0.8` (a near-equal competitor exists) is
**DEFERRED** (recorded, not merged) — the design's "defer ambiguous" rule.

**Retain all candidates for certification; process each undirected pair once.**
Every generated candidate is written to the certificate table with its cues and
decision. Each physical edge is **canonicalized to one undirected pair** `{u,v}`
ordered by `(chunk_key, label)`, and there is **exactly one union attempt per
undirected pair** (a `find(u)==find(v)` same-root pair is a no-op). Only
mutual-best, non-deferred, floor-passing pairs enter union processing.

### S5 — Nucleus-firewalled union-find (marker-set; quarantine)
Node = `(chunk_key, local_label)`; each root stores a marker-id `set` from `M`.
**Quarantine:** any base segment whose `M` set has ≥2 markers is a fixed singleton,
accepts no edges, is never modified; count and report. **`can_link(C,D,e)`**:
neither quarantined; `e` mutual-best, not deferred, `tau_tangent` passes, and
**`raw(e) ≥ tau_score`**; and **`|markers(C) ∪ markers(D)| ≤ 1`**. Commit in
descending `raw`. The `|·|≤1` rule blocks the transitive `A→unmarked→B` path.
**Invariant (assert):** no *accepted union* creates or absorbs a multi-marker
component; quarantined roots remain unchanged/edge-free.

### S6 — Outputs
`results/graph_link_remap.npz` (the total `R` + accepted-union list) and
`results/graph_link_certs.csv` — one row per **generated candidate** with: face,
u=(key,label), v=(key,label), IoU, tangent, caliber, perp, raw, ambiguity,
mutual_best (bool), deferred (bool), commit_order, **markers_u/markers_v (current
root sets at decision time)**, decision ∈ {accepted, deferred_ambiguous,
not_mutual_best, floor_fail, firewall_conflict}, reject_reason.

### S7 — Whole-volume evaluation via the remap
Extend `oracle_stitch_decode_v1.py` with `--remap results/graph_link_remap.npz`
(default unchanged) applying `R` while streaming each chunk through
`oracle_cc3d_chunked.sample_variant`. Report on 600-box-cropped
`test_50_skeletons.h5` with the exact existing config (`RES=[10,10,10]`,
`branch_merge` oracle, `merge_threshold=1`, break/length 1000, canonical crop):
**base** and **merge-oracle** for (a) decode_v1 baseline and (b) linked. No new
NERL code.

**Out of scope:** decode_v1 recipe/bg-fill; soma-root prior;
cc-cut/bg-fill/multi-plane corrections; weak-coverage recovery; any git commit; the
24 pre-existing dirty files.

## Proposed Changes

1. `dev/zebrafinch/fix_decode_v1_symlinks.py` (new) — S0 re-point + assert.
2. `dev/zebrafinch/sample_markers.py` (new) — S2 single marker stage; reuses
   `soma_recon_wholevol.py` mapping; numeric gate; `results/marker_membership.npz`.
3. `dev/zebrafinch/big_branch_extract.py` — additive caliber key + `--all`/driver;
   schema self-check.
4. `dev/zebrafinch/graph_link_whole.py` (new) — S1/S3-assert/S4/S5/S6: total
   namespace, candidate gen + raw score + ambiguity-defer, firewalled UF,
   remap + full certificate. Flags: `--markers results/marker_membership.npz`,
   `--r-search 200`, `--wI/--wT/--wC/--wP`, `--tau-tangent 0.5`, `--tau-amb 0.8`,
   `--tau-score`, `--out`.
5. `dev/zebrafinch/oracle_stitch_decode_v1.py` — additive `--remap`.
6. `dev/zebrafinch/graph_link_whole.README.md` (new) — symlink fix, marker gate,
   no-edge + structural controls, `tau_score` sweep, acceptance table.
7. No changes to `local_nerl_all_chunks.py`, `oracle_cc3d_chunked.py`, `em_erl`,
   `decode_v1_chunk.py`, `global_link.py`.

## Files and Areas

| Path | Change |
|---|---|
| `dev/zebrafinch/fix_decode_v1_symlinks.py` | new — S0 |
| `dev/zebrafinch/sample_markers.py` | new — S2 marker stage + numeric gate |
| `dev/zebrafinch/graph_link_whole.py` | new — namespace + candidate/raw-score + ambiguity-defer + firewalled UF + remap/cert |
| `dev/zebrafinch/big_branch_extract.py` | additive — caliber key + batch driver + schema check |
| `dev/zebrafinch/oracle_stitch_decode_v1.py` | additive — `--remap` |
| `dev/zebrafinch/graph_link_whole.README.md` | new — controls, sweep, acceptance |
| (reused) | `faces/*.h5`, `soma_recon_wholevol.py` (mapping ref), `oracle_cc3d_chunked.py`, `local_nerl_all_chunks.py`, `em_erl` |
| (data, read-only) | `results/*_decode_v1.h5`, `test_50_skeletons.h5`, `yl_cb_80nm_neuron.h5`, index JSON |

## Verification Plan

Acceptance = a whole-volume NERL table (baseline vs linked) + passing controls.
Single roundoff tolerance **`eps = 1e-4`** on NERL.

1. **Substrate control.** All 600 symlink targets resolve; re-score 2–3 chunks —
   base/oracle/missing/n_skel/gt_len match `dv1eval/{chunk}.csv` (deterministic ⇒
   exact/near-exact). Mismatch ⇒ stop.
2. **Marker control (numeric).** `n_onfg / n_inbox ≥ 0.95` for
   `yl_cb_80nm_neuron.h5`; else coord mapping wrong ⇒ stop.
3. **Namespace controls (hard, two-part).** (a) *Structural:* each enumerated node
   → one distinct nonzero id, bg 0 fixed, complete remap coverage. (b) *No-edge
   metric:* build `R` with zero accepted unions and run `--remap`; base/oracle/
   missing/n_skel/gt_len equal the no-remap run within `eps`. Either failing ⇒
   stop before trusting any link.
4. **Firewall safety (hard).** No accepted union yields a ≥2-marker component;
   quarantined roots unchanged/edge-free; report quarantined + conflict counts;
   unit test the transitive `A→unmarked→B` rejection.
5. **Crossing-schema (hard).** All 600 `crossings/*.npz` present, required keys,
   consistent row lengths — asserted before linking.
6. **Oracle gate (hard), single `eps`.** `|Δoracle| ≤ eps` ⇒ pass;
   `Δoracle < −eps` ⇒ **fail** (false merge); `Δoracle > +eps` ⇒ **fail pending
   investigation** (a pure join cannot raise the oracle).
7. **Realized gain (win).** `Δbase > +eps` at flat oracle; report Δ and the
   decode_v1 oracle ceiling (~0.906, L43) as context.
8. **Operating point (reproducible).** Report fixed weights + the bounded 1-D
   `tau_score` sweep `∈ {0.4,0.5,0.6,0.7,0.8}`; select the **smallest** `tau_score`
   whose `Δoracle` is within `eps` (max recall s.t. oracle-flat). Record the table.
9. **Cue quality (secondary).** `global_link.py`-style precision/recall on internal
   GT-crossing faces; precision vs sparse GT is a lower bound (L47).

Scope note on the metric: oracle-flat detects merges that separate **distinct
sampled test-50 owners**; markerless / off-GT merges are invisible to this sparse
metric — the requested acceptance signal, not global merge safety. The certificate
table localizes any single offending edge.

Reviewer (code) focus: (a) `R` total/collision-free/applied-once + structural
asserts pass; (b) raw score non-circular, ambiguity a separate defer gate, no
caliber floor, all candidates certified but only mutual-best merged; (c) marker
mapping reuses `soma_recon` and passes the 0.95 gate; (d) firewall set-union ≤1 +
quarantine + transitive, tested; (e) oracle via `branch_merge` exactly as existing;
(f) no edits to decode_v1, em_erl, or unrelated dirty files.

## Risks and Questions

- **Coordinate mapping** — highest risk; mitigated by reusing the proven
  `soma_recon_wholevol.py` conversion and the numeric 0.95 marker gate.
- **I/O over 600 chunks** — thin-slab reads + `faces/` reuse + parallelize + cache
  `crossings/`.
- **Endpoint vs overlap balance** — overlap is a soft cue; safety rests on the
  tangent floor + mutual-best + ambiguity-defer + firewall + the oracle-flat gate.
  If even `tau_score=0.8` drops the oracle, the cue is unsafe and Step 1 fails
  honestly (report), rather than shipping a merged result.
- **Marker source (resolved):** `yl_cb_80nm_neuron.h5`, no auto-fallback this run.
- **Scope** — larger than a block, but the eval machinery exists and new code is
  bounded to S1/S2/S4/S5/S6 + a `--remap` hook.

## Changes Since Previous Plan Version

Three precise fixes from the plan_v3 review (approach unchanged; plan rounds p3→p4):

- **Ambiguity competitor = best other partner (Codex [major]):** `comp_a =
  max({raw(a,x) | x ≠ b}, default=0)` (not "second-highest"), symmetrically
  `comp_b`; this is the strongest alternative and correctly drives the defer
  decision.
- **Union-consistent namespace asserts (Codex [major]):** node-wise distinct-id
  injectivity is asserted **only under the zero-union control**; the linked output
  asserts total nonzero mapping, `R(u)==R(v) ⇔ find(u)==find(v)`, and distinct ids
  across roots (accepted unions intentionally share an id).
- **One union attempt per undirected pair (Codex [minor]):** each physical edge is
  canonicalized to a single `{u,v}` pair ordered by `(chunk_key,label)`; exactly
  one union attempt per pair (same-root → no-op); the certificate still logs every
  generated candidate.

Prior plan_v2→v3 changes (raw/ambiguity split, caliber-floor removal, structural
+ numeric + schema hard checks, retain-all-candidates) are retained.
