# Plan v1

## Summary
`dev/mit_liconn/decode_axon.py`: a modular axon decoder run as strong-2D → 3D-cleanup → (optional)
weak-bridge → crumb. This revision makes every call contract, default, and safety/exclusivity rule
explicit so it is implementable without guesswork, per the plan_v0 review. The shippable default path
(strong + 3D + drop-only crumb) reproduces today's `decode_p1p2` and folds in the validated bump-safe
tube-extension (measured `decode_p1p2eg`: VALID vol 65.5%, bumps 89, parallel 0). The weak-bridge tier
is OFF by default, fully specified with two-anchor exclusivity, and isolated in the ablation.

Core code moves: (1) add ONE pure pipeline function to `decode_v2.py`; (2) extract pure cores from
`decode_v4_split.py`, `decode_v3_merge.py`, `tube_extend.py` (their `main()` become thin wrappers, no
behavior change); (3) add an array-level entry to `valid_tube_metric.py`; (4) the new weak-bridge and
crumb logic live in `decode_axon.py`.

## Scope
In: `decode_axon.py` (new); behavior-preserving core extractions in `decode_v2.py`,
`decode_v4_split.py`, `decode_v3_merge.py`, `tube_extend.py`; an additive `score_array(...)` in
`valid_tube_metric.py`; optional `sbatch_decode_axon.sh`.
Out: model/waterz/link-primitive changes; NERL/GT tuning (NERL is secondary, informational); weak-only
new axons (hysteresis forbids); Neuroglancer/meshing.

## Proposed Changes

### 0. Foreground masks (task a)
```
fg_max(aff)     = aff.max(axis=0)                      # (Z,Y,X) float, max over z,y,x channels
strong_mask(aff)= fg_max(aff) > 0.66                   # == decode_lib.AFF_BG; today's sections
weak_mask(aff)  = (fg_max(aff) > 0.30) & (fg_max(aff) <= 0.66)   # disjoint from strong by construction
affxy(aff)      = aff[1:].mean(0, dtype=float32)       # in-plane membrane (y,x); used by force-split/prob1/weak
```

### 1. Tier 1 — strong 2D-seg pipeline (decode_v2 steps 1-8) — ADD a pure function to decode_v2.py
Extract the a→split→c core of `decode_v2._run_pipeline` (lines ~406-462) into:
```
# in decode_v2.py
def decode_sections(seg2d, affxy, *, fuse_iou=0.2, area_tol=0.5, next_overlap=0.5, min_len=3) -> np.uint32:
    z_of, area = section_index(seg2d)                                    # from force_split_decode
    a,b,aa,ab,inter = _raw_iou_table(seg2d, z_of)
    spine, bb       = _link_recipe(a,b,aa,ab,inter)                       # base link (needed for eligibility)
    incoming, detected, _ = detect_fusions(a,b,aa,ab,inter, area, fuse_iou, area_tol)
    eligible, _     = eligible_fusions(detected, spine, z_of, min_len)
    split_seg, runs = force_split(seg2d, affxy, incoming, eligible, area_tol, next_overlap)
    sz,_            = section_index(split_seg)
    sa,sb,saa,sab,sinter = _raw_iou_table(split_seg, sz)
    ssp, sbb        = _link_recipe(sa,sb,saa,sab,sinter)
    cand            = _candidate_links(sa,sb,saa,sab,sinter, ssp, sbb)
    vol_c,_,_       = constrained_relabel(split_seg, runs, cand)
    return vol_c.astype(np.uint32)
```
`_run_pipeline` is refactored to call `decode_sections` (keeps the GT eval/save/selection around it), so
behavior is unchanged. `force_split(seg2d, affxy, incoming, eligible_runheads, area_tol, next_overlap)`
returns `(split_seg, runs)` where each `run` is a dict with `status/stop/seedA/seedB/sideA/sideB/
mand_links/pieces/terminals/split_sections/owned/run_id` (source: `force_split_decode`). `section_index`
also lives in `force_split_decode`. decode_axon calls `decode_v2.decode_sections(seg2d, affxy)`.

### 2. Tier 1 sectioning (task d-strong, steps 1-3) — decode_axon.build_strong_sections
Do NOT reuse `_load_inputs(args)`. Call `decode_lib.waterz_2d_spacefill` directly with the fixed recipe,
or load the cache:
```
def build_strong_sections(z0, z1, thr=0.3) -> np.uint32:   # (z1-z0, Y, X)
    full_cache = OUTDIR / f"sections_thr{thr}_z0-{DEPTH}.h5"        # DEPTH=800; the only cache that exists
    if full_cache.exists():  seg2d = h5[main][z0:z1]                # slice the full-volume cache
    else:
        aff = load_aff(z0, z1)                                       # (3, z1-z0, Y, X) float32 clipped [0,1]
        seg2d = waterz_2d_spacefill(aff, np.ones(shape,bool), thr, SMALL, AFF_LOW,
                                    rg_zero=RG_ZERO, score=getScoreFunc("aff30_his256_ran255"), aff_bg=AFF_BG)
    return seg2d.astype(np.uint32)
```
(`SMALL=150, AFF_LOW=0.01, AFF_BG=0.66, RG_ZERO=False` from `decode_lib`.) The cache is named by the
GLOBAL bounds `z0-800`; a sub-slice reads `[z0:z1]` from it (never a `z0-{z0}` exact name).

### 3. Tier 2 — 3D cleanup for incomplete tubes (task b2 = decode_p1p2, SPLIT then MERGE)
All three are extracted as pure functions; each script's `main()` becomes a thin load→call→save wrapper
with identical CLI (verify by array equality, §Verification). Restriction "incomplete = touch <2 faces"
is enforced INSIDE each core exactly as the current scripts already do (`nf.get(id,0) < 2`).

- **`decode_v4_split.prob1_carve(seg, affxy, *, tau=0.20, min_excess=200, max_len=30, med_w=31,
  area_tol=0.5, min_slices=40) -> np.uint32`** — the current `main` body minus h5 I/O. It is already
  structurally bump-safe (carves ONLY a detected through-intruder bump whose two halves are both orphans,
  area-matched, terminating at the bump; `_watershed_two`-seeded). No separate bump gate is added; the
  existing gates ARE the safety proof. Keep the `face_counts` default-absent-=-0 fix.
- **`tube_extend.prob1_extend(seg, affxy, *, min_vox=2000, area_tol=0.5, host_margin=0.3, mem_thr=1.0,
  max_steps=60, gate="bumpsafe") -> np.uint32`** — extract the current `main` full-apply loop: compute
  `zr` (`compute_bbox_all_3d`), `area=per_slice_area(seg)`, `nf=face_counts(seg)`; orphans = touch<2 faces
  & size≥min_vox, biggest-first; for each `open_ends(seg,oid,zr[oid])` run `extend_end(seg, affxy, oid,
  z_end, d, area_tol, host_margin, mem_thr, max_steps)` (BOTH ±z), then `gate_bumpsafe(area, oid, carved,
  bp=(tau,min_excess,max_len,med_w))`; commit voxels + update `area` only on accept. `main()` calls this.
  This reproduces the measured 27-accept/97-reject `decode_p1p2eg`.
- **`decode_v3_merge.prob2_merge(seg, aff_path=AFF, *, aff_thr=0.5, min_contact=20, min_size=500,
  max_zoverlap=0.5, area_tol=None) -> np.uint32`** — extract current `main`: `region_graph(seg)` reads the
  RAW affinity from `aff_path` and aggregates channel-correct boundary means over all 3 directions
  (unchanged); union-find merges pair (a,b) iff `contact≥min_contact AND mean_aff≥aff_thr AND nf[a]<2 AND
  nf[b]<2 AND z_overlap_frac(zr[a],zr[b]) ≤ max_zoverlap`. Checks apply to ORIGINAL labels (matches
  current behavior; no evolving-component re-check). `z_ranges`/`z_overlap_frac` (not a bare
  `compute_bbox_all_3d` call) provide the parallel veto. NEW optional `area_tol`: when set, additionally
  require `|size[a]-size[b]| ≤ area_tol*max(size)`; **default None = OFF so prob2_merge byte-reproduces
  decode_p1p2** (the user's "area-matching" is an experimental tightening, not the baseline).
- Composition: `decode_3d(seg, affxy) = prob2_merge(prob1_extend(prob1_carve(seg, affxy), affxy))`.

### 4. Tier 3 — weak recovery (task d-weak, e, f) — decode_axon, `--weak`, OFF by default
- **`build_weak_sections(aff, *, open_radius=1, min_cc=50, connectivity=8) -> np.uint32`**: per z-slice,
  `wm = weak_mask(aff)[z]`; `opened = fastmorph.spherical_open(wm.astype(uint8), radius=open_radius)` (2D
  binary open removes weak salt-bridges; coder pins the exact fastmorph signature against the installed
  version — `spherical_open`/`opening` both present); `lab = cc3d.connected_components(opened,
  connectivity=connectivity)`; drop CCs < `min_cc`; offset labels to be globally unique per slice (same
  scheme as `waterz_2d_spacefill`). Returns `wseg2d` on the SAME grid, disjoint from strong voxels.
- **`weak_bridge(vol3d, wseg2d, affxy, *, max_zgap=15, area_tol=0.5, min_iou=0.1) -> np.uint32`** — a
  two-anchor, mutex-exclusive gap-bridge. Algorithm:
  1. Endpoints: for each INCOMPLETE strong tube (touch<2 faces), take its top and bottom open ends (z,
     centroid, area) via `open_ends`+`section_index`.
  2. Weak chains: build z-consecutive IoU adjacency over `wseg2d` (reuse `segs_to_iou`/`_raw_iou_table`);
     a weak CHAIN is a connected run of weak sections through the gap.
  3. Candidate bridge = (strong end E_A at z_a) —weak chain— (strong end E_B at z_b) such that: the chain
     starts IoU-linked to E_A and ends IoU-linked to E_B; `|z_b - z_a| ≤ max_zgap`; the two strong tubes
     are z-SEQUENTIAL (`z_overlap_frac ≤ max_zoverlap`, the prob2 veto — rejects parallels); areas of
     E_A, chain, E_B all match within `area_tol`.
  4. EXCLUSIVITY: a weak chain may anchor EXACTLY TWO distinct strong ends. Reject any chain touching ≥3
     strong ends (branched/ambiguous). Each strong END and each weak node is a mutex resource: process
     candidates by descending min-IoU, union-find merge the two strong tubes + absorb the chain, and
     mark all consumed ends/weak-nodes used; skip a candidate whose end or nodes are already used.
  5. Bump-safe: before committing, run a merge bump-safety check (§5) on the merged {A, chain, B}; skip if
     it would create a bump. Unused weak sections are DISCARDED (never seed a new axon; weak-to-weak only
     exists inside an accepted two-anchor chain).

### 5. Generic merge/absorb bump-safety (addresses gate_bumpsafe being carve-specific)
Add `bump_safe_merge(area, ids, bp) -> bool` in decode_axon: given the cached per-slice areas of the
segments in `ids` (the two strong tubes + any absorbed weak/crumb voxels), form the UNION profile and
require `nbumps(union) ≤ Σ nbumps(ids_before)` AND no new bump vs the largest member. `nbumps` reuses
`decode_v4_split.bumps`. `gate_bumpsafe` stays the carve-specific check; `bump_safe_merge` is the
union-of-labels check used by `weak_bridge` and crumb absorption.

### 6. Crumb cleanup (task g) — DROP-ONLY default; absorption behind a flag
```
def crumb_cleanup(vol3d, *, min_vox=500, absorb=False, orphan_min_vox=2000) -> np.uint32:
```
- Default (`absorb=False`, DETERMINISTIC, SAFE): relabel every seg with size < `min_vox` to 0 (dust drop).
- `--crumb-absorb` (opt-in): additionally, for each orphan (touch<2 faces AND size<orphan_min_vox), find
  the unique adjacent seg by max shared-face contact over the RAW region graph; absorb ONLY if that seg
  is unambiguous (one dominant neighbor, contact ≥ min_contact), z-sequential (`z_overlap_frac ≤
  max_zoverlap`), and `bump_safe_merge` passes; else the orphan is LEFT AS-IS (not dropped). Report
  dropped/absorbed/left counts.

### 7. Orchestrator + CLI
```
run_pipeline(args):
  seg2d = build_strong_sections(z0,z1,thr)
  v1 = decode_v2.decode_sections(seg2d, affxy)                 # Tier 1
  score+save("_t1_strong", v1) [ablation]
  v2 = decode_3d(v1, affxy)                                    # Tier 2 (prob1_carve->prob1_extend->prob2_merge)
  score+save("_t2_3d", v2)
  v3 = weak_bridge(v2, build_weak_sections(aff), affxy) if args.weak else v2
  if args.weak: score+save("_t3_weak", v3)
  out = crumb_cleanup(v3, absorb=args.crumb_absorb) if not args.no_crumb else v3
  save decode_axon{tag}.h5 = out; score(out)                  # FINAL default artifact
```
- Metric: add `valid_tube_metric.score_array(seg, min_z=20, min_vol=10000, tau=0.20, min_excess=200,
  max_len=30, med_w=31, min_slices=40, min_cc=50, multi_thr=0.3, list_invalid=False)` (refactor the file's
  `score(path,...)` to load then delegate to `score_array`). decode_axon prints `score_array` after each
  tier; `--full` also saves the final h5 and prints `liconn_nerl.evaluate(final_path, do_oracle_merge=
  False)[0]` and `[True]` (secondary, informational).
- CLI (mirror decode_v2.py): mutually-exclusive REQUIRED `--full` | `--zslice Z0:Z1` | `--self-test`;
  `--tag` (literal suffix); toggles `--weak` (default off), `--no-prob1`, `--no-prob2`, `--no-crumb`,
  `--crumb-absorb`; tunables with the defaults above. `--self-test` returns BEFORE any data I/O.
  `--zslice` validates `0 ≤ z0 < z1 ≤ DEPTH`; a `--zslice` run is smoke-only and MUST use a non-empty
  `--tag` (assert), so it can never overwrite the full `decode_axon.h5`; slab z-boundaries (z0>0 or
  z1<DEPTH) are NOT counted as volume faces in that mode (label output "smoke, face counts not full-volume
  comparable").

### 8. Self-test (data-free, decode_v2 style)
`self_test()` builds tiny synthetic volumes and asserts: (a) strong/weak masks partition fg and are
disjoint; (b) `prob1_carve` carves a synthetic through-intruder and REJECTS a solid tube (no bump);
(c) `prob1_extend`+`gate_bumpsafe` accept a bump-removing carve, reject a bump-adding one; (d)
`prob2_merge` unions a z-SEQUENTIAL high-aff incomplete pair and SKIPS a z-OVERLAPPING (parallel) pair;
(e) `weak_bridge` merges a synthetic strong—weakchain—strong sequential triple, REJECTS a chain touching
3 strong ends (ambiguous) and a parallel pair, and enforces chain exclusivity (a used chain isn't reused);
(f) `crumb_cleanup` drops a sub-`min_vox` blob, preserves a big one, and (absorb=False) never merges.

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_axon.py` | NEW — masks, build_strong_sections, decode_3d wiring, build_weak_sections, weak_bridge, bump_safe_merge, crumb_cleanup, run_pipeline, self_test, CLI |
| `dev/mit_liconn/decode_v2.py` | ADD pure `decode_sections(...)`; `_run_pipeline` calls it (no behavior change) |
| `dev/mit_liconn/decode_v4_split.py` | extract `prob1_carve(seg, affxy, ...)`; `main()` thin wrapper |
| `dev/mit_liconn/decode_v3_merge.py` | extract `prob2_merge(seg, aff_path, ..., area_tol=None)`; `main()` thin wrapper |
| `dev/mit_liconn/tube_extend.py` | extract `prob1_extend(seg, affxy, ...)` from `main`'s full-apply loop; `main()` calls it |
| `dev/mit_liconn/valid_tube_metric.py` | ADD `score_array(seg, ...)`; `score(path,...)` loads then delegates |
| `dev/mit_liconn/sbatch_decode_axon.sh` | OPTIONAL — full-volume run mirror |

`dev/` and `.agent/` are gitignored → the code-review prompt must carry the FULL current contents of
`decode_axon.py` AND the diffs/contents of the five modified modules, not a `git diff`.

## Verification Plan
1. `conda run -n pytc python dev/mit_liconn/decode_axon.py --self-test` → all contract tests pass.
2. Behavior-preserving extraction — array equality (normalized partitions, NOT h5 bytes): the rewrapped
   `decode_v4_split.py`/`decode_v3_merge.py`/`tube_extend.py` reproduce `decode_p1`, `decode_p1p2`,
   `decode_p1p2eg` from the same inputs (compare `main` label arrays up to relabeling via a partition
   hash). `decode_v2.decode_sections` reproduces `decode_v2_c_constrained*.h5`.
3. Pilot `conda run -n pytc python dev/mit_liconn/decode_axon.py --zslice 0:96 --tag _smoke` → runs,
   prints smoke metric.
4. Full `--full` (inline ~5–8 min or `sbatch_decode_axon.sh`, `-p short -c 8 --mem 90G`) → prints the
   ablation table (t1 strong / t2 3d / t3 weak / final) and `valid_tube_metric.py decode_p1p2 decode_axon`.
5. Gates on the FINAL saved artifact: (a) strong+3D (weak off) ≥ decode_p1p2 (VALID vol ≥ 62.7, parallel
   ≤ 8, bumps ≤ 117) — expected ≈ decode_p1p2eg (65.5 / 89 / 0); (b) with `--weak`, VALID vol strictly ↑
   AND parallel ≤ 8 AND bumps ≤ 117, else `--weak` stays OFF by default and the final artifact is the
   strong+3D+crumb result. NERL reported secondary.

## Risks and Questions
- Extraction must be behavior-preserving; §Verification-2 (partition-hash equality) is the guard, stronger
  than h5-byte compare and not fooled by relabeling.
- `weak_bridge` remains the highest-risk tier (packed parallel axons); mitigations = z-sequential veto +
  two-anchor exclusivity + area-match + `bump_safe_merge` + OFF by default. If it nets ≤0, it ships off.
- fastmorph exact op/signature (`spherical_open` vs `opening`, radius, border) is pinned by the coder
  against the installed build; `open_radius=1`, `min_cc=50`, `connectivity=8` are the defaults.
- Coverage floor (~9.5% missed, 69% no-signal) caps weak recovery; success is vs decode_p1p2, not oracle.
- Assumption (accepted in v0 review): primary metric = valid-tube volume; NERL secondary (GT over-split).

## Changes Since Previous Plan Version
Addresses every plan_v0 review finding:
- #1/#2: added `decode_v2.decode_sections` as the single canonical pure pipeline fn; named all sources
  (`force_split`/`section_index` from `force_split_decode`) and the `run` dict keys; included
  `_candidate_links`.
- #3: `build_strong_sections` calls `waterz_2d_spacefill` directly with mapped args; cache is the global
  `z0-800` name, sub-slice reads `[z0:z1]`.
- #4: `decode_3d` composition gives every fn its full signature + measured defaults.
- #5: prob1_carve documented as structurally bump-safe and incomplete-only (existing gates); no unsafe
  carve path.
- #6: `prob1_extend` full signature + how it builds `zr/area/carved/bp`, bidirectional ±z, recompute-on-
  accept; extract-from-`main` chosen (not copy).
- #7: prob2 checks on original labels; `z_ranges`/`z_overlap_frac` named; region_graph channel-correct
  aggregation preserved; area-match added as opt-in `area_tol=None`.
- #8: weak sectioning pins fastmorph 2D binary open + cc3d, radius/min_cc/connectivity defaults.
- #9/#10: `weak_bridge` fully specified with z-gap/centroid/area limits, helper reuse, two-anchor
  exclusivity, mutex resources, deterministic descending-IoU order, ambiguous-chain rejection.
- #11: added generic `bump_safe_merge(area, ids, bp)` for bridge/absorb; `gate_bumpsafe` stays carve-only.
- #12: crumb DROP-ONLY default; absorption behind `--crumb-absorb` with explicit contact/z-overlap/bump
  rules; unmatched orphan left as-is.
- #13: added `valid_tube_metric.score_array`; all metric params stated.
- #14: NERL via `liconn_nerl.evaluate(path, do_oracle_merge=)[0]`, saved-then-evaluated, `--full` only.
- #15: FINAL saved artifact is gated on all three metric limits; weak on-by-default only if it passes.
- #16: one required mode; slice-bound validation; `--self-test` exits before I/O; `--zslice` requires a
  tag (no overwrite); slab boundaries excluded from faces in smoke mode.
- #17: self-test list expanded to the load-bearing contracts + explicit bump fixtures.
- #18: extraction verified by partition-hash array equality; code review will carry full contents of all
  six touched modules.
- #19/#20 (minor): corrected terminology (Tier1=decode_v2, strong+3D=decode_p1p2); `conda run -n pytc`
  in all verification commands.
