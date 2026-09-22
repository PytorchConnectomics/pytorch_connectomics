# Plan v2

## Summary
Final plan for `dev/mit_liconn/decode_axon.py`. To converge on an implementable, verifiable artifact, the
SHIPPED pipeline is scoped to the CONFIDENT path — the user's own directive: "first make sure confident
regions are decoded correctly; later add back less confident ones." That path is:

  strong 2D sections (decode_v2 steps 1-8) → 3D incomplete-tube cleanup (Prob-1 carve → bump-safe
  bidirectional extend → parallel-vetoed, area-matched Prob-2 merge) → drop-only crumb cleanup.

It reuses the validated helpers, composes the three measured-good 3D passes (Prob-1 carve, the bump-safe
extension that produced `decode_p1p2eg` = VALID vol 65.5 / bumps 89 / parallel 0, and the guarded Prob-2
merge), and is fully specified below (every signature, default, data-flow, and self-test).

DEFERRED to a focused follow-up CCC run (documented, `--weak`/`--crumb-absorb` raise `NotImplementedError`
with a pointer): the weak-region (0.3–0.66) fastmorph recovery + strong↔weak hysteresis bridge, and crumb
ABSORPTION. Those are the "less confident" tier; the plan_v1 review showed their contracts (weak-chain
exclusivity, cross-namespace bump safety, dominant-neighbor absorption) need their own design pass, and
they are OFF the shipped path so the confident core lands verified first.

## Scope
In (implement now): `decode_axon.py` (masks, orchestrator, drop-only crumb, self-test, CLI); a pure
`decode_sections(...)` in `decode_v2.py`; behavior-preserving pure-core extractions in
`decode_v4_split.py` (`prob1_carve`), `tube_extend.py` (`prob1_extend`), `decode_v3_merge.py`
(`prob2_merge`, with a signature-changed `region_graph(seg, aff)`); an additive
`valid_tube_metric.score_array(...)`; optional `sbatch_decode_axon.sh`.
Out (deferred, stubbed with NotImplementedError): `build_weak_sections`, `weak_bridge`, `bump_safe_merge`
for weak, crumb ABSORB. Also out: model/waterz/link changes, NERL/GT tuning, meshing/upload.

## Proposed Changes

### 0. Foreground masks + affinity (decode_axon)
```
DEPTH=800; OUTDIR=outputs/mit_liconn/DL288B_crop1; AFF=datasets/mit-liconn/raw_x1_head-aff_r1.h5
BUMP_PARAMS=(0.20, 200, 30, 31)   # (tau, min_excess, max_len, med_w) == valid_tube_metric defaults
def load_aff(z0,z1): a=h5[AFF][main][:, z0:z1].astype(float32); np.clip(a,0,1,out=a); return a  # (3,z1-z0,Y,X)
def affxy(aff):   return aff[1:].mean(0, dtype=float32)                 # in-plane (y,x) membrane
def strong_mask(aff): return aff.max(0) > 0.66
def weak_mask(aff):   return (aff.max(0) > 0.30) & (aff.max(0) <= 0.66)  # (deferred tier only)
```

### 1. Tier 1 — strong 2D pipeline: ADD pure `decode_v2.decode_sections`
```
def decode_sections(seg2d, affxy, *, fuse_iou=0.2, area_tol=0.5, next_overlap=0.5, min_len=3) -> np.uint32
```
Body = the a→split→c core of `_run_pipeline` (base `_raw_iou_table`→`_link_recipe`→`detect_fusions`→
`eligible_fusions`→`force_split`→re-`section_index`/`_raw_iou_table`/`_link_recipe`→`_candidate_links`→
`constrained_relabel`), returns `volume_c`. `_run_pipeline` is refactored to call it (behavior unchanged;
verified by array-equality vs `decode_v2_c_constrained*.h5`). `section_index`/`force_split` are imported
from `force_split_decode`; `force_split` returns `(split_seg, runs)`, `runs[i]` a dict with keys
`status/stop/seedA/seedB/sideA/sideB/mand_links/pieces/terminals/split_sections/owned/run_id`.

### 2. Tier 1 sectioning: `decode_axon.build_strong_sections(aff, z0, z1, thr=0.3) -> np.uint32`
```
full_cache = OUTDIR / f"sections_thr{thr}_z0-{DEPTH}.h5"       # the only cache that exists (z0-800)
if full_cache.exists(): return h5[full_cache][main][z0:z1].astype(uint32)   # slice the GLOBAL cache
shape = (z1-z0, aff.shape[2], aff.shape[3])
seg2d = waterz_2d_spacefill(aff, np.ones(shape,bool), thr, SMALL, AFF_LOW, rg_zero=RG_ZERO,
                            score=getScoreFunc("aff30_his256_ran255"), aff_bg=AFF_BG)   # SMALL=150,AFF_LOW=.01,AFF_BG=.66,RG_ZERO=False
return seg2d.astype(uint32)
```
Takes the already-loaded `aff` (no re-read); `shape` defined from `aff`.

### 3. Tier 2 — 3D cleanup (SPLIT then MERGE), all pure + measured defaults
- **`decode_v4_split.prob1_carve(seg, affxy, *, tau=0.20, min_excess=200, max_len=30, med_w=31,
  area_tol=0.5, min_slices=40, bump_safe=True) -> np.uint32`** — current `main` body minus h5 I/O.
  Restriction to incomplete tubes is intrinsic (`nf.get(id,0)<2` for both intruder halves; the through-
  intruder detector only fires on a bump bracketed by baseline). NEW `bump_safe` guard (default True):
  after computing a candidate carve+relink, verify with the module `bumps` fn that the host bump over the
  run is removed and NO new bump appears on host or the two relinked orphan halves; if the guard fails,
  SKIP that candidate. `bump_safe=False` reproduces the current script exactly (for the equality test);
  `decode_3d` uses `bump_safe=True`. `main()` is a thin wrapper calling `prob1_carve(bump_safe=False)`.
- **`tube_extend.prob1_extend(seg, affxy, *, min_vox=2000, area_tol=0.5, host_margin=0.3, mem_thr=1.0,
  max_steps=60) -> np.uint32`** — extract the current `main` bumpsafe full-apply loop verbatim: `zr=
  compute_bbox_all_3d`, `area=per_slice_area(seg)`, `nf=face_counts(seg)` (computed ONCE — this is the
  exact behavior that produced the measured `decode_p1p2eg`; carves only shrink hosts / grow orphans and
  `gate_bumpsafe` re-checks against the incrementally-updated `area`, so static `zr/nf` is intentional and
  validated, NOT a bug); `bp=BUMP_PARAMS` (module constant). For each orphan (touch<2 faces & size≥min_vox,
  biggest-first) and each `open_ends(seg,oid,zr[oid])`, `extend_end(...)` in BOTH ±z, `gate_bumpsafe(area,
  oid,carved,bp)`, commit voxels + update `area` on accept. `main()` calls this (reproduces 27-accept/
  97-reject / `decode_p1p2eg`).
- **`decode_v3_merge.prob2_merge(seg, aff, *, aff_thr=0.5, min_contact=20, min_size=500, max_zoverlap=0.5,
  area_tol=None) -> np.uint32`** — `region_graph` gets a SIGNATURE CHANGE `region_graph(seg, aff)`: it now
  takes the RAW affinity array `(3,Z,Y,X)` in memory (already z-sliced by the caller) instead of reading
  the full file, so a sub-slab uses the correct raw slices (fixes the z0 offset); channel-correct 3-dir
  boundary-mean aggregation is otherwise unchanged. Merge (a,b) via union-find iff `contact≥min_contact
  AND mean_aff≥aff_thr AND nf[a]<2 AND nf[b]<2 AND z_overlap_frac(zr[a],zr[b])≤max_zoverlap` (parallel
  veto, checks on ORIGINAL labels). NEW `area_tol`: when set, ALSO require per-tube caliber match
  `|cal[a]-cal[b]| ≤ area_tol*max(cal[a],cal[b])` where `cal[s]=size[s]/(zmax[s]-zmin[s]+1)` (mean voxels
  per occupied slice). `decode_3d` calls `prob2_merge(..., area_tol=0.5)` to HONOR the task's area-match;
  `main()` defaults `area_tol=None` to byte-reproduce `decode_p1p2` for the equality test.
- **Composition (decode_axon):** `decode_3d(seg, aff) = prob2_merge(prob1_extend(prob1_carve(seg,
  affxy(aff)), affxy(aff)), aff, area_tol=0.5)` — SPLIT (carve→extend) then MERGE per the user's sketch.
  Toggles `--no-prob1`/`--no-prob2` skip the respective stage.

### 4. Crumb cleanup (drop-only, deterministic) — decode_axon
```
def crumb_cleanup(vol3d, *, min_vox=500) -> np.uint32:   # relabel every seg with size<min_vox to 0
```
`--crumb-absorb` is DEFERRED: if passed, raise NotImplementedError("crumb absorption deferred to
decode_axon weak-recovery follow-up"). Drop-only is the safe default; report dropped count.

### 5. Orchestrator + CLI (decode_axon)
```
run_pipeline(args):
  z0,z1 = (0,DEPTH) if args.full else parse_zslice(args.zslice)
  aff = load_aff(z0,z1); axy = affxy(aff)
  seg2d = build_strong_sections(aff, z0, z1, args.thr)
  v1 = decode_v2.decode_sections(seg2d, axy);                 report("t1_strong", v1)
  v2 = v1 if args.no_prob1 and args.no_prob2 else decode_3d_with_toggles(v1, aff, axy, args); report("t2_3d", v2)
  if args.weak: raise NotImplementedError("weak recovery deferred ...")
  out = v2 if args.no_crumb else crumb_cleanup(v2);           report("final", out)
  save(OUTDIR/f"decode_axon{args.tag}.h5", out)
  if args.full: print liconn_nerl.evaluate(saved, do_oracle_merge=False)[0], ...[True][0]   # secondary
report(name, seg): print valid_tube_metric.score_array(seg, **METRIC_DEFAULTS)   # array-level, no temp h5
```
- Metric: ADD `valid_tube_metric.score_array(seg, min_z=20, min_vol=10000, tau=0.20, min_excess=200,
  max_len=30, med_w=31, min_slices=40, min_cc=50, multi_thr=0.3, list_invalid=False)`; refactor existing
  `score(path,...)` to load the h5 then delegate to `score_array` (behavior unchanged).
- CLI mirrors `decode_v2.py`: REQUIRED mutually-exclusive `--full` | `--zslice Z0:Z1` | `--self-test`;
  `--tag` (literal suffix); `--thr 0.3`; toggles `--no-prob1`, `--no-prob2`, `--no-crumb`, `--weak`
  (→NotImplementedError), `--crumb-absorb` (→NotImplementedError). `--self-test` runs and returns BEFORE
  any file I/O. `--zslice` asserts `0≤z0<z1≤DEPTH` AND a non-empty `--tag` (so a slab can never overwrite
  the full `decode_axon.h5`); slab runs are labeled "SMOKE — face counts not full-volume comparable"
  because Prob-1/Prob-2/crumb face logic (`face_counts`) is only meaningful on the full 800³. The full run
  (z0=0,z1=DEPTH) uses real volume faces and is the authoritative artifact.

### 6. Self-test (data-free, decode_v2 style) — shipped path only
`self_test()` asserts: (a) strong/weak masks partition fg, are disjoint; (b) `prob1_carve` carves a
synthetic through-intruder, produces NO carve on a solid tube, and with `bump_safe=True` REJECTS a
synthetic candidate whose carve would add a bump; (c) `prob1_extend`+`gate_bumpsafe` accept a bump-
removing carve and reject a bump-adding one; (d) `prob2_merge` unions a z-SEQUENTIAL high-aff incomplete
pair (with a synthetic in-memory `aff` injected into `region_graph(seg,aff)`), SKIPS a z-OVERLAPPING
(parallel) pair, and with `area_tol=0.5` REJECTS a caliber-mismatched pair; (e) `crumb_cleanup` drops a
sub-`min_vox` blob, preserves a big one, and never merges; (f) `--weak`/`--crumb-absorb` raise
NotImplementedError.

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_axon.py` | NEW — masks, load_aff, build_strong_sections, decode_3d wiring, crumb_cleanup(drop-only), run_pipeline, self_test, CLI, deferred `--weak`/`--crumb-absorb` stubs |
| `dev/mit_liconn/decode_v2.py` | ADD pure `decode_sections(...)`; `_run_pipeline` calls it (no behavior change) |
| `dev/mit_liconn/decode_v4_split.py` | extract `prob1_carve(seg, affxy, ..., bump_safe=True)`; `main()` thin wrapper (`bump_safe=False`) |
| `dev/mit_liconn/tube_extend.py` | extract `prob1_extend(seg, affxy, ...)` from `main`'s bumpsafe loop; `main()` calls it |
| `dev/mit_liconn/decode_v3_merge.py` | `region_graph(seg,aff)` signature change; extract `prob2_merge(seg, aff, ..., area_tol=None)`; `main()` thin wrapper loads aff + calls it |
| `dev/mit_liconn/valid_tube_metric.py` | ADD `score_array(seg, ...)`; `score(path,...)` loads then delegates |
| `dev/mit_liconn/sbatch_decode_axon.sh` | OPTIONAL — full-volume run mirror |

`dev/` and `.agent/` are gitignored → the code-review prompt carries the FULL current contents of
`decode_axon.py` and the diffs of the five modified modules (not a `git diff`).

## Verification Plan
1. `conda run -n pytc python dev/mit_liconn/decode_axon.py --self-test` → all contract tests pass.
2. Behavior-preserving extraction via partition-hash array equality (NOT h5 bytes): `prob1_carve(seg,
   affxy, bump_safe=False)` on `decode_v2.h5` == `decode_p1.h5`; `prob2_merge(seg, aff, area_tol=None)` on
   `decode_p1.h5` == `decode_p1p2.h5`; `prob1_extend` on `decode_p1p2.h5` == `decode_p1p2eg.h5`;
   `decode_sections` == `decode_v2_c_constrained.h5`.
3. Pilot `... --zslice 0:96 --tag _smoke` → runs, prints SMOKE metric (not full-volume comparable).
4. Full `... --full` (inline ~5–8 min or `sbatch_decode_axon.sh`, `-p short -c 8 --mem 90G`) → saves
   `decode_axon.h5`; prints the ablation (t1 strong / t2 3d / final) and `conda run -n pytc python
   dev/mit_liconn/valid_tube_metric.py decode_p1p2 decode_axon`.
5. Gate on the FINAL saved artifact: VALID vol ≥ 62.7 AND parallel ≤ 8 AND bumps ≤ 117 (expected ≈
   `decode_p1p2eg` 65.5 / 89 / 0, possibly better via the Prob-2 area-match). NERL secondary/informational.

## Risks and Questions
- Extraction fidelity is the main risk; §Verification-2 partition-hash equality is the guard (relabeling-
  invariant). If `prob1_carve(bump_safe=True)` rejects any of decode_p1's 45 carves, that is a documented
  improvement, and the `bump_safe=False` equality path still validates the extraction.
- Prob-2 area-match (`area_tol=0.5`, caliber = voxels/occupied-slice) is a NEW gate vs decode_p1p2; if it
  removes real end-to-end merges (VALID vol drops), the coder reports it and `decode_3d` can fall back to
  `area_tol=None` (still ≥ decode_p1p2). It must never re-introduce a parallel merge (the z-overlap veto
  is independent and unchanged).
- `--zslice` face logic is inaccurate on a slab (documented SMOKE-only); the authoritative run is `--full`.
- Deferred weak/absorb tiers are the "less confident" path; scoping them out is deliberate (user's
  confident-first directive) and keeps this artifact verifiable. Follow-up CCC will design weak recovery.
- Assumption: primary metric = valid-tube volume (GT over-split); NERL secondary.

## Changes Since Previous Plan Version
Resolves the plan_v1 review by (i) SCOPING OUT the under-specified weak-bridge, cross-namespace
`bump_safe_merge`, and crumb absorption (findings 5,6,7,8,11 → deferred, stubbed) and (ii) fully
specifying the remaining shipped-core findings:
- #1 prob1_carve: added the `bump_safe=True` post-carve guard + a self-test that rejects an unsafe
  candidate; `bump_safe=False` preserves extraction equality.
- #2 prob1_extend: `bp=BUMP_PARAMS` module constant; documented that static `zr/nf` + incrementally-
  updated `area` is the exact validated `decode_p1p2eg` behavior (not stale-state), extraction from `main`.
- #3 prob2 area-match: defined caliber statistic (voxels/occupied-slice) and `area_tol=0.5` on the shipped
  path; `main` keeps `area_tol=None` for the equality test.
- #4 prob2 raw-affinity transport: `region_graph(seg, aff)` now takes the in-memory z-sliced affinity, so
  the slab z0 offset is correct.
- #9 slab faces: `--full` uses real faces (authoritative); `--zslice` documented SMOKE-only.
- #10 orchestrator: `run_pipeline` loads `aff` once up-front, computes `affxy`, threads both to all tiers;
  `build_strong_sections(aff, ...)` no longer re-reads; `shape` defined from `aff`.
- #12 self-tests: added unsafe-carve rejection, parallel-reject, caliber-mismatch reject, and the
  NotImplementedError stubs; Prob-2 test injects a synthetic in-memory `aff`.
- #13 (minor): prose made consistent — shipped default = strong+3D+drop-crumb (a NEW composition expected
  to ≈ decode_p1p2eg, not byte-identical to any single prior artifact); weak explicitly deferred.
