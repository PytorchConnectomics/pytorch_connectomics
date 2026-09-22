# Plan v0

## Summary
Build `dev/mit_liconn/decode_axon.py`: one modular decoder that runs, in order, (1) the strong-fg
2D-section pipeline (decode_v2 steps 1-8: sections → force-split fused → constrained link), (2) the
3D incomplete-tube cleanup (decode_p1p2: Prob-1 bump-safe split, then Prob-2 parallel-vetoed merge),
(3) optional weak-fg recovery that bridges split strong tubes via fastmorph+cc3d weak sections under a
strict hysteresis guard, and (4) a crumb cleanup. Every tier is a separate importable function behind a
flag, so the strong-only output reproduces `decode_p1p2` and each new tier is independently ablatable.
Primary metric is the GT-free `valid_tube_metric.py` (VALID volume %, bumps, parallel-merge segs);
target: beat `decode_p1p2` (62.7% / 117 / 8) without adding parallel merges or contamination.

The only real code moves beyond orchestration: extract the monolithic Prob-1/Prob-2 script cores
(`decode_v4_split.py`, `decode_v3_merge.py`) into importable pure functions (their `main()` becomes a
thin CLI wrapper), and add three genuinely new functions — weak-section building, hysteresis weak
bridge, and crumb cleanup.

## Scope
In scope:
- New file `dev/mit_liconn/decode_axon.py` (orchestrator + new stages + `--self-test`).
- Refactor-extract importable cores from `decode_v4_split.py` (Prob-1) and `decode_v3_merge.py`
  (Prob-2) WITHOUT changing their behavior; keep their CLIs working via thin wrappers.
- Reuse as-is (import, do not copy): `decode_lib` (waterz/link primitives), `decode_v2`
  (`detect_fusions`, `eligible_fusions`, `force_split` via `force_split_decode`, `constrained_relabel`,
  `_raw_iou_table`, `_link_recipe`, `_apply_links`, `section_index`), `tube_extend`
  (`gate_bumpsafe`, `extend_end`, `open_ends`, `face_counts`), `valid_tube_metric` (`score`).
- Optional `dev/mit_liconn/sbatch_decode_axon.sh` mirroring the existing sbatch scripts.

Out of scope:
- Any change to the affinity model, to waterz, or to the linking primitives themselves.
- NERL/GT tuning (GT untrusted); NERL printed as secondary/informational only.
- Weak recovery beyond bridging split strong tubes (no weak-only new axons — hysteresis forbids it).
- Neuroglancer upload / meshing (separate, already-scripted step).

## Proposed Changes

### Foreground split (task a)
- `strong_mask(aff)` → `aff.max(0) > 0.66` (bool ZYX). Matches `decode_lib.AFF_BG`; today's cached
  sections already use this threshold, so strong sections = the existing `sections_thr0.3_z0-800.h5`.
- `weak_mask(aff)` → `(aff.max(0) > 0.30) & (aff.max(0) <= 0.66)`.
- `affxy(aff)` → `aff[1:].mean(0)` (in-plane membrane, reused by force-split, prob1, weak).

### Tier 1 — strong 2D-seg pipeline, split-then-link (task b1, d-strong, steps 1-8)
- `build_strong_sections(z0, z1, thr=0.3)` → uint32 `seg2d`. Reuse `decode_v2._load_inputs` section
  path (`waterz_2d_spacefill`, `aff_bg=0.66`, `fill 0`); use the cache
  `sections_thr0.3_z0-{depth}.h5` when present (steps 1-3).
- `decode_strong(seg2d, affxy)` → uint32 `vol3d`. This is exactly `decode_v2._run_pipeline`'s a→split→c
  without the GT eval: base-link (`_raw_iou_table` → `_link_recipe` → `_apply_links`) to get the spine
  used for upstream eligibility (steps 4-6 base) → `detect_fusions` + `eligible_fusions` + `force_split`
  (steps 7-8) → `constrained_relabel` (final mutex-constrained link, steps 4-6). Returns `volume_c`.
  Rationale: `constrained_relabel`'s per-split mutex is LOAD-BEARING (force-split alone regressed
  0.593→0.553); do not substitute a plain re-link.

### Tier 2 — 3D cleanup for incomplete tubes (task b2 = decode_p1p2, split THEN merge)
- Extract from `decode_v4_split.py`:
  `prob1_carve(seg, affxy, *, tau, min_excess, max_len, med_w, area_tol, min_slices) -> seg` — the
  through-intruder detect+carve+relink core (current `main` body minus h5 load/save/print).
- Add bidirectional orphan-endpoint extension using `tube_extend`:
  `prob1_extend(seg, affxy, *, min_vox=2000, gate="bumpsafe") -> seg` — wraps `tube_extend`'s
  full-apply loop (open_ends → extend_end → `gate_bumpsafe`), commit only bump-safe carves.
- Extract from `decode_v3_merge.py`:
  `prob2_merge(seg, aff_raw, *, aff_thr=0.5, min_contact=20, max_zoverlap=0.5) -> seg` — recompute the
  region graph on RAW affinity (before bg zero-out), union-find merge a pair iff BOTH incomplete
  (touch <2 faces) AND boundary aff ≥ aff_thr AND contact ≥ min_contact AND area-matched AND z-range
  overlap ≤ max_zoverlap (parallel-axon veto via `compute_bbox_all_3d`). The z-overlap veto is
  mandatory (guardless merge fused 584 parallel axons).
- `decode_3d(vol3d, affxy, aff_raw)` = `prob2_merge(prob1_extend(prob1_carve(vol3d)))` — SPLIT before
  MERGE (measured order).

### Tier 3 — weak recovery via hysteresis (task d-weak, e, f) — OFF by default
- `build_weak_sections(aff, weak_mask, *, open_radius, min_cc)` → uint32 `wseg2d`: per z-slice, take
  the weak band, `fastmorph` multi-label/binary opening to break weak salt-bridges, then `cc3d` 2D
  connected components (drop CCs < min_cc). Globally-unique labels per slice (offset like
  `waterz_2d_spacefill`).
- `weak_bridge(vol3d, wseg2d, affxy)` → `vol3d`: for each strong-tube pair that is (a) both INCOMPLETE
  (touch <2 faces), (b) z-SEQUENTIAL (end-of-A z ≈ start-of-B z, z-ranges non-overlapping — reuse the
  prob2 parallel veto), (c) centroid-close and area-matched at the gap, and (d) the intervening z-gap
  is spanned by a connected chain of weak sections IoU-linking A's tail to B's head — merge A and B and
  absorb the bridging weak sections. Hysteresis: a weak section is kept ONLY if it lies on an accepted
  strong-to-strong bridge; unused weak sections are discarded (never seed a new axon, never weak-weak).
  Guard every accepted bridge with the bump-safe check (`gate_bumpsafe`) so a bridge cannot create a
  contamination bump.

### Crumb cleanup (task g)
- `crumb_cleanup(vol3d, *, min_vox=500, orphan_min_vox=2000)` → `vol3d`: relabel-drop segments below
  `min_vox` (dust → 0); optionally absorb tiny orphans (touch <2 faces AND < orphan_min_vox) into an
  adjacent larger seg only when unambiguous and bump-safe. Report counts.

### Orchestrator + CLI
- `run_pipeline(args)`:
  `seg2d = build_strong_sections(...)` → `v1 = decode_strong(...)` → `v2 = decode_3d(v1,...)` →
  `v3 = weak_bridge(v2, build_weak_sections(...))` if `--weak` else `v2` → `out = crumb_cleanup(v3)` →
  save `decode_axon{tag}.h5` → print `valid_tube_metric.score(out)` (+ NERL if `--full`).
  Print the metric after EACH tier (strong / +3D / +weak / +crumb) for the ablation table.
- CLI mirrors `decode_v2.py`: mutually-exclusive `--full` / `--zslice Z0:Z1` / `--self-test`; `--tag`;
  stage toggles `--weak` (default off), `--no-prob1`, `--no-prob2`, `--no-crumb`; tunables with the
  measured defaults above.
- `self_test()`: data-free contract tests in `decode_v2.py` style — (i) strong/weak masks partition fg
  and are disjoint; (ii) `weak_bridge` merges a synthetic sequential split pair but REJECTS a
  z-overlapping (parallel) pair; (iii) `crumb_cleanup` drops a sub-threshold blob and preserves a big
  one; (iv) `prob1_extend` commits a bump-removing carve and rejects a bump-adding one (reuse the
  `gate_bumpsafe` fixture).

## Files and Areas
| File | Change |
|---|---|
| `dev/mit_liconn/decode_axon.py` | NEW — orchestrator, weak-section build, weak bridge, crumb cleanup, self-test, CLI |
| `dev/mit_liconn/decode_v4_split.py` | extract `prob1_carve(...)` importable core; `main()` becomes thin CLI wrapper (no behavior change) |
| `dev/mit_liconn/decode_v3_merge.py` | extract `prob2_merge(...)` importable core; `main()` thin wrapper (no behavior change) |
| `dev/mit_liconn/tube_extend.py` | none needed (already exposes `gate_bumpsafe`/`extend_end`/`open_ends`); optionally factor its full-apply loop into `prob1_extend(...)` here or import as-is |
| `dev/mit_liconn/sbatch_decode_axon.sh` | OPTIONAL — mirror existing sbatch scripts for the full-volume run |
| `.agent/features/decode_axon/` | CCC artifacts only (gitignored) |

Note: `dev/` and `.agent/` are gitignored, so the code-review stage will receive `decode_axon.py`
content inline in the review prompt, not via `git diff`.

## Verification Plan
1. `python dev/mit_liconn/decode_axon.py --self-test` — all contract tests pass.
2. Confirm the extracted cores are behavior-preserving: `decode_v4_split.py`/`decode_v3_merge.py` still
   reproduce `decode_p1` / `decode_p1p2` (byte-compare or valid-tube metric equality vs the existing
   `.h5`).
3. Pilot `--zslice 0:96`: runs, prints valid-tube metric; sanity vs the z0:96 references.
4. Full `--full` (inline ~5–8 min or sbatch): produces `decode_axon.h5`; print the ablation table
   (strong / +3D / +weak / +crumb) and compare to `decode_p1p2` baseline (62.7% / 117 / 8) via
   `python dev/mit_liconn/valid_tube_metric.py decode_p1p2 decode_axon`.
5. Gate: strong+3D ≥ decode_p1p2 (no regression); with `--weak`, VALID volume ↑ AND parallel-merge ≤ 8
   AND bumps ≤ 117. Report NERL as secondary.

## Risks and Questions
- **Prob-1/Prob-2 extraction risk:** their `main()` bodies interleave load/save/print with logic;
  extraction must preserve exact behavior (verify via step 2). Mitigation: keep `main()` calling the
  new core so the CLI path is identical.
- **Weak bridge is the novel, risky tier.** Weak-band bridging on packed parallel axons can
  re-introduce merges; the hysteresis guard (bridge only sequential, area-matched, bump-safe,
  strong-to-strong) is the defense, but it may reject nearly everything (net ~0 gain) or, if too loose,
  add parallel merges. Ship it OFF by default; the ablation isolates its effect; keep it a knob.
- **fastmorph API:** exact opening call (binary vs multi-label, radius, 2D per-slice vs 3D) needs to be
  pinned to the installed version; coder verifies against `import fastmorph` and picks the per-slice 2D
  opening + `cc3d.connected_components(..., connectivity=8)` path.
- **Coverage ceiling:** ~9.5% axons are missed at the model-recall floor (69% of missed are
  no-signal); weak recovery cannot exceed that. Success is measured relative to decode_p1p2, not the
  oracle.
- **Question (non-blocking, default chosen):** primary metric = valid-tube volume (GT untrusted), NERL
  secondary — proceeding on that assumption per the session's late pivot.

## Changes Since Previous Plan Version
Initial plan.
