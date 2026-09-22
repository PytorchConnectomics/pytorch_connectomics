# Plan v4

## Summary
Supersedes plan_v3 after the user's second amendment: **`decoders/branch_merge.py` (528 lines) and
`decoders/branch_split.py` (258 lines) already own these concepts**, so the change is a **refactor +
redesign**, not a new parallel implementation. The validated research work contributes *mechanisms*, which are
folded into the existing decoders; only genuinely new capabilities get new modules.

Overlap audit (why this is a refactor):
- `branch_merge` already does: z-boundary pair table with `mean_affinity`, the **bbox-touch filter**
  (merge only where a segment ends/begins), `_stage1_iou`, **`_stage2_best_buddy` (mutual best match)**,
  `_stage3_one_sided`. That *is* the v2 merge concept — and it already treats affinity as an optional
  *validator* rather than the selector, which matches the session's finding that shape must select.
- `branch_split` already does seeded splitting: `_erosion_seed_components`, `_affinity_seed_segmentation`,
  `_assign_parent_from_markers` (marker watershed) — the same machinery `confident_split` uses.

So v2_merge is **not** a new decoder; it is `branch_merge` + 2 missing mechanisms. Same for the split side.

## Scope
- IN: extend `branch_merge`/`branch_split` with the validated mechanisms; add the genuinely-new stages
  (weak-gap bridge, z-seam link-cut, v0 seed) as their own modules that **reuse** those helpers; canonical
  placement of generic utilities; delete the dev-path wrapper; tutorial; parity tests.
- OUT: `dev/mit_liconn/*` untouched; no algorithm change to the ported logic; no duplicate merger/splitter.

## Proposed Changes

### C1. `branch_merge.py` — add the 2 missing mechanisms (REFACTOR, no new merger)
1. **`complete_sections(seg, min_size, zfrag_iou=0.3)`** — new public pre-pass in this module: absorb
   cross-section fragments (`size < min_size` or span ≤ 2) into the big seg they most touch **laterally**,
   falling back to the best-IoU **z-neighbour** when the fragment is isolated (it *is* the whole
   cross-section). This is the session's biggest om-safe win (base +0.009, om +0.001) and the fix for
   "small segs are second-class citizens". Exposed as a `branch_merge(..., complete_first=True)` option.
2. **`margin` on `_stage2_best_buddy`** — the best partner must beat the runner-up by `margin` (default
   `0.15`); otherwise leave it split. Measured: rejects 5 ambiguous merges → **om +0.002 at zero base cost**.
   Implements the user's rule *"no other close/confusing match"*.
Both are additive and default-off-compatible: existing callers keep current behavior unless they opt in.

### C2. `branch_split.py` — add the two axon split mechanisms, reusing its seed/watershed helpers
1. **`link_cut(seg, *, drop_thr=0.25, w=4, min_frag=6, ...)`** — z-seam cut at a consecutive-slice **real-IoU
   change-point** (local-min + greedy min-frag spacing; persistence OFF). New *mechanism* (pure z-cut), but it
   belongs here because this module owns "splitting an over-merged label".
2. **`tunnel_split(seg, ...)`** (from `confident_parallel_split`) — close-ended parallel/tunnel carve, reusing
   `_erosion_seed_components` / `_assign_parent_from_markers` instead of re-implementing marker watershed.
   Keeps `host_both=False` and the anchor-slice carve skip verbatim.

### C3. New modules only where nothing exists
| new file | why nothing existing covers it |
|---|---|
| `decoders/axon/sections.py` (`affxy`, `build_strong_sections`) | 2D affinity → per-slice sections; no packaged equivalent |
| `decoders/axon/linking.py` (`decode_sections` closure, `no_force_split=True`) | cross-slice linking into 3D tracklets |
| `decoders/axon/base.py` (`axon_v0`) | composes the two above = the v0 seed (`tracklet_base`) |
| `decoders/axon/weak.py` (`axon_weak`) | bridges **gaps**; `branch_merge` only acts where segments touch |
| `decoders/axon/complete.py` (`axon_complete`) | opt-in radius link; docstring states **om-negative** |

### C4. Canonical placement of generic utilities (unchanged from plan_v3 P1)
- `seg_stats`, `apply_lut` → **`connectomics/data/processing/bbox.py`** (beside `compute_bbox_all`).
- `completeness_report` → **`connectomics/metrics/completeness.py`**.
- `branch_merge`/`branch_split` should be migrated onto `seg_stats` where they currently recompute
  bbox/size tables, provided parity tests pass (perf win, no behavior change).

### C5. Dev-freedom + deletion (unchanged from plan_v3 P2)
No `connectomics/` module imports `dev` when done. `decoders/axon_tracklet.py` and
`tutorials/axon_decoding/tracklet.yaml` are **deleted**; their 4 registrations removed from
`registry.py::_register_builtins()`.

### C6. Registration (adjusted to the consolidated design)
In `registry.py::_register_builtins()`, function calls only:
- `register_decoder("axon_v0", axon_v0)` — `[raw]`
- `register_decoder("axon_split", axon_split)` — `[seg]`; `axon_split` = `link_cut` then `tunnel_split`
  (thin composition living in `decoders/axon/split.py`, calling into `branch_split.py`)
- `register_graph_op("axon_merge", _axon_merge_op)` — `[raw, seg]`; wraps
  `branch_merge(..., complete_first=True, margin=0.15)`
- `register_graph_op("axon_weak", _axon_weak_op)` — `[raw, seg]`
- `register_decoder("axon_complete", axon_complete)` — `[seg]`
Adapters validate arity, unpack `(seg=inputs[1], afz=clip(inputs[0][0]))` /
`(fgmax=clip(inputs[0][:3]).max(0))`, and return seg only. Existing `branch_merge`/`branch_split`
registrations stay unchanged.

### C7. Contracts, tutorial, verification
Carried verbatim from plan_v3 P4/P5/P6: `(seg, n)` cores + adapters dropping `n`; `inplace` semantics and the
`n > 0` staleness rule; `tutorials/neuron_axon/{axon_decode, axon_decode_preferlen, waterz_baseline}.yaml` +
README with `evaluation.nerl_merge_threshold: 10`; parity fixtures for every sensitive gate; helper parity;
instrumented perf thresholds; **dev-freedom AST test**; full-volume exact-parity release gate.
**Added by this plan:** a **regression test that existing `branch_merge`/`branch_split` behavior is unchanged
when the new options are off** (`complete_first=False`, `margin=0.0`) — this is the guard that makes a refactor
of shared code safe.

## Files and Areas
| path | action |
|---|---|
| `connectomics/data/processing/bbox.py` | +`seg_stats`, +`apply_lut` |
| `connectomics/metrics/completeness.py` | create |
| `connectomics/decoding/decoders/branch_merge.py` | +`complete_sections`, +`margin`, +`complete_first` |
| `connectomics/decoding/decoders/branch_split.py` | +`link_cut`, +`tunnel_split` (reuse seed/watershed helpers) |
| `connectomics/decoding/decoders/axon/{__init__,base,sections,linking,split,weak,complete}.py` | create |
| `connectomics/decoding/registry.py::_register_builtins` | −4 axon_tracklet, +5 axon |
| `connectomics/decoding/decoders/axon_tracklet.py`, `tutorials/axon_decoding/tracklet.yaml` | delete |
| `tutorials/neuron_axon/*` | create |
| `tests/unit/test_axon_decoder_parity.py` | create (incl. the no-op-when-off regression) |
| `dev/mit_liconn/verify_port.py` | create |

## Verification Plan
Gate order: (1) **existing-behavior regression** for `branch_merge`/`branch_split` with new options off;
(2) dev-freedom AST test + boundary tests; (3) parity fixtures (sensitive gates, v4, completeness, sections/
linking) + helper parity; (4) config validation; (5) instrumented perf; (6) full-volume exact parity (release
gate), metrics secondary (v0 0.8284/0.9424 · v1 0.7302/0.9631 · v2 0.8377/0.9541 · v3 0.8434/0.9525 · waterz
0.6530/0.7580 @thr=10). Any inequality is a port bug — fix the port, never the tolerance.

## Risks and Questions
- **R1 (new, main) — refactoring shared code.** `branch_merge`/`branch_split` have existing callers/configs;
  adding options must not perturb defaults. Mitigated by the gate-(1) regression test; if `complete_sections`
  cannot be added without touching the default path, add it as a separate public function invoked by the axon
  adapter instead of inside `branch_merge`.
- **R2 — mechanism mismatch.** `branch_merge`'s pair-table architecture may not express the margin cleanly if
  its best-buddy stage lacks runner-up information; then the margin is computed in the same pass that ranks
  partners, or the axon adapter uses a dedicated code path. Coder decides after reading the stage, and records
  which.
- **R3 closure size** (from plan_v3): if the `decode_sections`/`build_strong_sections` closure exceeds ~600
  lines, stop and report before continuing.
- **R4 label determinism**: parity falls back to relabel-invariant partition equality.
- No open questions.

## Changes Since Previous Plan Version
Rewritten after the user flagged the existing `branch_split`/`branch_merge` decoders. plan_v3 would have
**duplicated** them with `decoders/axon/{split,merge}.py`; this plan instead performs the audit (documented in
§Summary), folds the v2 merge's two genuinely-new mechanisms — **cross-section completion** and the
**ambiguity margin** — into `branch_merge` (C1), moves the two split mechanisms (**link-cut change-point**,
**tunnel/confident carve**) into `branch_split` where they reuse its existing seed/watershed helpers (C2), and
restricts new modules to what nothing covers: sections, linking, the v0 seed, the weak-**gap** bridge, and the
opt-in completion link (C3). Registration now wraps the extended existing decoders rather than new ones (C6),
`branch_merge`/`branch_split` are migrated onto `seg_stats` for the perf win (C4), and a new **existing-
behavior regression gate** (options off ⇒ byte-identical results) is added as verification step 1, with new
risks R1/R2 covering shared-code refactor hazards.
