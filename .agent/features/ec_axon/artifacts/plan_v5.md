# Plan v5

## Summary
Final plan — self-contained (no cross-references to earlier versions). Consolidates the validated axon decode
into the **existing** `branch_merge`/`branch_split` decoders plus a small number of genuinely-new modules,
removes every `dev/mit_liconn` import from `connectomics/`, places generic utilities in their canonical files,
and ships a waterz-vs-axon tutorial. Fixes the four blocking findings of `plan_v4_review`: an **explicit axon
stage sequence** (full `branch_merge` ≠ the validated algorithm), **non-contradictory defaults** (shared
default stays `0.0`; only the axon adapter opts in), **graph-resolvability stated and tested**, and a
**self-contained, executable verification section**.

Verified facts this plan relies on:
- `DecoderRegistry.register()` calls `register_graph_op(name, as_graph_op(fn))`, so **`register_decoder` DOES
  make an op graph-resolvable**, with `as_graph_op` enforcing exactly one input. Multi-input ops must use
  `register_graph_op` directly.
- `GraphNodeConfig` fields are `name`, `op`, `inputs`, `kwargs` — **no `enabled`**.
- NERL tolerance lives at `evaluation.nerl_merge_threshold` (schema default `1`).
- Existing `branch_merge` stages: `_stage1_iou`, `_stage2_best_buddy`, `_stage3_one_sided`, gated by
  `_bbox_touch_mask`; `branch_split` has `_erosion_seed_components`, `_affinity_seed_segmentation`,
  `_assign_parent_from_markers`.

## Scope
- IN: extend the two existing decoders with the missing mechanisms; add sections/linking/v0-seed/weak-gap/
  completion modules; move `seg_stats`+`apply_lut` to `data/processing/bbox.py` and completeness to
  `metrics/`; delete `axon_tracklet.py` + `tutorials/axon_decoding/tracklet.yaml`; add `tutorials/neuron_axon/`;
  tests.
- OUT: `dev/mit_liconn/*` untouched; no algorithm change; no behavior change for existing callers.

## Proposed Changes

### S1. `branch_merge.py` — two new mechanisms, defaults unchanged
- **`complete_sections(seg, *, min_size, zfrag_iou=0.3)`** — new **standalone public function** (NOT called
  from inside `branch_merge`'s default path): absorb fragments (`size < min_size` or span ≤ 2) into the big
  seg they most touch laterally; if the fragment is isolated (it *is* the whole cross-section), fall back to
  the best-IoU z-neighbour. Returns `(seg, n_absorbed)`.
- **`margin: float = 0.0`** parameter threaded into `_stage2_best_buddy`: accept the best partner only if it
  beats the runner-up by `margin`. **Default `0.0` = today's behavior exactly.** Only the axon adapter passes
  `0.15`.
- **No other change to `branch_merge`'s signature, stage order, or defaults.**

### S2. `branch_split.py` — two new split mechanisms
- **`link_cut(seg, *, drop_thr=0.25, w=4, min_size=10000, min_frag=6, recover=1.1)`** — real-IoU change-point
  z-cut (local-min + greedy min-frag spacing; `recover=1.1` disables the persistence gate, matching the
  validated config). Returns `(seg, n_cuts)`.
- **`tunnel_split(seg, *, host_both=False, ...)`** — close-ended tunnel/parallel carve, reusing
  `_erosion_seed_components` and `_assign_parent_from_markers` rather than re-implementing marker watershed.
  Keeps the anchor-slice carve skip. Returns `(seg, n_splits)`.
- Existing `branch_split(...)` is untouched.

### S3. Explicit axon stage sequence (fixes review finding 1)
`decoders/axon/merge.py::axon_merge(seg, afz, *, min_size=2000, aff_lo=0.4, merge_iou=0.45, margin=0.15,
rounds=4, inplace=False)` implements the **validated sequence, not full `branch_merge`**:
```
1. complete_sections(seg, min_size=min_size)                 # branch_merge.complete_sections
2. repeat `rounds` times, until no change:
     rank z-seam partners by real IoU (aff_lo only excludes background)
     accept iff MUTUAL best  AND  best_iou - runner_up_iou >= margin
```
It **must not** run `_stage1_iou` or `_stage3_one_sided` — those are extra merge paths that would bypass or
extend the margin-qualified mutual stage and are not part of the validated algorithm. If the pair-table
machinery (`_slice_overlaps`, `_bbox_touch_mask`) can supply the ranked candidates, reuse it; otherwise
`axon_merge` keeps its own scan and reuses only `complete_sections`. The coder records which applied.
Analogously `decoders/axon/split.py::axon_split(seg, ...)` = `link_cut` then `tunnel_split`, in that order.

### S4. New modules (nothing existing covers these)
| file | contents |
|---|---|
| `decoders/axon/sections.py` | `affxy`, `build_strong_sections` (2D affinity → per-slice sections, `small=0`) |
| `decoders/axon/linking.py` | `decode_sections` closure for `no_force_split=True` only |
| `decoders/axon/base.py` | `axon_v0(aff, thr=0.3)` = `decode_sections(build_strong_sections(...), affxy(aff), no_force_split=True)` |
| `decoders/axon/split.py` | `axon_split` (composes S2) |
| `decoders/axon/merge.py` | `axon_merge` (S3) |
| `decoders/axon/weak.py` | `axon_weak` — bridges **gaps** (branch_merge only acts where pieces touch) |
| `decoders/axon/complete.py` | `axon_complete`, opt-in; docstring states **om-negative** |
**Closure rule:** port only what the entry points transitively reach; if the sections+linking closure exceeds
~600 lines, stop and report the measured size before continuing.

### S5. Canonical placement of generic utilities
- `seg_stats(seg, want_centroids=False)`, `apply_lut(seg, lut, chunk=64)` → **`connectomics/data/processing/
  bbox.py`** (beside `compute_bbox_all`).
- `completeness_report(seg, ...)` → **`connectomics/metrics/completeness.py`**.
- Migrating `branch_merge`/`branch_split` onto `seg_stats` is **optional and gated** on the S7.1 regression
  passing byte-identically; skip it if that cannot be shown.

### S6. Registration + graph resolvability (fixes finding 3)
In `registry.py::_register_builtins()` (lazy, `_BUILTINS_REGISTERED`-guarded), function calls only:
- `register_decoder("axon_v0", axon_v0, overwrite=True)` — 1 input `[raw]`
- `register_decoder("axon_split", axon_split, overwrite=True)` — 1 input `[seg]`
- `register_decoder("axon_complete", axon_complete, overwrite=True)` — 1 input `[seg]`
- `register_graph_op("axon_merge", _axon_merge_op, overwrite=True)` — 2 inputs `[raw, seg]`
- `register_graph_op("axon_weak", _axon_weak_op, overwrite=True)` — 2 inputs `[raw, seg]`
`register_decoder` is graph-resolvable via `as_graph_op` (arity 1 enforced) — **a test must assert all five
names resolve and execute inside a real `decoding.graph` DAG**. Adapters validate arity and unpack
`(seg=inputs[1], afz=clip(inputs[0][0]))` / `(seg=inputs[1], fgmax=clip(inputs[0][:3]).max(0))`, returning
seg only (cores return `(seg, n)`).
Remove the four `axon_tracklet_*` imports/registrations; delete `decoders/axon_tracklet.py` and
`tutorials/axon_decoding/tracklet.yaml` (audited: those are the only reference sites).

### S7. Verification — self-contained and executable (fixes finding 4)
1. **Golden regression, omitted-args (GATE 1).** Capture `branch_merge(seg)` and `branch_split(seg)` outputs
   on a fixture **before** any edit (pickled/`.npy` golden). After the change, call with the **old signature,
   all new arguments omitted**, and assert `np.array_equal` with the golden. Also run the existing decoder
   unit tests. This is what proves existing callers are unaffected.
2. **Dev-freedom (GATE 2).** AST scan asserting no module under `connectomics/` imports `dev`; and
   `decoders/axon/` imports nothing from `connectomics.training` / `connectomics.evaluation`.
3. **Parity fixtures (GATE 3).** Small deterministic synthetic volumes; ported function vs the
   `dev/mit_liconn` function on identical input, asserting exact equality, for: `complete_sections` (lateral
   **and** z-isolated), margin rejection (an ambiguous pair must NOT merge), `link_cut` (local-min +
   min-frag), `tunnel_split` (`host_both=False`, anchor-slice skip), `axon_weak` (gap projection),
   `axon_complete` (`prefer_length` both ways), `completeness_report`, `build_strong_sections`,
   `decode_sections`, `seg_stats` (vs `compute_bbox_all`+`bincount`; background label, sparse/high IDs,
   centroids), `apply_lut` (vs `lut[seg]`; non-divisible chunk boundary, in-place identity, dtype).
4. **Graph execution (GATE 4).** Build the `axon_decode.yaml` DAG on a small volume and run it end-to-end;
   assert all five ops resolve, arity errors raise, and `output: v3` produces a labelling.
5. **Perf (GATE 5).** `cc3d.statistics` call-count instrumentation: a 3-stage batch calls it
   `1 + #stages returning n>0`, never once per stage. `tracemalloc` peak around `apply_lut` on a ≥256³ volume
   < 25% of the volume's nbytes. Documented full-volume benchmark accepting wall < 150 s and peak RSS < 28 GB.
6. **Config (GATE 6).** `python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'`;
   confirm deleting `tutorials/axon_decoding/tracklet.yaml` breaks nothing.
7. **Full-volume exact parity (RELEASE GATE).** `dev/mit_liconn/verify_port.py`: each stage vs the on-disk
   research output (`decode_tracklet_v0_sm0.h5`, `v1_lcc_confident.h5`, `v2_merge.h5`, `v3_weak.h5`) by
   chunked `np.array_equal`, else relabel-invariant partition equality; pins affinity sha256, channel order,
   dtype, kwargs. Metrics secondary: v0 0.8284/0.9424 · v1 0.7302/0.9631 · v2 0.8377/0.9541 ·
   v3 0.8434/0.9525 · waterz 0.6530/0.7580 (all at `nerl_merge_threshold=10`).

### S8. Tutorial `tutorials/neuron_axon/`
- `axon_decode.yaml` — `v0(axon_v0)[raw] → v1(axon_split)[v0] → v2(axon_merge)[raw,v1] → v3(axon_weak)[raw,v2]`,
  `output: v3`; **no `enabled` key** (unsupported).
- `axon_decode_preferlen.yaml` — adds `v4(axon_complete)[v3] {prefer_length: true}`, `output: v4`; header
  states it is om-negative.
- `waterz_baseline.yaml` — `decode_waterz`, **same** `decoding.load_prediction_path`.
- All set `evaluation.nerl_merge_threshold: 10` and the GT path; run via `scripts/main.py --mode test`.
- `README.md` — exact commands, identical-input statement, the comparison table, thr=10 rationale, how to stop
  early (repoint `output:`).

## Files and Areas
| path | action |
|---|---|
| `connectomics/data/processing/bbox.py` | +`seg_stats`, +`apply_lut` |
| `connectomics/metrics/completeness.py` | create |
| `connectomics/decoding/decoders/branch_merge.py` | +`complete_sections`, +`margin=0.0` |
| `connectomics/decoding/decoders/branch_split.py` | +`link_cut`, +`tunnel_split` |
| `connectomics/decoding/decoders/axon/{__init__,base,sections,linking,split,merge,weak,complete}.py` | create |
| `connectomics/decoding/registry.py::_register_builtins` | −4 axon_tracklet, +5 axon |
| `connectomics/decoding/decoders/axon_tracklet.py`, `tutorials/axon_decoding/tracklet.yaml` | delete |
| `tutorials/neuron_axon/*` | create |
| `tests/unit/test_axon_decoder_parity.py` | create (GATES 1–5) |
| `dev/mit_liconn/verify_port.py` | create (release gate) |

## Verification Plan
Run gates in order 1→7; 1–6 must pass in CI, 7 is the manual release gate. Any inequality is a port bug — fix
the port, never the tolerance. If GATE 1 cannot be made byte-identical, do **not** modify the shared decoder:
put the mechanism in `decoders/axon/` and leave `branch_merge`/`branch_split` alone.

## Risks and Questions
- **R1 shared-code refactor** — mitigated by GATE 1 and the fallback in the Verification Plan.
- **R2 stage-sequence reuse** — if `branch_merge`'s pair table cannot supply ranked candidates with runner-up
  info, `axon_merge` keeps its own scan (S3); record which.
- **R3 closure size** — report if sections+linking exceeds ~600 lines.
- **R4 label determinism** — fall back to partition equality; record which applied.
- No open questions.

## Changes Since Previous Plan Version
Fixes the four blocking findings of `plan_v4_review`: (1) **explicit axon stage sequence** — `axon_merge` runs
completion + margin-qualified mutual-best only and is forbidden from invoking `_stage1_iou`/`_stage3_one_sided`
(plan_v4 wrapped full `branch_merge`, which is not the validated algorithm); (2) **defaults de-contradicted** —
the shared `margin` default is **`0.0`** (today's behavior) with `0.15` passed only by the axon adapter, and
`complete_sections` is a standalone function never called from `branch_merge`'s default path, with the
regression now exercising the **old signature with new args omitted**; (3) **graph resolvability** stated
(`register_decoder` → `as_graph_op`, arity 1) and covered by a new end-to-end DAG execution gate; (4)
**verification made self-contained and executable** — seven numbered gates with concrete oracles (pre-edit
goldens), pass criteria, perf thresholds, and an explicit fallback that abandons the shared-code edit rather
than weakening the regression.
