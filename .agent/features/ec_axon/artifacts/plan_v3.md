# Plan v3

## Summary
Supersedes plan_v2 after the user amendment: **no `dev/mit_liconn` import may survive anywhere in the package**
(including the v0 seed), and **each function goes to its canonical file** rather than an axon-only silo. This
turns the change from "vendor v1–v3" into "vendor the whole decode path", so `decoders/axon_tracklet.py` and
`tutorials/axon_decoding/tracklet.yaml` are **deleted** rather than kept.

Placement principle: generic utilities live with their peers (bbox helpers with `compute_bbox_all`, the GT-free
metric with the other metrics); only axon-specific decode stages live under `decoders/axon/`.

## Scope
- IN: port the transitive closure of the v0 seed + v1/v2/v3 (+opt-in v4) stages, the perf helpers, and the
  completeness metric; delete the dev-path wrapper and its tutorial; register the ops; ship
  `tutorials/neuron_axon/`; parity fixtures + full-volume verification.
- OUT: `dev/mit_liconn/*` stays untouched and working (reproduction path); no algorithm change; no GT change.

## Proposed Changes

### P1. Canonical placement (the user's core point)
| function(s) | source | **destination (canonical)** | why |
|---|---|---|---|
| `seg_stats`, `apply_lut` | `bbox_fast.py` | **`connectomics/data/processing/bbox.py`** | joins `compute_bbox_all` — bbox/label utilities already live here (user's example) |
| `completeness_report` | `completeness.py` | **`connectomics/metrics/completeness.py`** | it is a metric, not a decoder |
| `affxy`, `build_strong_sections` | `decode_axon.py` | **`connectomics/decoding/decoders/axon/sections.py`** | 2D affinity→section construction, axon-specific |
| `decode_sections` (+ closure: `_raw_iou_table`, `_link_recipe`, `_apply_links`, `_pair_array`, `_decode_sections_core`, and only the helpers those reach with `no_force_split=True`) | `decode_v2.py` | **`connectomics/decoding/decoders/axon/linking.py`** | cross-slice linking, axon-specific |
| `link_cut_change`, `confident_parallel_split` | `link_cut_change.py`, `confident_split.py` | `decoders/axon/split.py` | |
| `v2_merge` (+`_complete`) | `v2_merge.py` | `decoders/axon/merge.py` | |
| `v3_weak` | `v3_weak.py` | `decoders/axon/weak.py` | |
| `v4_complete` | `v4_complete.py` | `decoders/axon/complete.py` | opt-in; docstring states **om-negative** |

**Closure rule:** port only what the entry points transitively reach. `decode_sections(..., no_force_split=True)`
must **not** drag in `force_split`/`eligible_fusions`/`constrained_relabel` if that path never calls them — the
coder verifies by import-pruning and the parity fixtures. If any of them *is* reached, port it into
`linking.py` too; do not import from `dev`.

### P2. v0 seed op (replaces plan_v2's D1 compromise)
`decoders/axon/base.py::axon_v0(aff, *, thr=0.3)` = `decode_sections(build_strong_sections(aff, 0, Z, thr,
small=0), affxy(aff), no_force_split=True)` — i.e. `tracklet_base` verbatim, now dev-free. Registered as
`axon_v0`. **`decoders/axon_tracklet.py` is deleted**, its 4 imports/registrations removed from
`registry.py::_register_builtins()`, and `tutorials/axon_decoding/tracklet.yaml` deleted (reference audit
already established these are the only 3 sites).

### P3. Registration (unchanged from plan_v2 D2, plus `axon_v0`)
In `registry.py::_register_builtins()` (lazy, `_BUILTINS_REGISTERED`-guarded), function calls only:
- `register_decoder("axon_v0", axon_v0, overwrite=True)` — unary `[raw]`
- `register_decoder("axon_split", axon_split, overwrite=True)` — unary `[seg]`
- `register_decoder("axon_complete", axon_complete, overwrite=True)` — unary `[seg]`
- `register_graph_op("axon_merge", _axon_merge_op, overwrite=True)` — `[raw, seg]`
- `register_graph_op("axon_weak", _axon_weak_op, overwrite=True)` — `[raw, seg]`
Adapters validate arity and unpack `(seg=inputs[1], afz=clip(inputs[0][0]))` /
`(seg=inputs[1], fgmax=clip(inputs[0][:3]).max(0))`, returning seg only.

### P4. Contracts (carried from plan_v2 D3, unchanged)
Cores return `(seg, n)`; adapters drop `n`. `inplace=False` copies by default, `True` mutates and returns the
same object. `apply_lut` mutates in place, returns the same object, casts `lut` to `seg.dtype`. **Staleness:**
`stats` is valid only while every stage since has returned `n == 0`; any `n > 0` invalidates it (documented +
exercised by a batch-driver test). The bogus `max(zr) <= seg.max()` assertion stays removed.

### P5. Tutorial (carried from plan_v2 D5, with `axon_v0` as the first node)
`tutorials/neuron_axon/`: `axon_decode.yaml` (`v0(axon_v0)[raw] → v1(axon_split)[v0] → v2(axon_merge)[raw,v1]
→ v3(axon_weak)[raw,v2]`, `output: v3`, **no `enabled` key** — `GraphNodeConfig` has none),
`axon_decode_preferlen.yaml` (adds `v4(axon_complete)[v3] {prefer_length: true}`, `output: v4`, labelled
om-negative), `waterz_baseline.yaml` (`decode_waterz`, same `load_prediction_path`), `README.md` (commands,
identical-input statement, comparison table, thr=10 rationale). All set
**`evaluation.nerl_merge_threshold: 10`** (verified path) and run via `scripts/main.py --mode test`.

### P6. Verification (carried from plan_v2 D6, extended to the newly ported code)
1. **Exact parity (release gate)** `dev/mit_liconn/verify_port.py`: each ported stage vs the on-disk research
   outputs — now including **v0** (`decode_tracklet_v0_sm0.h5`) — by chunked `np.array_equal`, else
   relabel-invariant partition equality. Pins affinity sha256, channel order, dtype, kwargs.
2. **CI parity fixtures** (small synthetic, deterministic): the sensitive gates (local-min/min-frag,
   `host_both=False`, anchor-slice carve, lateral + z-isolated completion, mutual/margin rejection, weak-gap
   projection, `recover=False`), **plus** v4 (`prefer_length` both ways), completeness, **and now
   `build_strong_sections`/`decode_sections`** (ported vs research function on identical input).
3. **Helper parity:** `seg_stats` vs `compute_bbox_all`+`bincount`; `apply_lut` vs `lut[seg]` (chunk
   boundaries, identity, dtype, background label, sparse/high IDs).
4. **Perf with thresholds:** `cc3d.statistics` call-count (batch = `1 + #stages with n>0`, never per-stage);
   `tracemalloc` peak around `apply_lut` on ≥256³ < 25% of volume nbytes; documented full-volume benchmark
   accepting wall < 150s and peak RSS < 28 GB.
5. **Dev-freedom (the amendment's acceptance test):** a test asserting **no module under `connectomics/`
   imports `dev`** (AST scan, not just grep), and that `decoders/axon/` imports nothing from
   `connectomics.training` / `connectomics.evaluation`. Plus guardrail + API-snapshot tests.
6. **Configs:** `validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'`; confirm the deleted
   `tutorials/axon_decoding/tracklet.yaml` breaks nothing.

## Files and Areas
| path | action |
|---|---|
| `connectomics/data/processing/bbox.py` | **+`seg_stats`, +`apply_lut`** |
| `connectomics/metrics/completeness.py` | create |
| `connectomics/decoding/decoders/axon/{__init__,base,sections,linking,split,merge,weak,complete}.py` | create |
| `connectomics/decoding/registry.py::_register_builtins` | −4 axon_tracklet regs, +5 axon regs |
| `connectomics/decoding/decoders/axon_tracklet.py` | **delete** |
| `tutorials/axon_decoding/tracklet.yaml` | **delete** |
| `tutorials/neuron_axon/{axon_decode,axon_decode_preferlen,waterz_baseline}.yaml + README.md` | create |
| `tests/unit/test_axon_decoder_parity.py` | create |
| `dev/mit_liconn/verify_port.py` | create |

## Verification Plan
Gate order: (1) dev-freedom AST test + boundary tests; (2) CI parity fixtures + helper parity; (3) config
validation; (4) instrumented perf; (5) full-volume exact parity (release gate), metrics secondary
(v0 0.8284/0.9424 · v1 0.7302/0.9631 · v2 0.8377/0.9541 · v3 0.8434/0.9525 · waterz 0.6530/0.7580 @thr=10).
Any inequality is a port bug — fix the port, never the tolerance.

## Risks and Questions
- **R1 (new, main risk) — closure size.** `decode_sections` lives in an 810-line module and
  `build_strong_sections` in a 692-line one. If the `no_force_split=True` closure turns out to reach most of
  both, the port is large. Mitigation: port strictly by closure, and if it exceeds ~600 lines total, the coder
  reports back with the measured closure before continuing rather than silently ballooning the change.
- **R2 — waterz dependency.** `build_strong_sections` calls the 2D waterz section builder; the package already
  depends on waterz via `decoders/waterz.py`, so this should be reusable — the coder must confirm it uses the
  packaged waterz entry point rather than a dev-local copy.
- **R3 label determinism** (unchanged): if waterz ordering is not reproducible, parity uses partition equality.
- No open questions.

## Changes Since Previous Plan Version
Rewritten for the user amendment: **v0 is now fully ported** (`decoders/axon/base.py` + `sections.py` +
`linking.py`) instead of plan_v2's compromise of reusing the dev-wrapping `axon_tracklet_base`; **the dev
dependency is eliminated outright**, so `axon_tracklet.py` and `tutorials/axon_decoding/tracklet.yaml` are
**deleted** (plan_v2 kept them); **placement is now canonical rather than siloed** — `seg_stats`/`apply_lut`
go to `connectomics/data/processing/bbox.py` beside `compute_bbox_all` and `completeness_report` to
`connectomics/metrics/`, instead of living under `decoders/axon/`; added a **transitive-closure rule** so whole
modules are not dragged in; added `axon_v0` registration; extended parity to v0 and the newly ported
sections/linking; and added the **dev-freedom AST test** as the amendment's explicit acceptance criterion.
New risk R1 (closure size) with a report-back threshold, and R2 (packaged-waterz reuse).
