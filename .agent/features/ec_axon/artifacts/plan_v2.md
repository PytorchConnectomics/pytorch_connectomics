# Plan v2

## Summary
Final plan. Every item the reviewer left open is now **decided against verified repo facts** — no "to be
confirmed", no open questions. Ports the validated v1/v2/v3 (+opt-in v4) axon decode into
`connectomics/decoding/decoders/axon/`, registers them in `registry.py::_register_builtins()`, chains them as
a `decoding.graph` DAG, and ships `tutorials/neuron_axon/` that actually runs the waterz-vs-axon comparison.

Verified facts driving the decisions below:
- `GraphNodeConfig` fields are **`name`, `op`, `inputs`, `kwargs` only — there is no `enabled`** (that flag
  exists for decode *mode* entries via `pipeline._resolve_enabled`). A v4 node with `enabled: false` would
  fail strict config.
- NERL merge tolerance is **`evaluation.nerl_merge_threshold`** (schema default `1`; `evaluation/nerl.py:122`).
- `register_decoder()` unary-wraps via `as_graph_op`; `register_graph_op()` does not. Both are *function
  calls* in `registry.py::_register_builtins()` (lazy, `_BUILTINS_REGISTERED`-guarded).
- Source return shapes: `link_cut_change -> (seg, n_cut)`, `confident_parallel_split -> (seg, n, pairs)`,
  `v2_merge -> (seg, n)`, `v3_weak -> (seg, n)`, `v4_complete -> (seg, n)`.

## Scope
- IN: `decoders/axon/` (7 modules); 4 registrations; `tutorials/neuron_axon/` (3 files + a prefer-length
  variant); parity fixtures; opt-in full-volume verification script.
- OUT: porting `tracklet_base` (see D1); `dev/mit_liconn/*` unchanged; no algorithm changes; no new metric code.

## Proposed Changes

### D1. v0 seed — DECIDED: reuse the existing `axon_tracklet_base` op
The task's port table lists v1 split, v2 merge, v3 weak, v4, perf, metric — **v0 is not in it**. The seed
(`tracklet_base` = 2D waterz sections `small=0` + conservative linking, no crumb cleanup) is a separate,
larger component that is *already* registered as `axon_tracklet_base`.
**Decision:** the DAG's first node is `axon_tracklet_base`; `decoders/axon/` does **not** contain a v0.
**Accepted residual (stated plainly):** `axon_tracklet_base` still imports from `dev/mit_liconn` with a
fallback, so the dev-path dependency is **reduced to the seed only** and is **not** removed by this change.
Recorded as the single follow-up: "port `tracklet_base` → `decoders/axon/base.py`, then delete
`axon_tracklet.py`". `axon_tracklet.py` is therefore **kept**; its v1–v3 ops get a header note marking them
superseded by `decoders/axon/`, and `tutorials/axon_decoding/tracklet.yaml` gets a pointer to
`tutorials/neuron_axon/`. No deletion in this change.

### D2. Registration — reconciled explicitly (finding 6)
Both required forms are *function calls* in `registry.py::_register_builtins()`; the task's constraint is
"function call, **not a decorator**", which both satisfy. Split by arity:
- **unary** (`inputs: [seg]`) → `register_decoder("axon_split", axon_split_op, overwrite=True)` and
  `register_decoder("axon_complete", axon_complete_op, overwrite=True)`.
- **multi-input** (`inputs: [raw, seg]`) → `register_graph_op("axon_merge", _axon_merge_op, overwrite=True)`
  and `register_graph_op("axon_weak", _axon_weak_op, overwrite=True)` — `register_decoder` cannot be used
  here because `as_graph_op` enforces exactly one input.
Adapters validate arity and unpack in a documented order:
`_axon_merge_op(inputs, **kw)` → `axon_merge(seg=inputs[1], afz=np.clip(inputs[0][0],0,1), **kw)`;
`_axon_weak_op(inputs, **kw)` → `axon_weak(seg=inputs[1], fgmax=np.clip(inputs[0][:3],0,1).max(0), **kw)`.
Adapters return **seg only** (discard the source's `n`). Tests: input order, wrong arity raises, cold-start
lazy discovery, duplicate registration.

### D3. `stats` / `inplace` contracts (finding 2)
- Core signatures keep the source shape and **return `(seg, n)`**; only the adapters drop `n`.
- `inplace: bool = False` (default copies, matching source). `inplace=True` mutates and returns the *same
  object* (identity asserted in tests); dtype is always `uint32` in and out.
- `apply_lut(seg, lut, chunk=64)` mutates in place, returns the same object, casts `lut` to `seg.dtype`.
- **Staleness (concrete):** `stats=(zr, sizes)` describes one labelling. A caller may reuse it only while `n
  == 0` for every stage run since it was computed; **any stage returning `n > 0` invalidates it** and the
  caller must recompute. The v1 assertion (`max(zr) <= seg.max()`) was **wrong** (compares bbox to label id)
  and is **removed** — no freshness assertion is claimed; the rule is documented and exercised by a test that
  a stale-stats call is detected by the batch driver, not by the stage.
- Tests: both `inplace` modes for every stage (copy leaves input untouched; in-place returns identity).

### D4. Port mapping + permitted changes
Unchanged from plan_v1 §D (verbatim ports of `bbox_fast`, `link_cut_change`, `confident_split`, `v2_merge`,
`v3_weak`, `v4_complete`, `completeness`; permitted mechanical changes = import rewrites, placement, adapters,
annotations, docstrings, dropping `__main__`; forbidden = renaming/retuning any constant or gate, reordering
logic, simplifying conditions).

### D5. Tutorial — runnable and pinned (findings 3, 4)
`tutorials/neuron_axon/`:
- `axon_decode.yaml` — nodes: `v1(axon_split)[v0] ← v0(axon_tracklet_base)[raw]`, `v2(axon_merge)[raw, v1]`,
  `v3(axon_weak)[raw, v2]`; `output: v3`. **No `enabled` key anywhere** (unsupported by `GraphNodeConfig`).
- `axon_decode_preferlen.yaml` — identical plus `v4(axon_complete)[v3] kwargs:{prefer_length: true}`,
  `output: v4`; header states it is **om-negative** (0.8447/0.9457 vs v3 0.8434/0.9525). A separate file is
  the correct mechanism because nodes cannot be disabled inline.
- `waterz_baseline.yaml` — `decode_waterz` on the **same** `decoding.load_prediction_path`.
- All three set `evaluation.nerl_merge_threshold: 10` (exact verified path) plus the GT path, and are run via
  `python scripts/main.py --config <yaml> --mode test`, which produces NERL base + oracle-merge through the
  existing evaluation stage (decoding never imports NERL).
- `README.md` — the exact commands, the identical-input statement (same `load_prediction_path` in all three),
  the comparison table (waterz 0.6530/0.7580 · v1 0.7302/0.9631 · v2 0.8377/0.9541 · **v3 0.8434/0.9525**),
  the thr=10 rationale, and how to stop early (repoint `output:`).

### D6. Verification (findings 4, 5)
1. **Exact parity (release gate, `dev/mit_liconn/verify_port.py`):** each ported stage vs the on-disk research
   output (`v1_lcc_confident.h5`, `v2_merge.h5`, `v3_weak.h5`) by chunked `np.array_equal`; if labels are not
   deterministic, relabel-invariant partition equality (bijection check). Pins affinity path + sha256, channel
   order, dtype, kwargs. Metrics are secondary confirmation.
2. **CI parity fixtures** (small synthetic, deterministic) — ported vs research function on identical input,
   covering: local-min/min-frag; `host_both=False`; anchor-slice carve skip; lateral **and** z-isolated
   completion; mutual/margin rejection; weak-gap projection; `recover=False`; **plus `v4_complete`
   (`prefer_length` both ways) and `completeness_report`** (previously missing).
3. **Helper parity:** `seg_stats` vs `compute_bbox_all_3d`+`bincount` (bbox, sizes, centroids; background
   label; sparse/high IDs); `apply_lut` vs `lut[seg]` (equality; `chunk` non-divisible boundaries; identity;
   dtype).
4. **Perf, with thresholds (finding 5):** (a) monkeypatch-count `cc3d.statistics` — a 3-stage batch calls it
   `1 + (#stages returning n>0)` times, **never once per stage**; (b) peak-allocation via
   `tracemalloc.get_traced_memory()` around `apply_lut` on a ≥256³ volume must stay **< 25% of the volume's
   nbytes** (a `lut[seg]` temp would be ~100%) — `np.shares_memory` is dropped as insufficient; (c) documented
   full-volume batch benchmark with **explicit acceptance**: wall < 150s and peak RSS < 28 GB (references:
   63s, 25.6 GB; old: 820s, 33.7 GB).
5. **Boundaries:** guardrail + API-snapshot tests; new assertion that `decoders/axon/` imports nothing from
   `dev`, `connectomics.training`, `connectomics.evaluation`. Do not touch the API snapshot unless `__all__`
   intentionally changes.
6. **Configs:** `python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'`.

## Files and Areas
| path | action |
|---|---|
| `connectomics/decoding/decoders/axon/{__init__,stats,split,merge,weak,complete,completeness}.py` | create |
| `connectomics/decoding/registry.py::_register_builtins` | +2 `register_decoder`, +2 `register_graph_op` |
| `connectomics/decoding/decoders/axon_tracklet.py` | keep; header note (supplies the v0 seed) |
| `tutorials/neuron_axon/{axon_decode.yaml,axon_decode_preferlen.yaml,waterz_baseline.yaml,README.md}` | create |
| `tutorials/axon_decoding/tracklet.yaml` | pointer note only |
| `tests/unit/test_axon_decoder_parity.py` | create |
| `dev/mit_liconn/verify_port.py` | create |

## Verification Plan
Gate order: (1) CI parity fixtures + helper parity + boundary tests; (2) `validate_tutorial_configs`;
(3) instrumented perf assertions with the thresholds in D6.4; (4) full-volume exact-equality parity as the
release gate, metrics secondary. Any inequality in (1) or (4) is a port bug — fix the port, never the tolerance.

## Risks and Questions
- **R1 (accepted, not open):** the seed keeps a `dev/mit_liconn` import via `axon_tracklet_base`; the task's
  port table does not cover it. Follow-up recorded in D1. This is a deliberate scope boundary, not an
  unresolved question.
- **R2 label determinism:** if waterz ordering is not reproducible, D6.1 uses partition equality; the coder
  records which applied.
- **R3** `axon_tracklet_base`'s own dev-path fallback could fail on a machine without `dev/mit_liconn`; the
  tutorial README states this prerequisite for the v0 node.
- No open questions.

## Changes Since Previous Plan Version
Resolves all 6 majors of `plan_v1_review.md` with verified facts instead of deferrals: **D1 decides v0**
(reuse `axon_tracklet_base`, no v0 in the new package, `axon_tracklet.py` kept, follow-up named, residual
dev-dependency stated) — previously an open Q1; **D2 reconciles the registration conflict** (both are function
calls; `register_decoder` for unary, `register_graph_op` for multi-input because `as_graph_op` enforces arity
1) — previously contradictory; **D3 fixes the stats/staleness contract** (concrete `n > 0` invalidation rule,
return `(seg, n)`, identity/dtype specified, and the **incorrect** `max(zr) <= seg.max()` assertion removed) —
previously undefined/wrong; **D5 pins the tutorial** (`evaluation.nerl_merge_threshold: 10` verified path,
exact commands, identical-input statement) and replaces the unsupported `enabled: false` v4 node with a
**separate `axon_decode_preferlen.yaml`** after verifying `GraphNodeConfig` has no `enabled` field —
previously deferred/non-executable; **D6.2 adds v4 + completeness parity** — previously missing; **D6.4
replaces `np.shares_memory` with a tracemalloc peak-allocation bound and gives the benchmark explicit
acceptance thresholds** — previously unbounded.
