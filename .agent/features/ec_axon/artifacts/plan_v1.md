# Plan v1

## Summary
Same goal as v0 — vendor the validated axon decode into `connectomics/decoding/decoders/axon/`, chain it as a
`decoding.graph` DAG, ship a waterz-vs-axon tutorial — but every execution contract the reviewer flagged is now
pinned: the **seed stage is explicit** (`v0 = tracklet_base(aff)`, the existing `axon_tracklet_base`), the
**graph adapters** and **exact registration call site** are named, `stats=`/`inplace=` semantics (including
staleness) are specified, acceptance is **exact array equality** rather than a metric tolerance, and the
tutorial actually **runs the evaluation**.

Discipline: **port, not redesign.** Internals and constants move verbatim; the only permitted mechanical
changes are listed in "Permitted mechanical changes" and each is boundary-tested.

## Scope
- IN: `connectomics/decoding/decoders/axon/` (7 modules); registration in `decoding/registry.py`
  `_register_builtins()`; `tutorials/neuron_axon/`; explicit disposition of `axon_tracklet.py`; parity fixtures
  + an opt-in full-volume verification script.
- OUT: no change to `dev/mit_liconn/*`; no algorithm changes; no GT edits; no new metric implementation (NERL
  is invoked through the existing runtime/evaluation stage, never imported into `decoding`).

## Proposed Changes

### A. Seed stage (fixes finding 1)
The reference pipeline is `v0_sm0 → v1 split → v2 merge → v3 weak`, where **`v0_sm0 = tracklet_base(aff)`**
(`dev/mit_liconn/build_v0_sm0.py`: 2D waterz sections with `small=0`, conservative linking, **no** crumb
cleanup). Therefore:
- `axon_split` **never** builds a seed. Its signature is `axon_split(seg, *, ...)` — seg in, seg out.
- The DAG's first node is the seed. Port `tracklet_base` as **`axon_v0`** (`inputs: [raw]`), preserving
  `small=0` and no-crumb-cleanup. If porting `tracklet_base` proves out of budget, the fallback is to keep the
  existing `axon_tracklet_base` op as the v0 node and vendor only v1–v3 — **stated explicitly in the code and
  README**, never left implicit.

### B. Graph adapters + exact registration (findings 2, 3)
- Internal functions keep source-aligned signatures: `axon_split(seg, ...)`, `axon_merge(seg, afz, ...)`,
  `axon_weak(seg, fgmax, ...)`, `axon_complete(seg, ...)`.
- Each gets a thin adapter `def _<name>_op(inputs, **kwargs)` that **validates arity** and unpacks in a
  documented order, e.g. `axon_merge` ← `inputs=[raw, seg]` → `(seg=inputs[1], afz=inputs[0][0])`.
  Channel extraction (`aff[0]` = z-affinity, `aff[:3].max(0)` = fgmax) happens in the adapter, not the core.
- **Call site is `connectomics/decoding/registry.py::_register_builtins()`** (guarded by `_BUILTINS_REGISTERED`,
  lazy, no import-time side effects), alongside the existing `register_decoder("decode_waterz", ...)` calls:
  `register_graph_op("axon_v0"|"axon_split"|"axon_merge"|"axon_weak"|"axon_complete", _<name>_op, overwrite=True)`.
  Multi-input ops use `register_graph_op` (NOT `register_decoder`, which unary-wraps).
- Tests: correct input order, wrong arity raises, cold-start lazy discovery, duplicate registration.

### C. `stats=` / `inplace=` contracts (finding 4)
- `seg_stats(seg, want_centroids=False) -> (zr, sizes, cents)`; `zr[L]` = inclusive 6-tuple.
- `apply_lut(seg, lut, chunk=64)` mutates **in place** and returns the same object (identity preserved);
  `lut` is cast to `seg.dtype`.
- Every stage takes `inplace: bool = False` (default **copies**, matching the source) and, where the source
  has it, `stats: tuple | None = None`.
- **Staleness rule (documented + asserted):** `stats` describes one specific labelling; any topology-changing
  mutation invalidates it. Batch drivers must recompute after a stage reports a change (`n > 0`). A debug
  assertion checks `max(zr) <= seg.max()` when stats are supplied.

### D. Port mapping (finding 5)
| source (`dev/mit_liconn/`) | destination | notes |
|---|---|---|
| `bbox_fast.py` | `axon/stats.py` | verbatim |
| `link_cut_change.py` | `axon/split.py::link_cut_change` | verbatim incl. `recover` default OFF |
| `confident_split.py` | `axon/split.py::confident_parallel_split` | verbatim incl. `host_both=False`, anchor-slice carve skip |
| `v2_merge.py` | `axon/merge.py` | verbatim incl. `_complete` (lateral + z-isolated), `margin` |
| `v3_weak.py` | `axon/weak.py` | verbatim incl. `dim_tol`, projected-mask IoU |
| `v4_complete.py` | `axon/complete.py` | verbatim; docstring states **om-negative** |
| `completeness.py` | `axon/completeness.py` | verbatim |

**Permitted mechanical changes (exhaustive):** import rewrites (no `sys.path`, no `dev.` imports); module/
package placement; the graph adapters in B; type annotations; docstrings; removing `__main__` blocks. **Not
permitted:** renaming/retuning any constant or gate, reordering logic, "simplifying" a condition.

### E. Tutorial that actually evaluates (findings 6, 11)
`tutorials/neuron_axon/`:
- `axon_decode.yaml` — nodes `v0(axon_v0)[raw] → v1(axon_split)[v0] → v2(axon_merge)[raw,v1] →
  v3(axon_weak)[raw,v2]`, plus an **opt-in** `v4(axon_complete)[v3] {prefer_length: true, enabled: false}`;
  `output: v3`. Stopping earlier = repoint `output:`.
- `waterz_baseline.yaml` — `decode_waterz` on the **same** `decoding.load_prediction_path`, asserted identical
  in the README (same path + checksum) so the comparison is like-for-like.
- Both set the evaluation GT and the NERL config, including **`merge_threshold: 10`** at its real schema path
  (to be confirmed in `config/schema/evaluation.py`; if absent, the README documents the exact
  `scripts/main.py --mode test` invocation that produces base + oracle-merge).
- `README.md` — the comparison table, the thr=10 rationale, and copy-pasteable commands.

### F. Verification (findings 7, 8, 9, 10)
1. **Exact parity (mandatory, opt-in script `dev/mit_liconn/verify_port.py`):** run each ported stage on the
   real volume and compare to the on-disk research outputs (`decode_tracklet_v0_sm0.h5`, `v1_lcc_confident.h5`,
   `v2_merge.h5`, `v3_weak.h5`) with **exact chunked `np.array_equal`**; if label IDs are not deterministic,
   fall back to relabel-invariant partition equality (bijection of the label pairing). Pin affinity path +
   checksum, channel order, dtype, and all kwargs. Metrics are reported as a **secondary** check
   (v1 0.7302/0.9631, v2 0.8377/0.9541, v3 0.8434/0.9525, waterz 0.6530/0.7580 @thr=10).
2. **CI parity fixtures** (small synthetic volumes, deterministic, fast) covering each sensitive gate:
   local-min/min-frag cut; `host_both=False` tunnel; anchor-slice carve skip; lateral **and** z-isolated
   completion; mutual/margin rejection (ambiguous pair must NOT merge); weak-gap projection; `recover=False`.
   Each fixture asserts the ported function equals the research function on the same input.
3. **Helper parity:** `seg_stats` vs `compute_bbox_all_3d`+`bincount` (bbox/sizes equality; background label;
   sparse/high IDs; centroids); `apply_lut` vs `lut[seg]` (equality, `chunk` boundaries incl. non-divisible,
   in-place identity, dtype).
4. **Perf (instrumented, not timed-only):** monkeypatch-count `cc3d.statistics` calls — a batch of N stages
   must call it once + once per *changed* stage, not N times; assert `apply_lut` allocates no full-volume temp
   (peak-RSS delta or `np.shares_memory` identity); documented full-volume batch benchmark recording wall time
   and peak memory (reference: 820s→63s, 33.7→25.6 GB).
5. **Boundaries:** `pytest tests/unit/test_v3_guardrails.py tests/unit/test_public_api_snapshot.py -q`; a new
   assertion that `decoders/axon/` imports nothing from `dev`, `connectomics.training`, or
   `connectomics.evaluation`. **Do not** touch the API snapshot unless `__all__` intentionally changes.
6. **Configs:** `python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'` (filter to
   the new paths).

### G. Disposition of `axon_tracklet.py` (finding 12)
Reference audit found exactly three sites: `connectomics/decoding/registry.py` (imports 4 ops),
`tutorials/axon_decoding/tracklet.yaml`, and the module itself. **Decision: keep `axon_tracklet.py` for now** —
it supplies `axon_tracklet_base`, the v0 seed (§A). Action: (i) leave the module and its registrations intact;
(ii) add a header note that v1–v3 are superseded by `decoders/axon/`; (iii) update
`tutorials/axon_decoding/tracklet.yaml` with a pointer to `tutorials/neuron_axon/`. Deletion is deferred until
`axon_v0` is ported and verified — at which point the old ops are removed in one explicit follow-up.

## Files and Areas
| path | action |
|---|---|
| `connectomics/decoding/decoders/axon/{__init__,stats,split,merge,weak,complete,completeness}.py` | create |
| `connectomics/decoding/registry.py::_register_builtins` | add 4–5 `register_graph_op` calls |
| `connectomics/decoding/decoders/axon_tracklet.py` | keep; header note (supplies the v0 seed) |
| `tutorials/neuron_axon/{axon_decode.yaml,waterz_baseline.yaml,README.md}` | create |
| `tutorials/axon_decoding/tracklet.yaml` | pointer note only |
| `tests/unit/test_axon_decoder_parity.py` | create (fixtures §F2–F3, boundary §F5) |
| `dev/mit_liconn/verify_port.py` | create (opt-in full-volume exact parity) |

## Verification Plan
Gate order: (1) CI parity fixtures + helper parity + boundary tests must pass; (2) config validation;
(3) instrumented perf assertions; (4) opt-in full-volume exact-equality parity as the release gate, with the
metric table as a secondary confirmation. Any inequality in (1) or (4) is a port bug — fix the port, never the
tolerance.

## Risks and Questions
- **R1 `tracklet_base` port size.** It pulls in the 2D-waterz section builder; if too large for this change,
  use the documented fallback (§A) and keep `axon_tracklet_base` as the seed node. Decide before coding.
- **R2 label determinism.** waterz/watershed ordering may not be bit-reproducible across runs; §F1 falls back
  to relabel-invariant partition equality — the coder must determine which applies and record it.
- **R3 evaluation schema path** for `merge_threshold` may not exist in `config/schema/evaluation.py`; then the
  README documents the exact command instead of a config key (§E).
- **Q1** Is porting `tracklet_base` (§A) in scope, or is the fallback acceptable for this change?

## Changes Since Previous Plan Version
Addresses all 13 major + 2 minor findings of `plan_v0_review.md`: added the explicit **seed stage** (§A, was
undefined — the DAG previously fed `raw` straight into `axon_split`); specified **graph adapters and the exact
registration call site** `registry.py::_register_builtins` with `register_graph_op` for multi-input ops (§B,
was wrongly `decoders/__init__.py`); pinned **`stats=`/`inplace=` semantics + staleness** (§C, was absent);
added the **source→destination port table with an exhaustive list of permitted mechanical changes** and dropped
the polymorphic `aff_or_seg` API (§D, enforces "port not redesign"); made the tutorial **actually evaluate**
with GT/paths/both NERL modes/`merge_threshold=10` and an identical-input assertion, plus the **opt-in v4
node** (§E, findings 6+11); replaced the ±0.001 tolerance with **exact/partition equality**, added **CI parity
fixtures for every sensitive gate**, **helper parity** for `seg_stats`/`apply_lut`, and **instrumented perf**
checks (§F, findings 7–10); resolved `axon_tracklet.py` by **reference audit → keep as the v0 seed** rather
than an unresolved "delete or update" (§G, finding 12); added the **dependency-boundary assertion** and the
"don't touch the API snapshot" rule (§F5, minors 13–14).
