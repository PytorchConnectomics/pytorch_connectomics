# Plan v0

## Summary
Vendor the validated axon decode into the package as **one new module family** under
`connectomics/decoding/decoders/axon/`, exposing four graph ops (`axon_split`, `axon_merge`, `axon_weak`,
`axon_complete`) plus a shared fast-stats helper, then express the pipeline as a `decoding.graph` YAML DAG and
ship `tutorials/neuron_axon/` that runs **naive waterz** and the **axon decoder** on the same affinity and
reports NERL base + oracle-merge (`merge_threshold=10`) for both.

Guiding principle for the refactor: the research modules are already validated, so this is a **port, not a
redesign** — identical algorithms and defaults, restructured for the package (no `dev/` imports, no
`sys.path` games, typed signatures, docstrings that state the measured effect). Any behavior change is a bug.

## Scope
- IN: new `connectomics/decoding/decoders/axon/` package; registration in `decoders/__init__.py`;
  `tutorials/neuron_axon/{axon_decode.yaml, waterz_baseline.yaml, README.md}`; retire/replace the superseded
  `decoders/axon_tracklet.py` (it wraps the OLD v0–v3 pipeline via a dev-path import).
- OUT: no change to `dev/mit_liconn/*` (reproduction path must keep working); no new metric code in
  `evaluation/` (the tutorial uses the existing NERL path); no GT edits; no model/training changes.

## Proposed Changes
1. **`decoders/axon/stats.py`** — port `bbox_fast.py`: `seg_stats(seg, want_centroids=False) -> (zr, sizes,
   cents)` via one `cc3d.statistics` pass, and `apply_lut(seg, lut, chunk=64)` chunked in-place relabel.
   Docstring records the measured 38.3s → 7.0s and the reason (`compute_bbox_all_3d`'s non-contiguous column
   scan). Every stage below uses these; none may call `compute_bbox_all_3d`.
2. **`decoders/axon/split.py`** — port `link_cut_change.link_cut_change` (IoU change-point; gates: local-min +
   greedy `min_frag` spacing; `recover` persistence OFF by default) and `confident_split.confident_parallel_split`
   (caliber + extrapolated-velocity collinearity + host area-step + `host_both=False` + anchor gathering +
   tubeness; **keep the anchor-slice carve fix**). Public entry `axon_split(aff_or_seg, ...) -> seg`.
3. **`decoders/axon/merge.py`** — port `v2_merge`: `_complete` (lateral **and** z-isolated fragment absorption)
   then IoU-primary mutual best-buddy with `margin`. Defaults `aff_lo=0.4, merge_iou=0.45, margin=0.15`.
4. **`decoders/axon/weak.py`** — port `v3_weak`: projected-mask IoU over gap candidates, `dim_tol`, caliber,
   mutual + `margin`. Defaults `max_gap=15, min_iou=0.35, margin=0.15`.
5. **`decoders/axon/complete.py`** — port `v4_complete` behind `prefer_length`; docstring must state it is
   **om-negative** (best 0.8447/0.9457 vs v3 0.8434/0.9525) and is opt-in for run-length-over-precision.
6. **`decoders/axon/completeness.py`** — GT-free `completeness_report(seg)`: fraction of decent axons
   (size ≥ min_size, span ≥ span_frac·Z) touching the border ≥2×. Ranker, not a quota (axons do terminate
   inside). Used by the tutorial as a GT-free sanity number.
7. **Registration** — in `decoders/__init__.py` via the `register_decoder(name, fn, *, overwrite=False)`
   *function call* (not a decorator), names: `axon_split`, `axon_merge`, `axon_weak`, `axon_complete`.
   Ops taking `(aff, seg)` register through `register_graph_op`; unary ones through `register_decoder`.
8. **`tutorials/neuron_axon/`** — `axon_decode.yaml` (DAG: v1 `axon_split`[raw] → v2 `axon_merge`[raw, v1] →
   v3 `axon_weak`[raw, v2], `output: v3`), `waterz_baseline.yaml` (`decode_waterz` on the same input),
   `README.md` with the comparison table and the `merge_threshold=10` rationale.
9. **Retire** `decoders/axon_tracklet.py`: delete it and its `tutorials/axon_decoding/tracklet.yaml`
   registration path, or keep the file as a thin deprecation shim raising a clear message pointing at the new
   ops. Prefer deletion (repo contract: "one canonical owner per concept, no back-compat shims").

## Files and Areas
| path | action |
|---|---|
| `connectomics/decoding/decoders/axon/{__init__,stats,split,merge,weak,complete,completeness}.py` | create |
| `connectomics/decoding/decoders/__init__.py` | register the 4 ops |
| `connectomics/decoding/decoders/axon_tracklet.py` | delete (superseded) |
| `tutorials/neuron_axon/{axon_decode.yaml,waterz_baseline.yaml,README.md}` | create |
| `tutorials/axon_decoding/tracklet.yaml` | delete or update to the new ops |
| `dev/mit_liconn/*` | unchanged |

## Verification Plan
1. **Numerical equivalence (the acceptance test).** On `datasets/mit-liconn/raw_x1_head-aff_r1.h5` +
   `gt_label_clean_v3.h5` (943 labels, strongmax yardstick, `merge_threshold=10`) the ported ops must
   reproduce, within ±0.001: v1 **0.7302/0.9631**, v2 **0.8377/0.9541**, v3 **0.8434/0.9525**, waterz
   **0.6530/0.7580**. Compare against the existing `dev/mit_liconn` outputs already on disk
   (`v1_lcc_confident.h5`, `v2_merge.h5`, `v3_weak.h5`) — ideally assert **label-identical** output.
2. **Perf non-regression:** `seg_stats` ≈7s (not ≈38s) on the 800×1024×1024 volume; no `compute_bbox_all_3d`
   import remains under `decoders/axon/`.
3. **Config/CI:** `python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'`
   (filter output to the new paths), and `pytest tests/unit/test_v3_guardrails.py
   tests/unit/test_public_api_snapshot.py -q` (boundary + API snapshot; update the snapshot if the public API
   intentionally grows).
4. **Registry smoke:** the 4 names appear in `list_decoders()`; the DAG in `axon_decode.yaml` resolves.

## Risks and Questions
- **R1 (highest) — silent behavior drift.** These stages are gate-sensitive (e.g. `host_both=False`, the
  anchor-slice carve skip, margin=0.15, `recover` OFF). Porting must copy the logic verbatim; the equivalence
  test in V1 is what catches drift. Do not "clean up" a gate while moving it.
- **R2 — the ops need affinity AND seg.** v2/v3 take `(aff, seg)`; ensure they register as graph ops (not
  unary decoders) so the DAG can fan `raw` + a prior node.
- **R3 — runtime.** A full-volume run is minutes per stage; CI cannot run it. Keep the equivalence test as a
  documented manual/opt-in script (e.g. `dev/mit_liconn/verify_port.py`), not a unit test.
- **R4 — public API snapshot** may need an intentional update; call it out rather than loosening the test.
- **Q1** Delete `axon_tracklet.py` outright (repo contract favors this) or leave a deprecation shim? Plan
  assumes delete; flag if the reviewer disagrees.
- **Q2** Should `axon_complete` be registered at all, given it is om-negative? Plan registers it (opt-in,
  labelled) so the flag is usable; alternative is to omit it from the registry.

## Changes Since Previous Plan Version
Initial plan.
