# Task

Refactor the validated research axon-decode pipeline (currently in `dev/mit_liconn/`) into the package as
first-class decoders, chain the stages with a YAML decode graph, and add a tutorial that compares **naive
waterz** against the **axon-specific decoding**.

## 1. Refactor into `connectomics/decoding/decoders/`
Vendor the ALGORITHM (do not just shell out to `dev/mit_liconn`). The existing
`decoders/axon_tracklet.py` wraps the OLD v0–v3 tracklet pipeline via a dev-path import with a fallback —
it is superseded and should be replaced/retired by this work.

Stages to vendor (validated; numbers below on the fair thr=10 yardstick, 943-label GT):

| stage | source module | what it does |
|---|---|---|
| v1 split | `link_cut_change.py` + `confident_split.py` | cut false merges: IoU change-point cut (local-min + min-frag gates), then close-ended tunnel-merge (relaxed `host_both=False`) |
| v2 merge | `v2_merge.py` | cross-section **completion** (lateral + z-isolated fragment absorption) then **IoU-primary** mutual best-buddy merge + **ambiguity margin** |
| v3 weak | `v3_weak.py` | bridge weak-affinity gaps by projected-mask IoU, mutual + margin |
| (opt) v4 | `v4_complete.py` | completion-driven radius link — **om-negative**, keep behind a `prefer_length` flag, clearly labelled |
| perf | `bbox_fast.py` | `seg_stats` (one `cc3d.statistics` pass: bbox+sizes+centroids, 5.5×) and `apply_lut` (chunked in-place LUT) |
| metric | `completeness.py` | GT-free completeness (decent axon touches border ≥2×) — a ranker, not a quota |

**Effective & efficient** requirements (already measured, must be preserved):
- use `seg_stats`, never the O(Z+Y+X) `compute_bbox_all_3d` column scan (28.6s → 7.0s)
- use `apply_lut` for relabels (avoids a full extra volume per LUT apply)
- support `stats=` / `inplace=` so a batch computes stats once (batch was 820s → 63s, 33.7 → 25.6 GB)

## 2. Chain the steps via YAML
Register each stage as a `decoding.graph` op and express the pipeline as a DAG, following the existing
pattern in `tutorials/axon_decoding/tracklet.yaml` (nodes with `name`/`op`/`inputs`/`kwargs`, then `output`).
Stopping earlier must be a one-line change (point `output:` at an earlier node).

## 3. Tutorial `tutorials/neuron_axon/`
A runnable example that decodes the same affinity twice and compares:
- **baseline**: naive waterz (`decode_waterz`)
- **axon**: the staged axon decoder above

Report NERL base + oracle-merge for both. Reference numbers (thr=10, 943-label GT):

| decode | base | om |
|---|---|---|
| naive waterz | 0.6530 | 0.7580 |
| v0_sm0 | 0.8284 | 0.9424 |
| v1 split | 0.7302 | 0.9631 |
| v2 merge (+margin) | 0.8377 | 0.9541 |
| **v3 weak (+margin)** | **0.8434** | **0.9525** |

The metric must use **`merge_threshold=10`** (a ≤10-node graze must not halve a long neuron's ERL; thr=1 was
~90% metric artifact).

## Constraints
- Follow repo conventions: register via the `register_decoder(name, fn, ...)` *function call* in
  `decoding/registry.py` (NOT a decorator); respect the dependency direction `decoding → {config, data, utils}`
  (no `connectomics.training` imports).
- Keep the research scripts in `dev/mit_liconn/` working (they are the reproduction path).
- Validate: `python scripts/validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'` and the
  boundary tests (`tests/unit/test_v3_guardrails.py`, `test_public_api_snapshot.py`).
- No behavior change: the refactored decoders must reproduce the numbers above.

## AMENDMENT (user, 2026-07-25) — remove the dev dependency ENTIRELY
1. **No `dev/mit_liconn` import may remain** in any package code path, including the v0 seed. Port the
   transitive closure of what is actually needed (not whole modules).
   - v0 seed entry points: `decode_axon.affxy`, `decode_axon.build_strong_sections(..., small=0)`,
     `decode_v2.decode_sections(..., no_force_split=True)`.
   - `decoders/axon_tracklet.py` (the dev-path wrapper) is then **deleted**, along with its registrations and
     `tutorials/axon_decoding/tracklet.yaml`.
2. **Put each function in its CANONICAL file, not an axon-only silo** (user's example:
   `connectomics/data/processing/bbox.py`). Generic utilities belong with their peers; only axon-specific
   decode stages belong under `decoders/axon/`.
