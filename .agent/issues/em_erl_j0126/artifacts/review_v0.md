# Review v0

## Summary

`code_v0` is functionally correct and registration-verified end-to-end on the real
skeleton file + live CloudVolume (mip0): LUT length == graph node count, out-of-bounds
== 37, skeleton-0 dominant segment == 1465128 (registration sentinel passes), 2.4%
assignment-zero, finite voxel-unit ERL 96,338 / gt 179,066 over 50 skeletons; 44 offline
tests pass. Raw notes + evidence: `state/review_v0.review.raw.md`.

The verdict is NEEDS_CHANGES only to satisfy the **new user requirement** raised during
this run — minimize GCS cost, make evaluation fast, and enable reuse/sharing so the
segmentation is not re-downloaded again and again. These are enhancements, not defects
in what `code_v0` already does.

Key measured facts driving the change:
- The layer is **unsharded `compressed_segmentation`**, chunk [128,128,64] — the chunk
  object is the atomic GCS download, so `code_v0`'s "each occupied chunk once,
  chunk-aligned, no empty chunks" is already the **cost floor at mip0**. Do not change
  the fetch strategy.
- One-time mip0 egress ≈ **3.6 GB** (82,823 chunks × ~43 KB compressed, measured).
- The reusable `node_segment_lut` is **4.0 MB** raw → saving/sharing it is ~900× cheaper
  than re-querying, and re-scoring becomes seconds instead of 11 min.
- mip1/mip2 labels agree with mip0 on 98.5% / 98.0% of a node sample → a coarser mip is
  a valid opt-in cost/speed lever (~4× less data per level) but perturbs ERL ~1.5%.

## Diff Baseline

run_start_ref: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949

Review surface: working tree of the nested `lib/em_erl` repo vs run_start_ref
(scripts/j0126_workflow.py, scripts/README.md, tests/test_j0126_workflow.py).

## Findings

- **[major] Persist the LUT and reuse it (skip GCS when present).** This is the primary
  answer to the user's "save the LUT for others / avoid re-downloading" goal. Add
  `--lut <path>`: if the file exists, load `node_segment_lut` and score with **zero GCS**;
  otherwise sample from CloudVolume and **write it there** as the compact, shareable
  artifact (this is exactly the old `seg_lut_all.h5`). Always rebuild the ERL graph from
  `-g` (deterministic, seconds) and guard `len(lut) == graph.num_nodes` to catch a
  skeleton/LUT mismatch. Quantified benefit: 3.6 GB one-time → 4 MB reusable, ~900×
  cheaper and near-instant on every subsequent evaluation.
- **[minor] Optional `--mip` (default 0).** Keep mip0 as the faithful default (matches the
  verified registration and the old pipeline). Offer a coarser mip as a documented
  opt-in: ~4× fewer chunks/less egress and faster per level, at a measured ~1.5% node
  label change (mip1==mip0 0.985). Document the tradeoff in `--mip` help and README.
- **[minor] Optional local chunk cache (`--cache-dir` → CloudVolume `cache=`).** Lets a
  re-run of the one-time *generation* (before the LUT is saved) avoid re-downloading.
  Secondary to the LUT; do not enable by default (it stores several GB of raw chunks).
- **[confirm, no change] Sampling is already cost-minimal.** Each occupied chunk fetched
  exactly once, chunk-aligned (no over-fetch), no empty chunks — the floor for this
  unsharded layout. Do not switch to per-point queries (same bytes, worse) or to larger
  super-chunk boxes (would download empty chunks → more cost).

## Tests to Add

- **LUT round-trip (offline):** sample with the fake CloudVolume, write the LUT, then run
  the score path with `--lut` pointing at it and assert (a) no CloudVolume access occurs
  (inject a fake that raises on `__getitem__`), (b) identical ERL to the direct path.
- **LUT/graph mismatch guard:** a LUT whose length != graph.num_nodes raises a clear error.
- **`--mip` plumbing:** the chosen mip is passed to `open_seg_cloudvolume` (unit-level,
  no network — assert the mip argument reaches the opener).
- Keep the existing sampler + `normalize_seg_url` tests; keep `python -m pytest tests/ -q`
  green.

## Questions

- LUT file format: a single-array HDF5 (like the old `seg_lut_all.h5`, keyed by node
  index) is the minimal, back-compatible choice and is assumed for `--lut`. Bundling the
  graph alongside is unnecessary since it is rebuilt deterministically from `-g`.
- Default mip stays 0 (faithful). The coarser-mip option is opt-in only unless the
  maintainer wants the cheaper mip1 as default given the 98.5% agreement.

## Verdict

VERDICT: NEEDS_CHANGES
