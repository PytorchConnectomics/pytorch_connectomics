# Plan v0 Review

## Summary

Reviewer: Codex (gpt-5.6-sol, ultra), read-only. Attested summary of
`state/plan_v0_review.review.raw.md`. Codex agrees the plan picks the right lever
(global big-branch endpoint linker) but returns `READY: no`: several acceptance
and safety contracts are unresolved. The central objection is that block-cropped
NERL cannot be the acceptance metric — the task requires whole-volume
length²-weighted NERL. A concurrent user correction reinforces this: the full
600-chunk `decode_v1` base already exists in `dev/zebrafinch/results/`
(600 `{chunk}_decode_v1.h5`, dataset `main`, verified), so **no regeneration is
needed** and the whole-volume path is cheap — removing the plan's main reason to
restrict to a block. Both inputs push the same revision: go whole-volume, consume
the existing base via a union-find remap/LUT and the existing streamed scorer.

## Findings

Preserved verbatim in intent from the raw transcript; severity tags are Codex's.

- [major] Acceptance metric violates the task: block-cropped NERL truncates long
  chains and hides out-of-block ownership. Use whole-volume test-50 NERL for
  acceptance; block is a smoke test at most.
- [major] Block-assembly identity incomplete: every foreground segment needs a
  unique `(chunk_key, local_label)` identity (bg 0 preserved) before building
  baseline and linked variants; the two variants must share identical voxel
  support / sampled nodes / missingness and differ only by the accepted remap.
- [major] GT crop underspecified: a valid experiment needs a complete half-open
  rectangular region, one outer crop/split of canonical `test_50_skeletons.h5`,
  one global→local translation, fixed `RES=[10,10,10]`, length-threshold 1000,
  merge-threshold 1. (Going whole-volume dissolves most of this.)
- [major] Oracle gate wrong: `linked_oracle >= baseline_oracle` is not "flat."
  Require equality within a stated tight tolerance; investigate any deviation
  (a pure join cannot raise the oracle). "Oracle-flat" = no newly-detected merge
  among sampled test-50 owners.
- [major] Algorithm is overlap-gated, not endpoint continuation:
  `--min-ov 50` + IoU-threshold + mutual-best-IoU excludes the low/zero-overlap
  continuations that are the whole point of going beyond face stitching. Specify
  geometric candidate generation (search radius, tangent polarity, feature
  formulas, defaults, mutual-best basis, deterministic tie-break, endpoint
  ambiguity).
- [major] Nucleus firewall not yet safe/executable: fix one marker source with
  its axis order / resolution / origin / chunk-offset mapping; each root holds a
  SET of marker IDs (union allowed iff |combined set| ≤ 1); pre-existing
  multi-nucleus segments and empty/misaligned mappings must fail or be
  quarantined; add the transitive `nucleus A → unmarked → nucleus B` test.
- [major] Substrate validation too weak and contradicted by "hero-final" reuse:
  those files are not proven-equivalent to decode_v1. (Now moot: the canonical
  600 decode_v1 exist in `results/`; use them directly, symlink-fix only.)
- [major] Block/resource not fixed: `--radius 1` may select 27 chunks; dense
  1008³ assembly is expensive. Predeclare an exact in-bounds region; prefer
  chunk-backed/streamed relabel + a whole-volume scorer that consumes the
  union-find LUT/remap.
- [minor] Define the certificate schema (endpoint/global IDs, face, raw cues,
  thresholds, mutual ranks, commit order, pre-union marker sets, decision,
  rejection reason); nucleus conflict is an unconditional dynamic rejection.
- [minor] Rename Step 2: the firewall is mandatory already in Step 1, so Step 2
  is soma-root/identity-prior integration only.

## Questions

- Extend the existing `oracle_stitch_decode_v1.py` to consume a linker remap/LUT
  for the whole-volume base+oracle score, rather than a new block evaluator?
  (Reviewer and user correction both favor the whole-volume path.)
- Confirm the marker source: `yl_cb_80nm.h5` (used by `soma_recon_wholevol.py`)
  vs `test_50_nuclei.txt` vs `yl_cb_80nm_neuron.h5`.

## Verdict

VERDICT: NEEDS_CHANGES
