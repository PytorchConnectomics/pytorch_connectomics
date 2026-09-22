# Code v0
## Overview

Implemented the approved Plan v2 as a GT-firewalled, research-only experiment under
`dev/zebrafinch/ec_mid_piece/`. The implementation covers resumable whole-volume inventory and
candidate extraction, deterministic one-hop policy resolution and freeze attestation,
post-freeze LUT evaluation and ceilings, residual/corridor analysis, visual-probe manifests, and
focused CPU tests.

The compute-budgeted work completed in this session is the full focused test suite, the
evaluator-side Stage 0 reproduction gate, and the all-axis/chunk-border synthetic smoke. The
whole-volume Stage 0 and Stage 1 passes were not run; their exact eight-shard non-interactive
commands and resume behavior are in the README.

## What Changed

- Added source-indexed BANIS affinity IO with restore-sigmoid, keep-mask application, no-halo
  chunk stitching, Form 1 native edge access, and Form 2 ABISS truth-table conversion.
- Added a 726-chunk resumable Stage 0 inventory with full-population sizes, boxes, moments, RAG
  degree, corrected diameter, estimated endpoints, elongation, bands, and quarantine state.
- Added sharded Stage 1 contact extraction with low-side ownership, exact chunk provenance,
  fixed 64-bin affinity histograms, topology summaries, morphology features, and a 200 M row stop.
- Added the complete predeclared policy grid with quarantine-before-ranking, strict tie abstention,
  relative margins, morphology gates, one-anchor assertions, deterministic NPZ bytes, write-once
  assignment payloads, freeze attestations, and first-match residual reasons.
- Added directed evaluator-only assignment application, component-level strict contamination
  accounting, Wilson intervals, stratification, risk/coverage rows, L123/L126 mechanism ceilings,
  oracle-clean filtering without re-ranking, and machine/Markdown reports.
- Added the L126 residual funnel and fixed gap-escalation gate using the all-`>=50` substantial
  baseline, nearest estimated endpoints, absolute tangent, and corridor median affinity.
- Added focused tests for real affinity indexing, canonical offsets, geometry, seam deduplication,
  deterministic/immutable freezing, GT firewall behavior, directed LUT application, component
  contamination, and the published L123/L126 ceiling scores.
- Recorded input drift: a frozen autonomous substantial linker now exists, but raw arm0_96 remains
  the selected honest baseline because Plan v2 locked that experiment identity. The later artifact
  is hashed in the manifest and is not consumed.

## Implementation Details

All proposal-side artifacts use deterministic, pickle-free NPZ archives. Stage 0 and Stage 1
cache one file per affinity-grid chunk, skip completed chunks on restart, cap concurrency at eight,
and print progress, wall time, and peak RSS. Stage 1 owns every face by its low-side source voxel;
halo reads provide the neighboring segmentation label without duplicating evidence. The merged
candidate table retains all owning chunks as CSR provenance and contains no baked selector winner.

Stage 2 removes quarantined anchors before ranking. Ranking is score descending, contact count
descending, then anchor ID ascending. Singletons use runner-up `0`; margins are relative; exact
score ties abstain even at margin `0.00`. Directed fragment-to-anchor maps cannot use an anchor as
a fragment, cannot target a quarantined/non-anchor label, and cannot induce a component with more
than one anchor. Existing frozen payloads are compared byte-for-byte and cannot be overwritten.

Stage 3 reads GT only after verifying assignment bytes and both freeze markers. It counts false
merges over the complete anchor component, not independently per edge. L123 applies 277 frozen
oracle pairs; L126 applies 616 unique union pairs and reproduces `0.7079613273917434`. Frozen
assignment targets are mapped through each oracle root without re-ranking. Stage 4 correctly uses
that L126 all-substantial rung as the denominator state for the published `+0.138303` mid increment.

Phase 3 multiscale execution and multi-hop absorption are intentionally absent, as approved.
`multiscale_contradiction` remains a reserved, never-emitted residual code.

## Files Changed

| File | Purpose |
|---|---|
| `dev/zebrafinch/ec_mid_piece/README.md` | Contracts, environment, exact resume commands, deferrals, and verification caveats |
| `dev/zebrafinch/ec_mid_piece/__init__.py` | Experiment package marker |
| `dev/zebrafinch/ec_mid_piece/common.py` | Paths, numeric constants, deterministic artifacts, hashes, timing, bands, and quarantine |
| `dev/zebrafinch/ec_mid_piece/affinity_io.py` | Verified source-indexed affinity conversion and coordinate-local HDF5 reader |
| `dev/zebrafinch/ec_mid_piece/stage0_inventory.py` | Sharded GT-free full-population inventory and geometry derivation |
| `dev/zebrafinch/ec_mid_piece/stage0_reproduce.py` | Evaluator-side canonical NERL and L126 inventory gate |
| `dev/zebrafinch/ec_mid_piece/stage1_candidate_edges.py` | Sharded contact evidence extraction, merge, enrichment, and topology summary |
| `dev/zebrafinch/ec_mid_piece/stage2_resolve.py` | Predeclared deterministic policies, invariants, residuals, and freeze attestations |
| `dev/zebrafinch/ec_mid_piece/stage3_evaluate.py` | Directed LUT evaluation, contamination accounting, ceilings, and reports |
| `dev/zebrafinch/ec_mid_piece/stage4_residual.py` | L126 residual gain buckets, corridor gate, funnel, and recommendation |
| `dev/zebrafinch/ec_mid_piece/visual_probes.py` | Evaluator-only representative crop manifest generation |
| `dev/zebrafinch/ec_mid_piece/smoke_test.py` | All-axis seam/dedup smoke plus optional real cross-file probe |
| `dev/zebrafinch/ec_mid_piece/gt_free/input_manifest.json` | Prepared exact input, coordinate, affinity, baseline, and drift metadata |
| `dev/zebrafinch/ec_mid_piece/evaluation_gt/stage0_reproduction.json` | Real evaluator-side reproduction result |
| `dev/zebrafinch/ec_mid_piece/tests/conftest.py` | Local experiment import setup |
| `dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py` | Real Form 2 truth table, negative controls, and canonical Form 1 offsets |
| `dev/zebrafinch/ec_mid_piece/tests/test_chunk_smoke.py` | Synthetic three-axis chunk-border smoke assertion |
| `dev/zebrafinch/ec_mid_piece/tests/test_firewall.py` | Static opened-path audit and freeze-tamper/marker rejection tests |
| `dev/zebrafinch/ec_mid_piece/tests/test_geometry.py` | Corrected caliber, endpoint fraction, and tangent sign tests |
| `dev/zebrafinch/ec_mid_piece/tests/test_lut_evaluation.py` | Baseline, directed map, component contamination, and ceiling-score tests |
| `dev/zebrafinch/ec_mid_piece/tests/test_resolver.py` | Ranking, tie, quarantine, invariant, determinism, and immutability tests |
| `.agent/features/ec_mid-piece/artifacts/code_v0.md` | CCC code-stage implementation and verification artifact |

## Git Baseline

run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd
current_head: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd

The current `git status --short` is byte-identical to
`.agent/features/ec_mid-piece/state/run_start.status`; all implementation files are ignored by the
repository. Nothing was staged or committed.

## Verification

Focused suite:

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
$ pytest -q dev/zebrafinch/ec_mid_piece/tests
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: zarr-3.1.6, cov-7.0.0, anyio-4.12.1
collected 23 items

dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py ..                 [  8%]
dev/zebrafinch/ec_mid_piece/tests/test_chunk_smoke.py .                  [ 13%]
dev/zebrafinch/ec_mid_piece/tests/test_firewall.py ........              [ 47%]
dev/zebrafinch/ec_mid_piece/tests/test_geometry.py ..                    [ 56%]
dev/zebrafinch/ec_mid_piece/tests/test_lut_evaluation.py ....            [ 73%]
dev/zebrafinch/ec_mid_piece/tests/test_resolver.py ......                [100%]

=============================== warnings summary ===============================
<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type SwigPyObject has no __module__ attribute

<frozen importlib._bootstrap>:241
  <frozen importlib._bootstrap>:241: DeprecationWarning: builtin type swigvarlink has no __module__ attribute

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================= 23 passed, 3 warnings in 11.27s ========================
```

Evaluator-side reproduction:

```text
$ MPLCONFIGDIR=/tmp/ec_mid_piece_mpl /usr/bin/time -v python dev/zebrafinch/ec_mid_piece/stage0_reproduce.py
{
  "baseline_nerl": 0.4443760423975247,
  "mass_percent_of_all_skeleton_nodes": {
    "dust": 4.106260419890385,
    "mid": 7.647875091096048,
    "substantial": 87.91921652407432
  },
  "merge_threshold": 50,
  "passed": true,
  "piece_counts": {
    "dust": 8979,
    "mid": 1887,
    "substantial": 668
  },
  "piece_definition": "per-(skeleton,label)",
  "scope": "evaluator-side reproduction gate only; emits published L126 aggregate constants and must never be read by GT-free selectors",
  "skeletons": 50
}
	Command being timed: "python dev/zebrafinch/ec_mid_piece/stage0_reproduce.py"
	User time (seconds): 7.69
	System time (seconds): 0.88
	Percent of CPU this job got: 69%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:12.27
	Maximum resident set size (kbytes): 925636
	Exit status: 0
```

Small-crop synthetic smoke, including all affinity axes and a 1008 chunk border:

```text
$ /usr/bin/time -v python dev/zebrafinch/ec_mid_piece/smoke_test.py --skip-real
{
  "synthetic": {
    "axis_contact_counts_zyx": [
      8,
      8,
      8
    ],
    "fragment_anchor_edge": [
      10,
      20
    ],
    "low_side_border_owner_present": true,
    "one_box_equals_tiled": true,
    "owner_chunk_ids": [
      0,
      1,
      11,
      12,
      121,
      122,
      132,
      133
    ]
  }
}
	Command being timed: "python dev/zebrafinch/ec_mid_piece/smoke_test.py --skip-real"
	User time (seconds): 0.97
	System time (seconds): 0.12
	Percent of CPU this job got: 98%
	Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.11
	Maximum resident set size (kbytes): 204640
	Exit status: 0
```

Focused static checks:

```text
$ find dev/zebrafinch/ec_mid_piece -maxdepth 2 -name '*.py' -print0 | xargs -0 -n1 -P4 black --no-cache --fast --target-version py311 --check
$ isort --check-only dev/zebrafinch/ec_mid_piece/*.py dev/zebrafinch/ec_mid_piece/tests/*.py
$ flake8 --max-line-length=100 dev/zebrafinch/ec_mid_piece/*.py dev/zebrafinch/ec_mid_piece/tests/*.py
$ python -m py_compile dev/zebrafinch/ec_mid_piece/*.py dev/zebrafinch/ec_mid_piece/tests/*.py
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
All done! ✨ 🍰 ✨
1 file would be left unchanged.
black/isort/flake8/py_compile: PASS
```

## Review Focus

- Confirm Stage 1's low-side ownership and CSR provenance prevent both tile-overlap and chunk-seam
  duplication while retaining all `(fragment_id, anchor_id)` evidence.
- Confirm Stage 2 removes quarantine before ranking, applies strict score-tie abstention, follows
  the declared residual precedence, and never rewrites frozen assignment bytes.
- Confirm Stage 3's directed map and component-level materiality accounting cannot hide an
  anchor-anchor or cross-neuron union behind an aggregate NERL gain.
- Confirm Stage 4 starts its `+0.138303` bucket analysis from the tested L126 all-substantial score
  (`0.7079613273917434`), not the earlier L123 277-pair score.
- Confirm the static firewall covers literal and dynamically constructed evaluator paths and that
  evaluator output is never imported by proposal-side code.

## Risks and Unknowns

Plan v2 status against all seven major v1 findings:

1. **Finding A — firewall ordering and immutable assignment attestation: resolved.** Reproduction
   is evaluator-side and isolated; proposal sources are statically audited; assignment bytes are
   written before and hashed by a sibling freeze JSON; resumed freezes must be byte-identical.
2. **Finding B — undefined selector behavior: resolved.** Contact-area floor, singleton runner-up,
   relative margins, strict ties, quarantine-before-ranking, stable sort direction, and floor-safe
   artifact keys are all explicit and tested.
3. **Finding C — dimensionally invalid/vacuous morphology: resolved.** Diameter, second-moment
   axis/endpoints, `[0,1]` endpoint fraction, and absolute tangent cosine implement the corrected
   formulas and pass rod/sign-invariance tests.
4. **Finding D — insufficient source-indexed affinity contract: resolved.** Form 1 is grounded in
   canonical offsets; Form 2 names source/destination slices and zero padding; a real crop gives
   `maxdiff == 0`, while four wrong conventions fail.
5. **Finding E — incomplete/leaky multi-hop: resolved by deferral.** No multi-hop selector exists;
   the README records both missing fragment-fragment evidence and the GT feedback leak.
6. **Finding F — unreachable residuals/incomplete escalation: resolved.** Every fragment is in the
   frozen contact summary, residual order is explicit, estimated endpoints exist, and the fixed
   distance/tangent/affinity/material-share escalation is implemented.
7. **Finding G — undefined evaluator ceiling composition: resolved.** Oracle joins precede
   unchanged frozen assignments with target-root mapping; oracle-clean filters without re-ranking;
   both directed composition and the canonical L123/L126 scores are tested.

Remaining risks and unknowns:

- Whole-volume Stage 0 (~2.4 h planned) and Stage 1 (~3.5 h planned) were intentionally not run.
  Candidate-table size, real wall time/RSS, candidate coverage, assignments, Stage 3 NERL, Stage 4
  funnel shares, results files, and visual probes therefore remain unmeasured/pending.
- The optional real cross-file seam read stalled on shared `/projects` storage in both an eight-file
  corner and a narrowed two-file seam and was stopped. The real single-file source-index truth
  table passed, and synthetic three-axis seam ownership/dedup passed; real cross-file seam IO is
  still unverified in this session.
- A frozen autonomous linker appeared before implementation although Plan v2 stated it was absent.
  The implementation did not change the approved experiment identity: raw arm0_96 remains the
  honest baseline, and the drift artifact/hash is recorded but not consumed.
- The predeclared band ladder may poorly capture GT-mid pieces; estimated second-moment endpoints
  remain weak for curved/branched fragments; and no independent calibration volume exists. These
  are experiment outcomes/limitations, not knobs to retune on the test skeletons.
- Edge-table row count is still unknown until Stage 1. The implementation aborts above the declared
  200 M shard-row upper bound without sampling.
- Values that Plan v2 did not fully specify and therefore had to be made explicit are: histogram
  quantiles use the selected bin's upper edge; reciprocal-best anchor-side ties use score,
  contact count, then ascending fragment ID; eligible rows in the `none` control use
  `ambiguous_multiple_anchors` as the final abstention bucket because no `policy_disabled` reason
  exists; corridor lines use deterministic six-connected rasterization with Z/Y/X tie order; and
  multi-chunk ownership provenance is stored as a CSR list of all owning chunk IDs.
- Concurrent repository activity created the initial implementation files during this coding turn.
  The final audited state is stable, HEAD matches the run baseline, and tracked status is identical
  to the recorded run-start status, but this concurrency is worth checking during review.

## Changes Since Previous Code Version

Initial implementation.
