# Task

Implement and run a bounded Zebrafinch experiment that tests whether **chunk-local
CC3D at affinity threshold 0.70** can improve the canonical arm0_96 whole-volume
ABISS segmentation without sacrificing ABISS's global identity and coverage.

The fusion is intentionally asymmetric:

```text
arm0_96 CC3D@0.70 chunks  -> local high-confidence pieces and boundary evidence
arm0_96 ABISS             -> global labels, identity anchors, and coverage backbone
fusion                    -> conservative one-anchor assignments and diagnostics
```

Do not implement voxel voting, choose a decoder per neuron, replace ABISS with an
unstitched CC3D volume, or transitively union labels merely because the two partitions
overlap. Those controls have already failed on the older pretrain substrate.

This is one CCC research task under `dev/zebrafinch/`. It must first establish whether
the current arm0_96 pair contains actionable complementarity, then implement the smallest
safe fusion policy justified by that measurement. A rigorous negative result is a valid
completion.

## Motivation and current evidence

The canonical whole-volume substrate is:

- affinity checkpoint/run:
  `outputs/nisb_base_banis_plus_zebrafinch_heavy/20260726_114349/test_step=00200000/0`;
- whole-volume segmentation:
  `dev/zebrafinch/wholevol_arm096_fullmask/seg_arm096_fullmask/precomputed/seg/seg_arm096_fullmask`;
- decoder: ABISS with the corrected full-1008 FFN tissue+border keep-mask and
  `AGG_THRESHOLD=0.3`;
- canonical test-50 funlib NERL, merge threshold 50: **0.444376**;
- union-only oracle: **0.941244**;
- ERL-visible cross-neuron merged labels: **3**.

The error analysis in `dev/zebrafinch/error_analysis_arm0_96.md` shows that this substrate
is primarily false-split limited: joining each neuron's three largest oracle pieces would
reach about **0.539887**, but a wrong real-real union costs roughly eight correct joins.
Precision is therefore the binding constraint.

Chunk-local CC3D@0.70 is strong on the current arm0 affinity family:

| affinity tag | human-GT tier10 CC3D@0.70 | human-GT local ABISS |
|---|---:|---:|
| `arm0_win144` | 0.775832 | 0.787576 |
| `arm0_native` | **0.8544** | 0.785181 |

The comparison must not proceed until the naming is resolved. The whole-volume artifact
called `arm0_96` uses the chunk store that `dev/zebrafinch/build_tier10_jobs.py` calls
`arm0_win144` (the filename without `_win48x96x96`). The tag `arm0_native` refers to the
sibling `_win48x96x96.h5.chunks` store. Record the exact source and prove the mapping from
manifests and file metadata; do not select the more favorable local row by name.

The local result is not a whole-volume claim. `dev/zebrafinch/lesson_abiss.md` L125 measured
an inversion between ten-chunk CC3D affinity ranking and whole-volume ABISS ranking because
local windows cannot see long-range false merges. All fusion acceptance decisions in this
task must therefore use the complete test-50 skeletons and global label identities.

Earlier pretrain ABISS/CC3D fusion controls in
`dev/zebrafinch/cc3d/results_abiss_cc3d_fusion_0729.md` were negative:

| control | NERL |
|---|---:|
| ABISS masked by CC3D coverage | 0.356806 |
| partition meet | 0.210336 |
| sampled transitive partition join | 0.101183 |
| split ABISS by large CC3D overlaps | 0.235815 |
| CC3D plus disjoint ABISS fill | 0.241685 |

The partition meet nevertheless had a high GT split-repair ceiling, showing that the two
partitions can supply useful atoms while still lacking a deployable identity assignment.
This task tests that narrower opportunity on arm0_96; it must not repeat the failed label
algebra and call it a new fusion method.

## Primary questions

Answer these in order:

1. Does CC3D@0.70 from the exact arm0_96 affinity reproduce its recorded local behavior?
2. On whole test-50 skeletons, how do raw chunk-local CC3D atoms complement the ABISS
   partition: better boundaries, continuation evidence, coverage, or only GT-dependent
   per-neuron routing?
3. How often does one CC3D component overlap exactly one established ABISS anchor plus one
   or more smaller ABISS fragments, versus zero or multiple anchors?
4. Can the one-anchor cases be converted into assignment-only ABISS label remaps with a
   positive whole-volume NERL delta and no measured high-impact merge?
5. Do CC3D disagreements identify any of the three ERL-visible ABISS false merges, and if
   so are they usable as local split proposals without broadly fragmenting healthy ABISS
   objects?
6. Which residual cases require global morphology, endpoint linking, nuclei/glia
   quarantine, or a later gap-completion experiment?

## Scientific hypotheses

### H1 — CC3D supplies useful local atoms

At 0.70, CC3D retains high-confidence connected foreground that can expose local
continuations or alternative boundaries that ABISS's watershed/agglomeration represented
as separate or fused labels.

### H2 — ABISS must retain global ownership

CC3D labels are independently namespaced by chunk and cannot represent whole-neuron
identity. ABISS remains the initial global label system. A CC3D component may support
assignment of a small ABISS fragment to one established ABISS anchor, but it may not union
multiple established anchors.

### H3 — complementarity is directional

The plausible positive operation is bounded fragment-to-anchor assignment. Broad ABISS
splitting from CC3D disagreement and broad CC3D stitching from ABISS overlap are expected
to be unsafe. Measure their oracle potential, but promote them only if the data overturns
that expectation under explicit safety gates.

## Ground-truth firewall

Separate the experiment into immutable GT-free proposal artifacts and evaluator-only
diagnostics.

### GT-free stages may use

- the arm0_96 affinity chunks, their index/manifest, and the canonical tissue mask;
- the arm0_96 ABISS segmentation and GT-free segment statistics;
- CC3D@0.70 components, overlap counts, face evidence, affinities, morphology, nuclei,
  glia/tissue masks, and deterministic global graph state;
- thresholds fixed in this task or calibrated on an independent calibration set.

They may not read test skeletons, skeleton-node LUTs, GT owners, per-skeleton NERL, or any
artifact derived from those inputs. Write GT-free products under `gt_free/`, and record
`gt_free: true` and `frozen_before_evaluation: true` in each policy manifest.

### Evaluator-only stages may use

- `/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`;
- the canonical arm0_96 node LUTs under `dev/zebrafinch/reports/arm096_lut/`;
- post-hoc ownership, per-skeleton deltas, oracle controls, and NERL.

Write these products under `evaluation_gt/`. The evaluator must consume a frozen assignment
or diagnostic manifest; it may not modify candidates, thresholds, anchors, margins, or
conflict resolution.

CC3D threshold 0.70 was selected using existing tier evaluations, so this run is an
**exploratory fixed-policy evaluation**, not an unbiased holdout claim. Do not sweep CC3D
thresholds on test-50 and report the best one. A later generalization claim requires an
independent calibration/confirmation volume.

## Phase 0 — input identity and fidelity gates

Create `gt_free/input_manifest.json` before decoding anything. It must include:

- exact affinity directory, index JSON, checkpoint/run identity, file count, channel count,
  dtype, per-chunk shapes, and representative hashes or stable file metadata;
- an explicit declaration that the selected source is `arm0_win144` or `arm0_native`, with
  the evidence used to resolve the historical naming ambiguity;
- ABISS segmentation URI, shape, dtype, coordinate origin, voxel size, mask provenance,
  parameter JSON/log, and expected base NERL;
- CC3D threshold `0.70`, connectivity `26`, foreground rule, mask path, and output namespace
  convention;
- skeleton/evaluator inputs recorded only in the separate evaluator manifest.

The canonical CC3D foreground recipe is:

```python
fg = (aff[:3] > 0.70).any(axis=0)
fg &= tissue_mask
labels = cc3d.connected_components(fg, connectivity=26)
```

Reuse the established coordinate and mask logic in
`dev/zebrafinch/cc3d_chunks_tissue.py`, but do not blindly inherit its defaults: pass and
record the arm0_96 chunk directory and matching index explicitly. Verify border chunks,
especially clipped Y/X/Z extents, rather than assuming every chunk is `1008^3`.

Required fidelity checks:

1. reproduce arm0_96 ABISS NERL **0.444376** to the existing script's tolerance;
2. reproduce the recorded tier10 CC3D@0.70 score for the exact selected affinity tag and
   substrate before extrapolating to 726 chunks;
3. verify affinity values/channel order against direct HDF5 reads on several non-symmetric
   probes;
4. verify ABISS and CC3D chunk coordinate alignment with known voxels and displaced controls;
5. verify the tissue mask and clipped border shapes match the ABISS run's keep-mask contract;
6. prove that local CC3D label IDs are namespaced by chunk and never collide globally.

If the whole-volume arm0_96 input maps to `arm0_win144`, the human-GT local control is
0.775832, not the more favorable `arm0_native` 0.8544 row. A mismatch is a blocker to the
experiment, not permission to rename an artifact.

## Phase 1 — cheap whole-skeleton complementarity audit

Before building a dense fusion graph, stream CC3D@0.70 over all 726 chunks and sample its
labels at complete test-50 skeleton nodes. The CC3D node identity must be
`(chunk_id, local_cc3d_label)`; never treat equal local integers from two chunks as equal.

This audit may run CC3D in memory, sample/store only required values, and discard the dense
label array. Do not materialize hundreds of gigabytes of compressed CC3D chunks merely to
produce an evaluator LUT.

Adapt or reuse the pure partition helpers in
`dev/zebrafinch/cc3d/compare_cc3d_abiss.py`. Keep its pretrain defaults and prior
reproduction path working. Score at least:

| row | definition | status |
|---|---|---|
| `abiss_arm096` | canonical ABISS LUT | honest baseline |
| `cc3d_t070_unstitched` | unique chunk-local CC3D labels | diagnostic |
| `partition_meet` | factorized `(ABISS, CC3D)` labels | diagnostic atomization |
| `partition_join` | transitive sampled overlap join | expected-negative control |
| `abiss_masked_by_cc3d` | ABISS retained only on CC3D foreground | expected-negative control |
| `gt_best_decoder_per_skeleton` | per-skeleton max of ABISS/CC3D | GT upper bound |
| `gt_meet_merge_oracle` | GT-optimal reassembly of meet atoms | GT upper bound |

Also compute **directional candidate oracles** restricted to operations a deployable policy
could express:

- selectively assign an ABISS fragment to an ABISS anchor only when a shared CC3D@0.70
  component generated that candidate;
- selectively join CC3D face pieces only when ABISS/global overlap generated that candidate;
- selectively cut an ABISS label only along CC3D boundaries inside an independently flagged
  suspect object.

These are evaluator-only ceilings. They must preserve the candidate population generated
without GT and report candidate precision/base rate, obtainable headroom, joins/cuts needed,
and the cost of taking all candidates.

### Gate A — is fusion worth building?

Proceed to full-population candidate extraction only if at least one directional candidate
family contains material positive oracle headroom and a plausible conservative subset.
As a working materiality threshold, require either:

- at least `+0.01` canonical NERL oracle gain from candidate-restricted assignments; or
- recovery of an important known false merge/long-backbone break whose biological value is
  under-represented by test-50 NERL.

If the only positive row is GT per-skeleton routing or unrestricted meet reassembly, stop and
report that local atoms are complementary but deployable identity evidence is absent.

## Phase 2 — streaming full-population evidence graph

If Gate A passes, build the GT-free evidence graph by streaming one chunk at a time:

1. read the exact arm0_96 r1 affinity chunk and matching tissue-mask crop;
2. compute CC3D@0.70 in memory;
3. read the spatially matching arm0_96 ABISS labels;
4. accumulate compact component statistics and ABISS/CC3D overlap contingencies;
5. store compact face/border evidence needed for neighboring chunks;
6. discard dense arrays before advancing.

Do not save dense CC3D chunks by default. Sharded compact NPZ artifacts are preferred over
one enormous CSV. The stage must be resumable and deterministic, with one `.done`/manifest
record per completed shard and an atomic finalize step that refuses missing or duplicate
chunks.

For every local CC3D component, record at least:

- globally namespaced `(chunk_id, cc3d_id)`;
- voxel count, bounding box, centroid, face touches, and relevant shape descriptors;
- all overlapping nonzero ABISS labels and voxel counts/fractions;
- ABISS label global size/extent/anchor-quality fields from a separately built global table;
- connectivity bottleneck or threshold-persistence evidence when cheaply available;
- whether evidence touches a clipped volume boundary, tissue-mask boundary, or chunk face;
- local affinity summary at each proposed ABISS-label transition;
- neighboring-chunk face candidates with overlap/contact provenance.

Deduplicate face evidence deterministically. A synthetic object crossing a chunk boundary
must yield the same candidate graph regardless of shard count or processing order.

The graph must classify each CC3D component as:

- `zero_anchor`: no eligible ABISS anchor;
- `one_anchor`: exactly one eligible ABISS anchor plus optional fragments;
- `multi_anchor`: two or more eligible ABISS anchors;
- `anchor_only`: no new fragment to assign;
- `invalid_or_boundary`: insufficient or unreliable evidence.

Anchor and fragment definitions must use GT-free global ABISS observables. Predeclare a
small set of interpretable size/extent bands rather than selecting a test-optimal threshold.
At minimum distinguish established multi-chunk/large anchors, plausible mid-sized fragments,
and dust. Report sensitivity across the predeclared bands without choosing the best test row
as validated.

## Phase 3 — primary fusion policy: one-anchor assignment

The primary deployable operation starts from the ABISS segmentation and changes only global
ABISS label ownership:

- a smaller ABISS fragment may be assigned to one established ABISS anchor when one
  CC3D@0.70 component supplies direct local continuation evidence;
- an assignment changes the fragment's global LUT target; it does not copy a local CC3D label
  into the global namespace;
- a CC3D component containing multiple established ABISS anchors is always an abstention in
  the primary policy;
- the policy may never union two established anchors, even transitively through fragments;
- one-hop runs first; an assigned fragment cannot become a new anchor or relay;
- deterministic best/runner-up scoring and an explicit abstain state are required;
- candidates touching uncertain mask/chunk boundaries require separate reporting and may not
  silently share the interior operating point.

Score or gate one-anchor assignments using only predeclared GT-free evidence, such as:

- fraction and spatial extent of the fragment supported by the shared CC3D component;
- native r1 affinity mean/p90/low-tail across the ABISS transition;
- CC3D path bottleneck or persistence at a stricter threshold such as 0.75;
- fragment-to-anchor size ratio;
- endpoint-versus-flank geometry, tangent, caliber, and compartment compatibility;
- best-versus-runner-up margin where a fragment has several candidate components/hosts;
- nuclei, glia, multi-root, or low-quality-anchor vetoes.

CC3D membership is proposal evidence, not sufficient positive identity evidence by itself.
Evaluate a small fixed ablation sequence:

1. no fusion;
2. take every one-anchor CC3D candidate — unsafe diagnostic control;
3. CC3D support/overlap gate;
4. CC3D support plus affinity floor;
5. add winner margin and size-ratio guard;
6. add morphology/nuclei/glia vetoes when available.

Cache features so all ablations are cheap graph/LUT rescoring. Do not reread volume data for
each policy.

### Global resolution invariant

Resolve candidates over the complete ABISS-label graph, not independently per chunk. Before
accepting an assignment, union-find or equivalent state must prove that the resulting
component contains at most one established anchor. Conflicting claims, cycles that connect
anchors, or inconsistent chunk-local hosts must abstain. Process-order independence is a
required test.

A bounded multi-hop ablation is allowed only if one-hop is positive and precision-safe.
Multi-hop must retain a single original anchor, cap depth, store the relay path, and resolve
the complete population before skeleton evaluation. It may not turn a zero-anchor CC3D
group into an anchor-anchor bridge.

## Phase 4 — split and cross-chunk diagnostics

These branches are secondary and must not expand the primary implementation without passing
their directional oracle gates.

### ABISS split proposals from CC3D

One ABISS label containing multiple CC3D components is normal because high-threshold CC3D
has coverage holes. Decoder disagreement alone must never split it.

Only inspect ABISS objects independently flagged by GT-free global evidence, such as:

- multiple incompatible nucleus/root claims;
- glia/lamellar morphology mixed with a neurite trajectory;
- incompatible long centerlines, caliber transitions, or branch geometry;
- another existing arm0_96 error-correction detector.

For such suspects, report whether CC3D@0.70 supplies a boundary separating the conflicting
anchors. This task may emit split proposals and evaluator-only ceilings, but it must not
perform a dense voxel split unless Gate A shows material value and the CCC plan demonstrates
a bounded, testable realization. The arm0_96 audit has only three ERL-visible merged labels,
so broad split machinery is not the default use of this task.

### Cross-chunk CC3D continuation

CC3D face matches may corroborate a candidate continuation, but local IDs from adjacent
chunks remain different. Reuse the complete-face graph concepts in
`dev/zebrafinch/cc3d/build_face_iou_graph.py` rather than immediate face unions.

Require reciprocal/best-with-margin face evidence, deduplicated overlaps, and global
one-anchor resolution. ABISS identity agreement may support an already safe assignment;
ABISS overlap must not join an unrestricted CC3D transitive component. Report the
unstitched control and any protected forest as diagnostics, not as the default fused output.

## Evaluation contract

Use complete test-50 skeletons, canonical unscaled voxel coordinates, funlib aggregation,
and merge threshold 50. Reuse the validated arm0_96 evaluator rather than reimplementing
NERL.

Assignment-only policies should be evaluated through frozen global ABISS label remaps and
the cached node LUT where exact. Candidate construction and graph resolution must still run
over the full dense label population; sampled skeleton nodes cannot define the graph or omit
an unsampled relay/competing anchor.

For every baseline, control, oracle, and honest policy report:

- canonical whole-volume NERL and delta from arm0_96;
- accepted assignments, fragments, anchors, and final components;
- candidate topology counts: zero/one/multiple anchors and invalid/boundary cases;
- post-hoc correct-host precision and candidate recall on test-50 pieces;
- improved/unchanged/regressed skeleton counts and maximum single-skeleton regression;
- gained and lost funlib-weighted ERL;
- ERL-visible multi-owner components and zero-tolerance cross-neuron interactions;
- nuclei/glia/multi-anchor conflicts, including those invisible to test-50 NERL;
- coverage changes and background-node counts;
- result by fragment size/extent band, confidence, chunk-interior/boundary status, and hop;
- wall time, peak memory, bytes read, and compact-cache size.

Keep these tables separate:

1. **GT-free frozen policies**;
2. **expected-negative label-algebra controls**;
3. **GT-only ceilings and test-selected diagnostics**.

Only table 1 may be compared with FFN **0.538003**. Do not call a local NERL, oracle route,
or test-selected operating point an FFN-beating result.

If a compatible frozen arm0_96 error-correction/anchor-absorption map already exists, score:

- CC3D fusion directly on raw arm0_96;
- the prior correction alone;
- deterministic composition in both semantically valid orders;
- conflicts where both maps claim the same label.

Do not double-count the same fragment assignment under two task names. Prefer one canonical
resolver if this experiment and `.agent/features/ec_mid-piece/task.md` converge on the same
operation.

## Decision gates

### Gate B — promote a fused policy

A policy is promising only if:

- its assignments were frozen before GT evaluation;
- it improves canonical NERL beyond deterministic numerical noise;
- it introduces no measured high-impact real-real merge;
- no established-anchor component acquires a second anchor;
- gains occur across multiple skeletons rather than one outlier;
- the maximum regression and all multi-owner/nucleus conflicts are explicitly reported;
- the result remains positive under at least one nearby predeclared confidence/margin setting,
  rather than existing at a single test-selected point.

An aspirational milestone is to exceed FFN 0.538003, but failure to do so does not justify
relaxing the one-anchor or contamination firewalls.

### Gate C — promote CC3D splitting

Implement a real split only if the candidate-restricted split oracle shows material value,
an autonomous suspect detector has useful precision, and the proposed CC3D boundary preserves
the established trajectories on both sides. Otherwise leave split proposals as a report for
the nuclei/glia/error-correction task.

### Negative conclusion

Stop and record a negative result if any of the following holds:

- local CC3D strength does not survive the whole-skeleton complementarity audit;
- candidate oracle gain comes only from GT decoder routing or unrestricted reassembly;
- useful CC3D components usually contain multiple ABISS anchors;
- conservative one-anchor policies are flat, while permissive settings create regressions;
- apparent improvement depends on the wrong arm0 affinity tag, mask, or coordinate mapping.

## Deliverables

Place research code and artifacts under:

```text
dev/zebrafinch/abiss_cc3d_fusion/
```

Use small stage-specific scripts or modules; do not create one monolithic script. Expected
responsibilities are input audit, CC3D streaming/evidence extraction, candidate resolution,
LUT evaluation, and reporting. The CCC plan may choose exact filenames after inspecting
existing reusable helpers.

Required artifacts:

- `README.md` with exact commands, environment, stage status, and conclusions;
- `gt_free/input_manifest.json`;
- `gt_free/abiss_segment_stats.*`;
- `gt_free/cc3d_chunk_shards/` with compact resumable evidence, not necessarily dense labels;
- `gt_free/overlap_candidates.*` containing full-population ABISS/CC3D candidate edges;
- `gt_free/policy_<name>.json` with complete frozen rules and parameters;
- `gt_free/assignments_<name>.*` and explicit abstentions/reason codes;
- `evaluation_gt/partition_audit.json` and a concise table;
- `evaluation_gt/results.json` and `evaluation_gt/results.md`;
- representative visual-probe manifests for correct, incorrect, multi-anchor, mask-boundary,
  chunk-boundary, suspected-split, and no-benefit cases;
- focused CPU tests under the experiment directory or the nearest existing Zebrafinch test
  location.

The report must include a fusion funnel:

| stage | groups/fragments | candidate edges | accepted | correct | wrong | abstained | NERL |
|---|---:|---:|---:|---:|---:|---:|---:|

and explicit attribution of lost opportunity to no CC3D support, zero anchor, multiple
anchors, low confidence, morphology veto, mask/chunk boundary, and unresolved global identity.

## Required tests

Add synthetic/focused tests for at least:

1. correct CC3D@0.70 foreground semantics and 26-connectivity;
2. clipped border-chunk tissue-mask alignment;
3. collision-free global namespacing of identical local CC3D IDs;
4. deterministic ABISS/CC3D overlap counts and sharded reduction;
5. one CC3D component with one anchor plus fragments produces assignment candidates;
6. one CC3D component with two anchors always abstains;
7. transitive fragment chains cannot connect two anchors;
8. a single ABISS label split into several CC3D components is unchanged by the primary
   assignment policy;
9. chunk-face candidates are deduplicated and independent of processing order/shard count;
10. GT-derived paths/artifacts are rejected by the GT-free stages;
11. LUT remapping reproduces dense relabel semantics on a small synthetic volume;
12. existing pretrain `compare_cc3d_abiss.py` behavior remains reproducible if shared helpers
    are modified.

Tests must run on CPU. Large-volume execution may use the interactive or scheduled cluster,
but correctness may not depend on a GPU.

## Verification

At minimum:

1. run focused tests in the `pytc` conda environment;
2. run a synthetic plus one-border-chunk smoke test;
3. reproduce the correct arm0 tier10 CC3D@0.70 row and arm0_96 ABISS 0.444376 baseline;
4. complete the Phase 1 whole-skeleton audit before Gate A;
5. if Gate A passes, process all 726 chunks with manifest completeness and deterministic
   shard-reduction checks;
6. rerun policy selection from cached features and require identical assignments;
7. freeze assignments, then run the canonical whole-volume test-50 evaluation;
8. audit every GT-free manifest for skeleton/owner/NERL leakage;
9. record skipped stages and the exact decision gate that stopped them.

Do not rewrite a full segmentation volume merely to evaluate a global label LUT. If a later
visualization needs dense output, materialize only bounded crops or a reversible overlay after
the LUT result is established.

## Constraints

- Keep research orchestration in `dev/zebrafinch/abiss_cc3d_fusion/`.
- Reuse canonical CC3D, coordinate, mask, arm0_96 LUT, and NERL helpers where possible.
- Do not break the older pretrain CC3D/ABISS comparison or its recorded reproduction path.
- Do not add dependencies.
- Do not modify or overwrite canonical affinities, ABISS segmentation, tissue masks, LUTs,
  or prior reports.
- Do not sweep CC3D thresholds; this task tests fixed `0.70`.
- Do not use local-chunk NERL to select a whole-volume policy.
- Do not use decoder disagreement alone as a split, join, or routing decision.
- Do not union multiple established ABISS anchors, directly or transitively.
- Do not hide false merges behind an improved aggregate NERL.
- Do not create a production decoder/config/public API until this experiment establishes
  value and a later task requests promotion.
- Do not create commits during the CCC run.

## Out of scope

- Implementing affinity TTA or recreating deleted r10 TTA/min artifacts.
- Training or tuning an affinity model.
- Tuning ABISS agglomeration parameters.
- Generic CC3D whole-volume stitching as the primary output.
- GT-driven per-neuron decoder routing.
- Broad ABISS splitting based only on partition disagreement.
- Unrestricted lower-threshold fill, partition join, or coverage replacement.
- General endpoint curve fitting, Hungarian gap completion, or global learned linking.
- Full-volume dense output before a LUT-expressible fusion demonstrates value.
- Production integration into `connectomics.decoding`.
