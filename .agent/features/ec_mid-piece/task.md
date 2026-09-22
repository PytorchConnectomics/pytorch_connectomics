# Task

Implement and run a GT-leakage-safe Zebrafinch experiment that tests whether the
**mid-piece rung** identified in `dev/zebrafinch/lessons.md` L126 can be recovered by
affinity-weighted assignment of small fragments to clean neuron anchors.

The experiment must answer, in order:

1. Can native-scale affinity identify the correct host for a meaningful fraction of
   mid-sized pieces without introducing false merges?
2. Is failure caused by no physical contact, ambiguous anchor ownership, contaminated
   anchors, or weak/misregistered affinity?
3. On the exact same frozen candidates, does a verified larger-context/multiscale affinity
   source add independent information?
4. Which remaining cases genuinely require a separate gap-completion method such as
   curve fitting, bidirectional tracking, or global Hungarian assignment?

This is a bounded diagnostic and error-correction experiment under `dev/zebrafinch/`.
Do not turn it into a production decoder refactor, retraining campaign, or full-volume
segmentation rewrite.

## Motivation and measured target

The canonical arm0_96 error decomposition in `dev/zebrafinch/lessons.md` L126 reports:

| oracle step | NERL | increment |
|---|---:|---:|
| arm0_96 base | 0.444376 | — |
| safely join substantial pieces | 0.567056 | +0.122680 |
| absorb GT 10–49-node mid pieces | 0.846264 | +0.138303 |
| absorb dust | 0.941244 | +0.094980 |

The exact L126 counts are 668 substantial pieces, 1,887 mid pieces, and 8,979 dust
pieces. Mid pieces contain only about 7.65% of skeleton mass but account for about 37%
of the measured post-join oracle gain. They are therefore a separate absorption problem;
they must not be recovered by simply making the large-branch linker more permissive.

The `10–49 GT skeleton nodes` definition is an evaluator-only oracle category. It is not
available to a real decoder. This experiment must learn whether a GT-free candidate
population based on segmentation observables contains those pieces and whether affinity
can assign them safely.

Prior Zebrafinch evidence provides useful controls:

- `dev/zebrafinch/aff_absorb.py` showed that affinity-weighted fragment absorption can
  improve a local Ran crop while contact-area ownership can be harmful.
- `dev/zebrafinch/aff_multihop.py` showed that assignment-only multi-hop absorption can
  help when fragments never union two anchors, but relay chains need explicit bounds.
- L116 showed a GT-free core/fragment policy where reciprocal affinity and a winner margin
  improved NERL without measured regressions; permissive use of the same cue was unsafe.
- L118 showed that source-indexed r10 was aligned correctly but a face-local r10 scalar did
  not resolve continuation identity. Larger-offset evidence may veto or corroborate an
  ownership decision; it must not be assumed to solve it.
- L123/L125 showed that geometry creates proposals but is not sufficient evidence for
  merging. Most substantial breaks touch directly; only a small residual requires true
  gap completion.

The current FFN reference is **NERL 0.538003**. Only a fully GT-free result may be compared
against that number. Oracle-backbone and oracle-clean-host rows must be labelled as ceilings,
never as an FFN-beating result.

## Scientific hypotheses

### H1 — direct absorption

Many deployable mid-sized fragments already touch their true host. A fragment-to-host
boundary-affinity score, combined with a best-versus-runner-up margin and conservative
topology gates, can assign some of them with high precision.

### H2 — multiscale corroboration

For candidates that are geometrically valid but ambiguous at native scale, a verified
larger-context affinity source may improve host ranking or supply a useful veto. It is a
feature on an already proposed fragment-host edge, not a coarse connected-component
decoder and not permission to bridge arbitrary space.

### H3 — staged residual

Zero-contact pieces and pieces with no confident host after H1/H2 form a different problem.
Only that measured residual should be handed to curve fitting, bidirectional weak-region
tracking, or global matching. Hungarian assignment can enforce exclusivity among evidenced
connectors; it cannot manufacture evidence through an irrecoverable affinity region.

## Ground-truth firewall

The experiment must physically and logically separate proposal from evaluation.

### GT-free side

Candidate generation, anchor/fragment definitions, feature extraction, scoring,
thresholds, tie-breaking, component resolution, and frozen assignments may use only:

- the segmentation and raw image/affinity artifacts;
- GT-free segment statistics, contact geometry, endpoint geometry, and topology;
- settings calibrated outside `test_50_skeletons.h5`;
- explicitly verified metadata describing coordinates, affinity offsets, and storage mode.

Write all such artifacts beneath a `gt_free/` directory. Every frozen assignment manifest
must record `gt_free: true` and `frozen_before_evaluation: true`.

### Evaluator-only side

Only after the assignment manifest is immutable may the evaluator read
`/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`. GT may then:

- identify which sampled fragments belong to the L126 mid-piece oracle category;
- label the proposed host as correct, wrong, contaminated, or unresolved;
- compute NERL, per-skeleton deltas, oracle-gain recovery, and risk/coverage curves;
- produce clearly named diagnostic ceilings such as an oracle-clean-host row.

Write these artifacts beneath `evaluation_gt/`. No data derived there may be read back by
the selector in the same test run. Add a focused test or manifest audit that fails if the
GT-free stage accepts a skeleton/LUT/owner input.

Do not tune a threshold, size range, weight, or feature combination on the 50 test
skeletons and then report that same result as held-out. If an independent calibration
volume with compatible annotations is not available, report the complete predeclared
risk/coverage sweep as diagnostic and nominate a setting for a future confirmation run;
do not call the selected test point validated.

## Inputs and baselines

Use the canonical arm0_96 artifacts already referenced by the analysis scripts:

- segmentation:
  `dev/zebrafinch/wholevol_arm096_fullmask/seg_arm096_fullmask/precomputed/seg/seg_arm096_fullmask`;
- canonical evaluator and LUT ownership from `dev/zebrafinch/analyze_arm096.py` and
  `dev/zebrafinch/reports/arm096_lut/`;
- error inventory in `dev/zebrafinch/reports/arm096_error_structure.json` and
  `dev/zebrafinch/reports/arm096_breaks_v2.json`;
- canonical metric implementation and weights used by the existing arm0_96 scripts.

Before feature extraction, resolve and record the exact affinity artifact, dataset, shape,
voxel order, spatial crop, channel offsets, and affinity storage convention. Do not silently
reuse the legacy `2 - axis` channel assumption in `aff_absorb.py`. Affinity semantics must
agree with the canonical helpers in `connectomics/data/processing/affinity.py` and with the
contract in `.agent/features/affinity_tta/task.md` when a TTA artifact is used.

Evaluate two explicitly separated baselines:

1. **Honest GT-free baseline:** the best already frozen autonomous substantial-piece linker
   artifact, if one exists and passes its own manifest/metric checks; otherwise raw arm0_96.
   Record the exact input artifact and its reproduced NERL before doing any mid-piece work.
2. **Mechanism ceiling:** the L123/L126 oracle substantial-piece joins, used only to measure
   how well the absorption mechanism behaves when the backbone is already correct.

The experiment must not substitute the oracle backbone when reporting the honest score.
If the autonomous linker is unavailable, proceed on raw arm0_96 and state that the result
tests mid-piece absorption before the substantial-linking stage is complete.

Anchor contamination is not yet solved autonomously. Therefore report both:

- the honest policy with only GT-free anchor-quality/quarantine signals; and
- an evaluator-only oracle-clean-anchor ceiling that removes the known contaminated hosts.

The latter diagnoses lost potential but may not alter the honest assignment file.

## Experiment design

### Phase 0 — reproduce and inventory

Reproduce the canonical baseline NERL to numerical tolerance and the L126 piece counts and
mass fractions before testing a policy. Abort with an actionable error if the segmentation,
skeleton set, LUT, coordinate mapping, or metric configuration does not reproduce.

Build a full-population, GT-free segment inventory containing at least:

- segment ID, voxel count, bounding box, centroid, and border-touch flags;
- inexpensive shape descriptors available without GT, such as extent, elongation,
  estimated centerline length, caliber, and endpoint/bushiness indicators where reliable;
- membership in each predeclared anchor/fragment size band;
- direct RAG degree and count of eligible anchor neighbors.

Do not choose one magical proxy for `10–49 GT nodes` after looking at test labels. Predeclare
a small set of interpretable GT-free fragment bands (for example voxel mass and/or estimated
centerline length bins), run candidate generation for all of them in one pass, and use GT
only afterward to report which band captures the oracle mid-piece population.

The first report must partition candidate fragments into:

- zero eligible anchor contacts;
- exactly one eligible anchor contact;
- multiple eligible anchor contacts;
- contacts only to quarantined/low-quality anchors.

This topology inventory is the first decision gate: it determines how much of the mid-piece
rung is a direct-absorption problem versus a no-contact gap problem.

### Phase 1 — freeze the candidate table

Construct a face-contact RAG over the full segmentation. A candidate row represents one
fragment-host edge and must contain enough information to rescore without rereading the
whole volume. At minimum include:

- fragment ID and candidate anchor ID;
- contact voxel count and spatial extent of contact;
- native-scale boundary affinity count, mean, p50, p90, and a low-tail statistic;
- endpoint-versus-flank contact descriptor;
- fragment/anchor caliber compatibility and simple tangent compatibility when available;
- number of candidate anchors for the fragment;
- best score, runner-up score, and margin fields computed by the selector rather than baked
  into raw evidence;
- validity/missingness flags for every evidence source.

Use coordinate-local streaming or bounding-box reads; do not load a whole affinity volume
into RAM. Verify feature extraction on synthetic contacts and a small visual crop before the
whole-volume pass. Include at least one source-indexed affinity test that would fail under a
channel swap, sign flip, or one-voxel anchoring error.

Freeze this native-scale candidate table. Later multiscale comparisons must join onto these
same `(fragment_id, anchor_id)` rows. They may not generate a more favorable candidate
population.

### Phase 2 — native-scale assignment-only policies

Evaluate the following predeclared ablations:

| policy | purpose |
|---|---|
| no absorption | reproduced baseline |
| largest contact area | negative/control ownership cue |
| native affinity mean | simple affinity baseline |
| native affinity p90 | robust strong-boundary cue |
| native affinity + winner margin | ambiguity-aware assignment |
| native affinity + margin + morphology | conservative fused policy |

The primary operation is **assignment-only absorption**:

- a fragment may inherit the label of one existing anchor;
- an absorbed fragment must never union two anchors;
- a multi-anchor fragment is accepted only when one host is a clear winner under the frozen
  rule; otherwise abstain;
- a fragment assigned in one-hop mode cannot act as an independent anchor;
- reciprocal-best logic may be tested where its definition is meaningful;
- low-quality/quarantined anchors may not receive new fragments in the honest policy;
- deterministic tie-breaking and stable ordering are required.

First run one-hop absorption only. A bounded multi-hop ablation may follow only if one-hop is
precision-safe. Multi-hop must resolve on the complete segment graph before evaluation, enforce
one-core/one-anchor ownership for every resulting component, cap hop depth, record relay paths,
and never allow a relay chain to connect two anchors.

Parameter selection should emphasize risk/coverage, not a single maximized test score. Use a
small predeclared grid over affinity floors and best-versus-second margins. Cache features so
the grid is a cheap LUT-level rescore rather than repeated volume inference.

### Phase 3 — conditional multiscale evidence

Run this phase only if a larger-context affinity artifact exists or can be produced with a
clear, independently verified contract. A corrected TTA artifact is optional, not required for
Phase 1/2. Deleted or historically buggy `r10 TTA/min` outputs must not be recreated or reused
without passing `.agent/features/affinity_tta/task.md`.

The preferred comparison is the same frozen edge table with added evidence from one of:

- a verified long-offset affinity such as r10;
- a model/inference path with approximately 2× physical context;
- a coarse-scale affinity field mapped back to native coordinates with explicit offset and
  interpolation semantics.

Before scoring, use a non-symmetric synthetic oracle and several real visual probes to validate
channel order, direction, indexing endpoint, spatial alignment, scale mapping, and valid faces.
An `x2` field is not a native r1 edge merely because it has been upsampled; record the physical
edge represented by every channel.

Compare at least:

- native score alone;
- multiscale score alone;
- agreement gate: native accepts and multiscale does not contradict;
- conservative fusion of native and multiscale scores;
- multiscale veto on ambiguous or morphologically implausible native assignments.

Report incremental information on fixed candidates: changes in correct-host rank, winner
margin, precision at matched coverage, coverage at matched risk, and cases rescued/vetoed.
Do not decode the coarse affinity into connected components and merge those components into the
native segmentation in this experiment.

### Phase 4 — classify the unresolved residual

Do not implement broad low-threshold decoding, curve/Hungarian linking, or weak-region filling
as part of this task. Instead, emit a frozen residual manifest with one reason per unresolved
fragment:

- `no_anchor_contact`;
- `only_quarantined_anchor`;
- `ambiguous_multiple_anchors`;
- `weak_affinity`;
- `geometry_conflict`;
- `multiscale_contradiction`;
- `missing_or_invalid_evidence`.

For evaluator-only analysis, report how much L126 mid-piece oracle gain lies in each bucket.
The next gap-completion experiment should consume only geometrically plausible
`no_anchor_contact` cases and possibly the abstained ambiguous cases with explicit null options.

Recommend curve fitting / bidirectional tracking / Hungarian assignment only when the report
shows a material population that:

1. lacks direct contact to the true host;
2. has endpoint geometry supporting a bounded connector corridor; and
3. has at least some image or affinity evidence along that corridor.

Lowering a decoder threshold is allowed only in a future connector-realization stage, inside a
narrow corridor for an already selected source-target hypothesis. It is not an experiment-wide
way to create candidates.

## Local/global fusion diagnostic

To support the broader linking investigation, keep local evidence and global ownership as two
separable passes:

1. **Local pass:** generate contact/corridor evidence and per-edge scores independently within
   chunks, including overlap provenance and invalid-boundary flags.
2. **Global pass:** resolve anchor ownership, winner margins, quarantine state, one-anchor
   invariants, and conflicts over the full graph.
3. **Fusion:** accept only globally consistent assignments supported by local evidence; retain
   an abstention state.

Do not let chunk borders define ownership or permit the same fragment to receive inconsistent
hosts in different chunks. Deduplicate overlap evidence deterministically and add a test where
the same synthetic object crosses a chunk boundary. This experiment need not implement a new
distributed framework; a chunked feature extractor followed by one global table resolver is
sufficient.

## Evaluation contract

Reuse the canonical arm0_96 LUT-level evaluator where exact for assignment-only changes. Build
candidates over the full segment population even when only skeleton-carrying labels affect the
cheap NERL score. For multi-hop, resolve the full graph before mapping skeleton nodes so an
unsampled relay cannot be omitted.

For every policy and baseline report:

- canonical NERL and change from its direct input baseline;
- fraction of the `+0.138303` L126 mid-piece oracle increment recovered;
- per-skeleton NERL changes, with counts improved/unchanged/regressed;
- accepted fragments and components;
- post-hoc correct-host precision and coverage of GT-mid pieces;
- wrong-host assignments, cross-neuron unions, and high-risk multi-owner components;
- maximum single-skeleton regression and total lost/gained weighted ERL;
- result stratified by zero/one/multiple anchor contacts, fragment proxy band, contact type,
  anchor quality, and confidence/margin bin;
- risk/coverage curves rather than only the best row;
- wall time and peak memory for candidate construction, rescoring, and evaluation.

Use strict contamination accounting: any accepted component containing material ownership from
multiple GT neurons is a false merge, even if the aggregate NERL happens to rise. State the exact
materiality threshold used by the existing evaluator and include zero-tolerance counts as well.

Only the honest GT-free row may be compared with FFN 0.538003. Put oracle and test-selected
diagnostics in a separate table with an unmistakable label.

## Decision gates

The experiment is scientifically complete even if the method fails, provided the frozen
artifacts and report make the failure mode identifiable.

### Proceed from inventory to direct absorption

- canonical baseline and L126 inventory reproduce;
- a nontrivial fraction of oracle mid pieces falls in GT-free candidate bands;
- affinity alignment and contact statistics pass focused tests.

### Promote a native-scale policy

- no high-risk anchor-to-anchor merge is introduced;
- no material weighted NERL regression versus its direct baseline;
- correct-host precision is high enough for error correction, reported with sample count and
  uncertainty;
- gain appears across multiple skeletons rather than one outlier;
- thresholds were calibrated independently or are explicitly labelled diagnostic.

An aspirational milestone is to exceed FFN 0.538003 on the honest pipeline, but failure to do so
does not justify relaxing safety gates.

### Claim multiscale value

- all geometry/offset/alignment tests pass;
- comparison uses the same frozen candidate rows;
- multiscale evidence improves precision at matched coverage or coverage at matched risk on an
  independently chosen operating point;
- the gain is not merely a test-selected threshold shift.

### Escalate to gap completion

- a material part of remaining L126 gain is in `no_anchor_contact` rather than merely low-score
  direct contacts; and
- visual/quantitative probes show recoverable corridor evidence.

If direct-contact candidates dominate but host ranking remains ambiguous, improve ownership or
anchor quarantine before building a curve/Hungarian connector.

## Deliverables

Place the experiment under `dev/zebrafinch/ec_mid_piece/` with a short README and narrow scripts
whose names make stage ownership clear. Reuse existing arm0_96 evaluator utilities rather than
copying metric code.

Required artifacts:

- `gt_free/input_manifest.json` — exact segmentation/affinity inputs, hashes or stable metadata,
  coordinate conventions, offsets, affinity mode, and baseline selection;
- `gt_free/segment_inventory.*` — cached GT-free segment descriptors;
- `gt_free/candidate_edges.*` — frozen native-scale fragment-host candidate table;
- `gt_free/policy_manifest_<name>.json` — complete frozen rule and parameter values;
- `gt_free/assignments_<name>.*` — deterministic fragment-to-anchor assignments and abstentions;
- `gt_free/residual_<name>.*` — unresolved cases with reason codes;
- `evaluation_gt/results.json` — machine-readable metrics and ablations;
- `evaluation_gt/results.md` — concise tables, risk/coverage summary, failure taxonomy, and next
  recommendation;
- a small visual-probe directory or manifest for representative correct, incorrect, ambiguous,
  contaminated-anchor, and no-contact cases;
- focused CPU tests for affinity indexing, chunk-boundary deduplication, one-anchor invariants,
  deterministic assignment, GT firewall, and LUT-level evaluation.

The result report must include a compact stage accounting table:

| stage | candidates | accepted | correct | wrong | abstained | NERL | delta |
|---|---:|---:|---:|---:|---:|---:|---:|

and a funnel showing how much GT-mid oracle headroom is lost to candidate coverage, quarantine,
ranking ambiguity, confidence gates, and residual no-contact gaps.

## Verification

At minimum:

1. run focused unit tests in the `pytc` conda environment;
2. run a small crop/synthetic smoke test that exercises each affinity axis and a chunk border;
3. reproduce the arm0_96 baseline and L126 inventory;
4. rerun policy selection from cached candidates and require byte-stable or semantically identical
   assignments;
5. run the canonical 50-skeleton LUT-level evaluation only after freezing assignments;
6. audit manifests to confirm no GT path or owner label entered the GT-free stage;
7. if Phase 3 runs, include the multiscale/TTA geometry validation result in the report.

Record exact commands, environment, wall time, and peak memory in the README or report. If a
full-volume phase cannot run within the available interactive allocation, preserve the completed
cache and provide a resumable non-interactive command; do not replace the requested experiment
with an unrepresentative hand-picked crop result.

## Constraints

- Follow repository dependency ownership if any reusable package code is touched. Research-only
  orchestration stays in `dev/zebrafinch/ec_mid_piece/`.
- Do not add dependencies.
- Do not modify or overwrite canonical arm0_96 inputs, LUTs, FFN outputs, or prior reports.
- Do not recreate deleted `r10 TTA/min` artifacts unless the affinity-TTA contract is implemented
  and its correctness tests pass.
- Do not use GT to choose candidates, hosts, thresholds, proxy bands, or quarantine decisions in
  an honest run.
- Do not union anchors during fragment absorption.
- Do not hide unsafe merges behind an improved aggregate NERL.
- Do not make a production decoder API or config change unless the experiment establishes value
  and a later task explicitly requests promotion.
- Do not create commits during the CCC run.

## Out of scope

- Implementing `.agent/features/affinity_tta/task.md`.
- Training a new affinity model or tuning ABISS.
- Broad whole-volume lower-threshold decoding.
- General curve fitting, Hungarian gap linking, or weak-region fill implementation.
- Dust absorption after the mid-piece stage.
- A production `connectomics.decoding` integration or public API change.
- Writing a new full segmentation volume merely to evaluate LUT-expressible assignments.
