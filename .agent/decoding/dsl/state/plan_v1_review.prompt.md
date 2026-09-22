You are reviewing plan v1 for a CCC run. You are the CODER: you will implement it.

You reviewed plan v0 and returned READY: no with 16 major findings. Plan v1 accepts all of them;
four changed the design (the circular regression oracle, ROI-local anchor fractions, repairing
bridge-class conflicts that are known-invalid for this operator, and the invented 0.02 threshold).

Judge plan v1 on:
1. Are the four design changes correct and sufficient, or do they introduce new holes?
2. Is the golden-fixture ordering actually enforceable, and does it give a genuine oracle?
3. Test #3 and the V4 measured gate are the two things standing between this and "passes every
   test while repairing nothing". Are they now strong enough to fail a bad implementation?
4. Is anything from your 16 findings addressed only nominally?
5. Is the plan now executable end to end, or does it still defer decisions that must be made
   before coding starts?

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

=========================== BEGIN task.md ===========================
You are working inside an existing connectomics segmentation repository. Implement a minimal, production-quality framework for local structural checkpointing of affinity-based segmentation.

The concrete application is zebra finch EM segmentation. False merges can amplify during union-only agglomeration: once a provisional component contains multiple biological identities, later global growth preserves and often enlarges the contamination. We want to repair or constrain such errors at a local checkpoint, then run a fresh global agglomeration pass over the corrected segmentation graph.

Do not build a general biological reasoning system. Do not implement glia classification, morphology reasoning, junction reasoning, or an agent loop in this task.

Implement exactly one useful vertical slice:

    nucleus-instance anchors
        → descriptive observations
        → deterministic conflict policy
        → local split and cannot-link actions
        → verification
        → export for a later global pass

The framework must nevertheless be extensible so future independent passes can add morphology, junction, artifact, or acquisition descriptors without being merged into one monolithic pass.

Before changing code
====================

1. Run `git status --short`.
2. Do not overwrite or revert unrelated local changes.
3. Inspect the repository structure and current conventions.
4. Search for and read the existing nucleus-related code, especially files or equivalents named:

   - `nucleus_competitive_split.py`
   - `nucleus_anchor_merge.py`
   - `nucleus_shell_contamination.py`
   - `nucleus_fusion_audit.py`
   - existing ABISS / waterz / watershed cannot-link handling
   - any `NUC_PATH` handling
   - current volume, RAG, chunk, and bounding-box I/O utilities

5. Reuse the current competitive-split and anchor-merge implementations as backends where possible. Refactor reusable functions out of CLI scripts if needed, but preserve existing command-line behavior.

Do not stop after writing a plan. Briefly state the implementation plan, then implement it, add tests, and run the relevant tests.

Conceptual architecture
=======================

Implement a typed “Segmentation Checkpoint DSL” with this stable protocol:

    DESCRIBE → CERTIFY → ACT → VERIFY

After checkpoint execution, a separate global decoder will consume the corrected segmentation and exported constraints.

The DSL is an intervention intermediate representation, not a reasoning language. Today, deterministic code maps descriptions to actions. A future agent may replace that policy, but the future agent must emit the same validated action records and must never mutate a segmentation directly.

Keep these layers separate:

1. Observation
   Immutable descriptive measurements.

2. Certificate
   A hard assertion justified by a trusted anchor.

3. Action plan
   A replayable list of requested mutations or constraints.

4. Executor
   Deterministic code that validates preconditions and applies actions.

5. Verifier
   Deterministic code that checks postconditions and records results.

Avoid semantic labels
=====================

Do not add core labels such as:

    is_glia
    is_neuron
    is_vessel
    cell_type

The core framework must use descriptive, namespaced fields such as:

    anchor.distinct_count
    anchor.overlap_fraction
    component.volume_voxels
    component.physical_extent_um
    morphology.porosity
    morphology.sheetness
    graph.contact_degree
    graph.distinct_tube_contacts
    junction.incident_arm_count
    acquisition.tile_seam_distance

Only the first three nucleus-related fields need to be computed in this task. The others may appear in documentation as future examples, but do not implement their detectors.

Anchors are different from semantic labels. Implement one anchor kind:

    nucleus_instance

Its contract is:

- one trusted nucleus instance belongs to one cell;
- two distinct trusted nucleus IDs imply distinct biological identities;
- one nucleus does not certify that the entire component is pure;
- absence of a nucleus makes no assertion.

Core data model
===============

Adapt names and locations to the repository, but implement typed equivalents of the following concepts.

1. EntityRef

Fields should include, where applicable:

- entity kind:
  `fragment`, `component`, `contact`, `anchor`, `territory`
- stable ID
- optional bounding box
- optional source chunk / volume identifier

2. Descriptor

An immutable record containing:

- stable descriptor ID
- subject EntityRef
- namespaced key, such as `anchor.distinct_count`
- typed value
- optional units
- optional confidence
- operator name and version
- provenance:
  input artifacts, input hashes or versions, configuration hash, timestamp
- scope / bounding box

Use namespaced string keys rather than a closed enum containing every possible future descriptor.

3. Certificate

For this MVP, implement:

    distinct_anchor_identity_conflict

Fields:

- certificate ID
- affected component
- anchor kind
- distinct anchor IDs
- descriptors that support it
- strength: `hard`
- operator version
- provenance

The certificate is emitted when one provisional component has significant overlap with at least two distinct trusted nucleus instances.

4. ActionSpec

Support typed schemas for at least:

- `annotate`
- `split_by_anchor`
- `forbid_merge`
- `consolidate_same_anchor`
- `rebuild_local_rag`

It is acceptable to define schema placeholders for future:

- `hold_edge`
- `release_edge`

Do not implement their behavior yet.

Each action should include:

- stable action ID
- operation
- target or targets
- parameters
- preconditions
- expected postconditions
- supporting certificate or descriptors
- execution status
- failure message, if any

5. CheckpointPlan

Contains:

- plan version
- checkpoint/pass ID
- input artifact references
- descriptors
- certificates
- ordered actions
- expected invariants
- deterministic configuration hash

6. CheckpointResult

Contains:

- original plan ID
- executed actions
- skipped or failed actions
- verification outcomes
- output artifact references
- summary statistics

Use Pydantic if it is already a repository dependency. Otherwise use standard-library dataclasses plus explicit validation. Do not introduce a heavyweight dependency solely for this framework.

Serialization
=============

Use the repository’s existing configuration and serialization conventions.

Preferred behavior:

- YAML or JSON for pass specifications
- JSON or JSONL for immutable descriptor and action records
- deterministic, stable IDs derived from content where practical
- append-only execution records
- explicit schema version fields

Do not use Python `eval`, arbitrary expressions, or arbitrary code embedded in YAML.

Implement a minimal safe condition syntax if needed:

    field
    operator: eq | ne | gt | ge | lt | le
    value

For this MVP, the only required policy condition is:

    anchor.distinct_count >= 2

First checkpoint operator
=========================

Implement an independently executable operator analogous to:

    NucleusAnchorCheckpoint

It should operate on a provisional local or composite-chunk segmentation, not require modification of ABISS internals.

Inputs should be adapted to actual repository formats and utilities:

- provisional segmentation label volume
- nucleus-instance label volume
- affinity volume or current split backend inputs
- optional RAG / fragment metadata
- bounding box or checkpoint scope
- output directory
- configuration

The operator must have four distinct phases.

A. Describe

For each relevant provisional component, compute:

- component label / stable ID
- component voxel count
- overlapping nucleus anchor IDs
- overlap count for each anchor
- fraction of each anchor’s own mass represented in this component
- distinct qualifying anchor count
- repair bounding box or scope

Selection thresholds must be relative to each anchor’s own mass, not only absolute voxel counts. Preserve current measured behavior and existing configuration where available; do not invent new scientifically meaningful defaults without documenting them.

Emit descriptors even if no action is required.

B. Certify

If a component has at least two distinct qualifying nucleus IDs, emit:

    distinct_anchor_identity_conflict

No semantic inference is involved.

C. Plan actions

For each conflict, emit an ordered plan:

1. `split_by_anchor`

   Use the existing competitive split backend.

   Required safety properties:

   - operate only inside the original conflicting component or a bounded repair ROI;
   - do not absorb voxels from unrelated components;
   - support unassigned or abstained voxels if the current backend can support this;
   - do not claim equivalence between post-hoc and online agglomeration;
   - record the exact affected voxel scope.

2. `forbid_merge`

   Emit pairwise cannot-links between distinct resulting anchor territories.

   These constraints must be durable and exportable to the later global pass.

3. `consolidate_same_anchor`

   Run only after exclusion / splitting.

   Consolidate pieces belonging to the same trusted nucleus only within the valid repair scope and only if this does not cross a distinct-anchor cannot-link.

4. `rebuild_local_rag`

   Recompute or invalidate the affected local graph state using existing repository utilities.

If the global decoder already consumes a specific cannot-link format, implement an adapter to that format. Also keep a canonical framework-level constraint manifest independent of ABISS.

D. Verify

Implement explicit invariant checks:

1. No qualifying output component contains more than one distinct nucleus anchor.

2. Each qualifying nucleus anchor has one dominant output territory, subject to existing expected tolerances.

3. Voxels outside the declared repair scope are unchanged up to label renumbering / partition equivalence.

4. Distinct anchor territories have exported cannot-link constraints.

5. Same-anchor consolidation happens only after the identity conflict has been resolved.

6. Running the checkpoint a second time on its own output produces no new repair actions, or only a documented no-op plan.

7. The result is deterministic under identical inputs and configuration.

Where appropriate, reuse or refactor the strict contamination logic currently used by `nucleus_shell_contamination.py`.

Do not use `nucleus_fusion_audit.py` as the only verifier if it measures only dominant-segment sharing and can miss minority contamination.

Execution modes
===============

Provide a CLI following repository conventions with conceptually equivalent commands:

    checkpoint describe
    checkpoint plan
    checkpoint apply
    checkpoint verify
    checkpoint run

Exact command names should follow existing CLI style.

Required modes:

- dry-run / plan-only:
  write descriptions, certificates, and actions without changing segmentation

- apply:
  execute a previously serialized plan

- verify:
  verify a previously executed plan and its outputs

- run:
  describe, plan, apply, and verify in order

The plan must be inspectable before application.

Do not hardcode development paths. All data paths, bounding boxes, thresholds, pooling factors, and output names must come from CLI arguments or configuration.

Pipeline composition
====================

Do not build one giant checkpoint class.

Implement:

- a small operator interface or protocol;
- a registry or explicit operator factory;
- a sequential pipeline runner;
- one implemented operator: `nucleus_anchor`;
- independent pass specifications.

A later configuration should be able to express:

    passes:
      - nucleus_anchor
      - morphology_descriptors
      - junction_descriptors

without those passes sharing one classifier or one action policy.

Only `nucleus_anchor` is implemented in this task.

The output of one pass may become the input state for the next pass. Every pass must retain separate provenance and action records.

Recommended repository structure
================================

Adapt to existing conventions rather than forcing these exact paths, but a reasonable internal decomposition is:

    checkpoint/
        schema.py
        engine.py
        registry.py
        actions.py
        verification.py
        serialization.py
        cli.py
        operators/
            base.py
            nucleus_anchor.py

Keep numerical algorithms separate from orchestration and schema code.

Refactor existing split / anchor functions into reusable numerical modules if necessary. CLI wrappers should remain thin.

Testing
=======

Add focused unit and integration tests using small synthetic 3D arrays.

At minimum test:

1. One component, one nucleus
   - descriptors are emitted;
   - no conflict certificate;
   - no split action.

2. One component, two distinct nuclei
   - hard identity-conflict certificate;
   - split action;
   - pairwise cannot-link;
   - postcondition: no output component contains both anchors.

3. One atomic supervoxel overlapping two nuclei
   - region-level edge rejection alone is insufficient;
   - voxel-level split action is planned and applied.

4. Multiple fragments overlapping the same nucleus
   - consolidation is planned only after exclusion;
   - no distinct-anchor territories are joined.

5. Minority contamination
   - a small but qualifying overlap is detected using relative anchor mass;
   - verifier does not rely only on dominant-segment sharing.

6. Subthreshold overlap
   - no false conflict certificate.

7. Scope safety
   - labels outside the repair scope are partition-equivalent to the input.

8. Idempotence
   - a second run is a no-op.

9. Serialization round trip
   - plan and result records preserve all typed fields.

10. Determinism
    - identical inputs produce identical plans and stable IDs.

11. Failed precondition
    - executor refuses an action if the input segmentation or configuration hash no longer matches the plan.

12. Existing regression
    - current nucleus split and anchor scripts continue to behave as before.

Use physical z/y/x ordering and anisotropic coordinates consistently with the existing repository. Do not silently change axis conventions.

Performance and I/O
===================

Preserve existing scalable I/O behavior.

- Do not load the whole zebra finch volume into RAM if current code works blockwise.
- Restrict expensive operations to conflict components and their bounded ROIs.
- Reuse memory-mapped, HDF5, Zarr, CloudVolume, or repository-specific abstractions already present.
- Record timing and affected-voxel counts per action.
- Avoid copying full volumes merely to create audit artifacts.

Documentation
=============

Add a concise design document, for example:

    docs/segmentation_checkpoint_dsl.md

It should explain:

1. Why the DSL separates descriptions, certificates, actions, and verification.
2. Why descriptive labels are preferred to semantic labels.
3. Why nucleus instances are anchors rather than cell-class predictions.
4. Why the first implementation contains only one anchor pass.
5. How future morphology and junction passes can emit descriptors without changing the executor.
6. How a future agent can consume descriptions and emit action plans while remaining unable to mutate segmentation directly.
7. The distinction between:
   - hard `forbid_merge`;
   - future temporary `hold_edge`;
   - local partition refinement;
   - later global reclustering.

Include one complete example pass specification and example emitted plan.

Also document future descriptor examples without implementing them:

    morphology.porosity
    morphology.sheetness
    graph.contact_degree
    graph.distinct_tube_contacts
    growth.threshold_velocity
    junction.incident_arm_count
    junction.orientation_modes
    acquisition.tile_seam_distance

Non-goals
=========

Do not:

- implement an LLM or agent loop;
- implement glia classification;
- add semantic cell-type labels to the core schema;
- implement morphology or junction detectors;
- modify the global agglomeration algorithm itself unless a small constraint adapter is required;
- combine all future priors into one pass;
- replace the existing numerical split algorithm without evidence;
- claim that post-hoc splitting is mathematically identical to online constrained agglomeration;
- add arbitrary YAML expression execution;
- perform unrelated repository refactoring.

Acceptance criteria
===================

The task is complete when:

1. A dry run produces an inspectable, serialized checkpoint plan.
2. The plan contains descriptive observations rather than semantic labels.
3. A multi-nucleus component emits a hard identity-conflict certificate.
4. Applying the plan calls the existing competitive split / anchor logic through reusable APIs.
5. Distinct output anchor territories receive durable cannot-links.
6. Verification catches minority contamination.
7. The stage is idempotent and scope-safe.
8. Existing nucleus-related CLIs remain usable.
9. Tests pass.
10. The final response gives:
    - files changed;
    - architectural decisions;
    - tests and commands run;
    - exact dry-run and full-run commands for the existing `worst3` development crop, based on the repository

---

Run context (appended by the CCC coordinator; the specification above is unchanged)
===================================================================================

This run was requested as: *"read task.md and the lessons and consider the context above, code it
up this new watershed method."* The measured findings that justify this design are recorded in
`dev/zebrafinch/lesson_abiss.md` **L126**; read it before planning. The load-bearing ones:

* **The error class is a burst, not noise.** Between three fused somata the pairwise affinity
  bottleneck is ~0.999 — any separating surface must sever affinity >= 0.999 against a 0.3
  agglomeration threshold. There is no weak surface to find, so cut placement cannot be derived
  from affinity and must come from the anchor. (`dev/zebrafinch/conflict_bottleneck.py`)
* **Region-graph repair cannot fix it.** One watershed supervoxel carried **100.0%** of the
  contamination. This is why the task demands a *voxel-level* `split_by_anchor` and why test 3
  ("one atomic supervoxel overlapping two nuclei — region-level edge rejection alone is
  insufficient") is the discriminating test.
* **Three in-pipeline attempts each measured exactly 0.0000** (merge ordering; a
  `nuc_can_merge` CONFLICT-clause veto; a global per-supervoxel nucleus table). Any claim that a
  new mechanism works must be backed by a number, not by the mechanism looking correct.
* **Merging is a union and can only coarsen; competition is an assignment and can refine.** This
  is why the operator is a seeded flood, not a re-ordering of merges.
* **Exclusion must precede anchoring.** `consolidate_same_anchor` before the split would fuse all
  conflicting anchors through the shared component. The task's action ordering encodes this;
  preserve it.
* **Thresholds must be relative to each anchor's own mass.** An absolute floor let a 1,396-voxel
  leak (5 mask voxels, 0.3% of that nucleus) trigger splitting a healthy soma in half.
* **`nucleus_fusion_audit.py` is structurally blind** to minority-mass fusion — it scored 0 on a
  run that had the bug. `nucleus_shell_contamination.py --tol 0.0` is the strict test.
* **NERL cannot score this error class.** Correcting five fused somata changed NERL by exactly
  0.000000 (0 of 500,845 skeleton nodes affected). Do not add NERL as an acceptance criterion for
  this framework; verification is the instance-level invariants in section D.

Measured results the implementation should be able to reproduce as a regression:

* `worst3` crop (arm1_ft): misplaced mask voxels **115,479 -> 5,071**, cross-nucleus shared
  segments **1 -> 0**, dominance 0.8957/0.9104/0.8988 -> 0.9965/0.9949/0.9959.
* `arm0_96` whole volume: **32** components hold >= 2 nuclei, **74 of 465** neurons (15.9%).
  These split **16 soma-contact / 16 neurite-bridge** by nearest-anchor-pair gap; the competitive
  split is correct only for the contact class. A future pass handles bridges — do not conflate
  them in this one.

Existing code to reuse as backends (all in `dev/zebrafinch/`, which is gitignored scratch):
`nucleus_competitive_split.py`, `nucleus_anchor_merge.py`, `nucleus_shell_contamination.py`,
`flood_contact_units.py`, `nucleus_split_wholevol.py`. Note `dev/` is NOT tracked by git, so
reusable numerical code must be lifted into a tracked package for this framework to be testable.
=========================== END task.md ===========================

=========================== BEGIN artifacts/plan_v0_review.md ===========================
# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v0_review.review.raw.md`. Sixteen major findings and one minor, none softened here.

Placement inside `connectomics/decoding/checkpoint/` is judged sound and non-circular, and the plan
nominally covers the five commands, twelve tests, four ordered actions, seven invariants and the
design document. It is not executable safely: the regression oracle is circular (the refactored
scripts would be compared against themselves), several action and verification contracts are
vacuously satisfiable, and — the finding with teeth — the proposed tests could all pass while the
framework repairs nothing useful.

## Findings

- [major] Test #12 has no independent pre-refactor oracle. Rewiring the development scripts and then comparing them with the lifted functions risks comparing the new implementation with itself. The plan must first capture checked-in golden inputs and exact expected arrays, metrics, or hashes from the untouched scripts. Changes to gitignored `dev/` scripts are also absent from a normal review or clean checkout, so tests cannot depend on those modified files.

- [major] The plan does not require the measured `worst3` outcomes to be reproduced or asserted. Merely running dry-run and full-run commands does not demonstrate improvement. The validation should record the strict contamination, cross-nucleus sharing, and dominance figures and compare them with the supplied `115,479 → 5,071`, `1 → 0`, and dominance targets. Otherwise the implementation could be another mechanism with an effective result of 0.0000.

- [major] Synthetic test #3 is insufficient as proposed. An implementation could carve away anchor voxels, assign arbitrary checkerboard territories, or discard most of the component and still satisfy “no component contains both anchors.” The test needs a known seeded-watershed oracle, voxel conservation or explicit abstention accounting, expected anchor-driven territories, and assertions that non-anchor component mass is meaningfully assigned.

- [major] The numerical lift omits essential backend behavior. The proposed function list does not include the reusable same-anchor consolidation logic from `nucleus_anchor_merge.py`, nor does it specify cost construction from affinities, marker construction, connectivity, tie-breaking, pooling/upscaling, anisotropic z/y/x handling, or deterministic handling of unreachable voxels. These choices determine the actual watershed result.

- [major] `min_share = 0.02` is unsupported by the supplied evidence and conflicts with the instruction not to invent scientific defaults. The supplied 0.3% failure establishes that an absolute threshold was unsafe; it does not establish 2% as correct. The implementation must preserve a cited existing value or require an explicit configured threshold.

- [major] `qualifying_anchors(hist, min_share)` cannot reliably calculate a fraction of each anchor’s own mass when execution is restricted to a repair ROI or block. It needs authoritative whole-anchor totals or an explicitly defined reference scope. Summing only the cropped histogram can turn a tiny truncated overlap into 100% of the observed anchor mass.

- [major] The action dependency rule is not fail-closed. Refusing consolidation only when an overlapping split remains unexecuted allows consolidation when the split is missing from a malformed plan, when a serialized status falsely says it executed, or before `forbid_merge` has installed the cannot-links. Consolidation must require executor-recorded successful split and cannot-link actions, a resolved supporting conflict, and no unresolved distinct-anchor conflict over the exact scope.

- [major] Planning `forbid_merge` before split results exist is not defined. The plan needs deterministic symbolic territory references or an explicit, validated result-binding mechanism that resolves anchor territories after splitting without silently synthesizing new unreviewed actions. The same issue affects stable action IDs and replay.

- [major] “Every output territory is a subset of the input component” is too weak and not operationally defined. Deleting all voxels would satisfy it. Verification must use exact parent-component and scope masks and check refinement, conservation or explicit abstention, absence of voxels from other input components, exact changed-voxel accounting, and outside-scope partition equivalence. A bounding-box overlap alone is insufficient for action ordering or this postcondition.

- [major] The seven invariants are named but not made executable. In particular, the plan does not define the dominant-territory tolerance for invariant 2, authoritative execution evidence for invariant 5, what constitutes a no-op for invariant 6, or which deterministic artifacts invariant 7 compares. Hashing records that include timestamps contradicts stable IDs and identical-plan determinism unless volatile provenance and timing fields are excluded or normalized.

- [major] The placement is only partly resolved. Keeping the implementation inside `decoding` fits the supplied dependency direction and creates no inherent cycle, provided it imports only permitted dependencies and sibling modules. However, a `scripts/checkpoint.py` that dispatches directly into `decoding` conflicts with the stated `scripts → runtime` contract. The CLI should enter through `runtime`, which may call the decoding checkpoint engine. A backend constraint adapter must also avoid importing a downstream runtime or decoder owner back into `decoding`.

- [major] The scalable I/O and export layer is absent from the proposed decomposition. Pure array kernels are appropriate, but the operator still needs bounded/blockwise reads, input artifact hashing/versioning, exact scope-mask storage, corrected-segmentation output in a decoder-consumable format, append-only execution records, timing and affected-voxel counts, and avoidance of full-volume copies. None of these are covered by the synthetic tests.

- [major] The cannot-link adapter is left as “whatever the global decoder consumes,” with no verification that the exported constraints are consumable. The plan needs to identify the existing format during implementation and add an adapter-level integration test. Otherwise it can pass by writing a canonical manifest that no later global pass can use.

- [major] The action/schema coverage is incomplete. The task requires typed support for `annotate` in addition to the four ordered operational actions, but the plan only discusses the latter. It also says only three descriptor keys will be emitted without explaining how per-anchor overlap voxel counts, anchor IDs, component identity, and repair scope required by Describe are represented.

- [major] The plan applies splitting to every component satisfying `distinct_count >= 2` but does not reconcile this with the supplied result that competitive splitting is valid only for soma-contact cases and not the 16 neurite-bridge cases. It must define a conservative eligibility boundary—such as explicitly supplied contact scopes or an existing trusted selector—and record unsupported bridge conflicts without pretending to repair them.

- [major] The five command names are present, but their artifact contracts are not. The revised plan should define whether `describe` serializes descriptors only, how `plan` consumes or reproduces them, how `apply` binds a frozen plan to hashed inputs, and how `verify` locates the immutable plan, execution log, exact scope, and output segmentation.

- [major] The design-document plan does not enumerate the required contents. It must explicitly include all seven requested explanations, the four-way distinction among hard forbids, future holds, local refinement, and global reclustering, all listed future descriptor examples, and both a complete pass specification and a complete emitted-plan example.

- [minor] Updating the public API snapshot should not be assumed. A private decoding subpackage can remain outside `connectomics.decoding.__all__`; adding a facade export merely to satisfy discoverability would conflict with the small-public-API rule.

## Questions

1. What existing configuration or historical run establishes the qualifying-anchor threshold? If none does, should the CLI require it explicitly with no scientific default?
2. How will soma-contact scopes be selected without applying the known-invalid watershed repair to neurite bridges?
3. What checked-in fixture and immutable pre-lift outputs will serve as the independent regression oracle when `dev/` is unavailable in a clean checkout?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v0_review.md ===========================

=========================== BEGIN artifacts/plan_v1.md ===========================
# Plan v1

## Summary

All sixteen findings accepted. Four of them change the design rather than tighten it:

* **the regression oracle cannot be the thing being refactored** — golden fixtures must be captured
  from the *untouched* `dev/` scripts and checked in before a line is moved, because `dev/` is
  gitignored and absent from a clean checkout;
* **anchor-mass fractions cannot be computed inside a repair ROI** — cropping turns a truncated
  overlap into 100% of the observed anchor mass, so whole-anchor totals become a required input
  artifact rather than something derived on the fly;
* **the operator must not repair every conflict** — this session measured that competitive
  splitting is valid for the 16 soma-contact cases and *not* for the 16 neurite-bridge cases
  (nearest-anchor gap up to 80.6 um), and v0 applied it to everything with
  `distinct_count >= 2`, contradicting the evidence in its own run context;
* **`min_share = 0.02` was invented.** The measured 0.3% failure shows an absolute floor is unsafe;
  it establishes nothing about 2%. There is now no default.

The plan also stops claiming the framework is useful because it passes synthetic tests. The
acceptance gate is the measured `worst3` numbers — 115,479 -> 5,071 misplaced, 1 -> 0 shared,
dominance 0.8957/0.9104/0.8988 -> 0.9965/0.9949/0.9959 — asserted, not merely produced.

## Scope

Unchanged from v0 in what is built; changed in what is *required to be demonstrated*. Bridge-class
conflicts are explicitly in scope as **certified-but-unrepaired** records, and explicitly out of
scope for repair.

## Proposed Changes

### A. Placement, corrected (v0-F11, v0-F17)

`connectomics/decoding/checkpoint/` stands, but the CLI does **not**. `scripts/ -> runtime` is the
repo's contract, so:

```
scripts/checkpoint.py                 thin argparse shim, no domain logic
connectomics/runtime/checkpoint.py    entry functions: describe/plan/apply/verify/run
connectomics/decoding/checkpoint/     schema, engine, registry, actions, verification, io
connectomics/decoding/nucleus.py      pure array kernels
```

The constraint adapter lives in `runtime`, not `decoding`, so no downstream owner is imported
back into `decoding`. The subpackage stays private: **do not** add it to
`connectomics.decoding.__all__` (v0-F17) — the public-API snapshot should be unchanged, and if it
fails, that is a signal the layering is wrong.

### B. The regression oracle, captured before anything moves (v0-F1)

Order is mandatory:

1. build a small synthetic fixture plus a real `worst3` sub-ROI;
2. run the **untouched** `dev/zebrafinch/nucleus_competitive_split.py`,
   `nucleus_anchor_merge.py`, `nucleus_shell_contamination.py`, `flood_contact_units.py` on it;
3. check the outputs into `tests/fixtures/checkpoint/` as arrays plus a manifest of exact values
   and sha256s;
4. only then lift the kernels.

Tests compare the lifted kernels against those **immutable checked-in artifacts**, never against
the live `dev/` scripts. This makes the suite valid in a clean checkout where `dev/` does not
exist.

### C. The numerical lift, fully specified (v0-F4)

v0 listed six functions and omitted the choices that determine the result. The lift must carry, and
pin in the fixture manifest:

| aspect | contract |
|---|---|
| cost construction | `1 - min over affinity channels`, float32, after the store's convention/sigmoid restore |
| pooling | `min` for cost (preserves bottlenecks), `max` for masks, factor from config |
| upsampling | nearest, with the resulting +/- factor boundary error documented |
| markers | anchors pooled with unanimity (a pooled voxel takes an anchor only if its whole block agrees) |
| connectivity | explicit, and asserted in a test |
| tie-breaking | deterministic and documented |
| anisotropy | physical z/y/x preserved; no silent axis reordering |
| unreachable voxels | **abstain**, never forced to a winner; counted and reported |
| same-anchor consolidation | lifted from `nucleus_anchor_merge.py`, which v0 omitted entirely |

### D. Anchor totals as an input artifact (v0-F6)

`anchor.overlap_fraction` is a fraction of the anchor's **whole** mass. Inside an ROI that number
is not computable. So:

* a `describe --anchor-totals` step computes per-anchor whole-volume totals once and writes them
  as a first-class artifact with its own hash;
* every later stage takes that artifact as an input and refuses to run without it;
* the descriptor records both the local overlap count and the authoritative total it was divided
  by, so the provenance is auditable.

### E. Eligibility: repair only what the evidence supports (v0-F15)

A certificate is emitted for **every** component with two or more qualifying anchors — certifying
is cheap and honest. Repair is gated separately:

* `split_by_anchor` is planned only for conflicts inside an explicitly supplied **contact scope**
  (a caller-provided list, or a selector with a configured anchor-gap threshold);
* every other conflict is recorded as `certified_unrepaired` with the reason, and is exported so a
  later pass can pick it up;
* the summary reports both counts. On arm0_96 that is 16 repaired / 16 recorded.

This is the difference between a framework that reports what it cannot do and one that quietly
mis-repairs half its cases.

### F. Thresholds: no invented defaults (v0-F5)

`min_share` has **no default**. The CLI requires it explicitly and the config records it. The
documentation states plainly: an absolute floor was measured unsafe (a 1,396-voxel leak, 0.3% of
its anchor, triggered splitting a healthy component); the exploratory runs in this session used
0.02; **that value is not established** and is not a repository default.

### G. Actions: fail-closed, and bound to results (v0-F7, v0-F8, v0-F14)

* Add the missing `annotate` action type (v0-F14).
* `consolidate_same_anchor` requires, from the **executor's own record**: a successfully executed
  `split_by_anchor` over the exact scope, successfully executed `forbid_merge` constraints, and no
  unresolved distinct-anchor conflict on that scope. A missing action fails the precondition — it
  does not pass by absence.
* `forbid_merge` is planned against **symbolic territory references** (`territory(component,
  anchor)`), which the executor resolves against the recorded split result. Resolution is a
  validated binding step, never a silent synthesis of new actions, and symbolic refs are part of
  the hashed plan so replay is stable.

### H. Postconditions that cannot be satisfied vacuously (v0-F9)

"Subset of the input component" is satisfied by deleting everything. Replace with exact accounting
against the parent-component mask and the declared scope:

```
sum(territory voxels) + abstained = parent component voxels within scope
territories contain no voxel from any other input component
voxels outside scope: partition-equivalent to input (not label-equal)
changed-voxel count recorded and equal to the sum of territory reassignments
```

### I. Executable invariants (v0-F10)

Each of the seven gets a concrete definition: invariant 2's dominance tolerance comes from config
(no invented number); invariant 5's evidence is the executor's append-only log, not action order;
invariant 6's no-op means an empty action list *and* an unchanged output hash; invariant 7 compares
the canonical plan JSON with **volatile fields excluded** — timestamps and timings are recorded
outside the hashed region, which v0's "hash the record" would have made self-contradictory.

### J. I/O, export and the adapter (v0-F12, v0-F13)

An explicit `io.py`: bounded blockwise reads reusing the repo's volume abstractions, input artifact
hashing, exact scope-mask persistence, corrected-segmentation output in a decoder-consumable
format, append-only JSONL execution records, and per-action timing plus affected-voxel counts. No
full-volume copies.

The cannot-link adapter must **identify the real consumer format during implementation** and ship
an integration test that round-trips a manifest through it. A canonical manifest no later pass can
read is a failure, not a deliverable.

### K. CLI artifact contracts (v0-F16)

`describe` writes descriptors + anchor totals; `plan` consumes a descriptor set and emits a frozen,
hashed plan; `apply` binds that plan to hashed inputs and refuses on mismatch; `verify` locates the
immutable plan, the execution log, the scope mask and the output segmentation by reference;
`run` chains them, writing every intermediate.

## Files and Areas

| File | Purpose |
|---|---|
| `tests/fixtures/checkpoint/` | **golden artifacts captured pre-lift** |
| `connectomics/decoding/nucleus.py` | pure kernels incl. consolidation |
| `connectomics/decoding/checkpoint/{schema,serialization,registry,actions,verification,engine,io}.py` | framework |
| `connectomics/decoding/checkpoint/operators/{base,nucleus_anchor}.py` | the one operator |
| `connectomics/runtime/checkpoint.py` | entry functions + constraint adapter |
| `scripts/checkpoint.py` | thin shim |
| `dev/zebrafinch/*.py` | rewired last, CLI unchanged |
| `docs/segmentation_checkpoint_dsl.md` | all seven explanations, the four-way distinction, every listed future descriptor, one complete pass spec, one complete emitted plan (v0-F15) |
| `tests/unit/test_checkpoint_*.py` | the twelve tests, strengthened below |

## Verification Plan

**V1 — golden capture (must precede the lift).** Fixtures produced by the untouched scripts,
checked in with exact values and hashes.

**V2 — the twelve tests**, with #3 and #12 rewritten:

* **#3** (v0-F3) now asserts against a hand-computed seeded-watershed oracle on a small array:
  exact expected territories, full voxel conservation (assigned + abstained = component), no
  voxels drawn from other components, and that non-anchor mass is *meaningfully* assigned rather
  than discarded. A checkerboard or carve-away implementation must fail.
* **#12** compares lifted kernels to the checked-in golden artifacts, never to live `dev/` code.

**V3 — repo guardrails.** `test_v3_guardrails.py`, `test_v2_boundaries.py`,
`test_public_api_snapshot.py` pass **unmodified**.

**V4 — the measured gate (v0-F2).** On the `worst3` crop, assert:

```
misplaced mask voxels   115,479 -> 5,071    (+/- a documented tolerance)
cross-nucleus shared    1 -> 0
dominance 275/319/373   0.8957/0.9104/0.8988 -> >= 0.99
```

and on arm0_96 that eligibility reports **16 repaired / 16 certified-unrepaired**. Without this the
framework could be the fourth mechanism to measure 0.0000 while passing every unit test.

**V5 — adapter round trip.** Exported cannot-links are read back by the real consumer.

## Risks and Questions

* **[major] The golden capture is now on the critical path** and must happen before any refactor.
  If it slips, test #12 silently becomes a tautology.
* **[major] Contact-scope selection is a real dependency.** The 16/16 split came from an ad-hoc
  nearest-anchor-gap computation in this session. Either it is lifted and tested, or the caller
  supplies scopes; the operator must not re-derive it informally.
* **[major] V4 runs on real data** and needs the crop artifacts present. If unavailable in the
  implementation environment, report NOT RUN — do not substitute a synthetic proxy and call the
  gate met.
* **[minor] The `worst3` numbers came from arm1_ft.** The tolerance must accommodate the pooling
  factor and any config difference, or the gate will fail for reasons unrelated to correctness.

**Answers to the reviewer's questions.** (1) Nothing establishes the threshold; the CLI will
require it explicitly, with no default. (2) Contact scopes come from a caller-supplied list or a
configured gap threshold, never from an implicit rule inside the operator. (3) The oracle is
`tests/fixtures/checkpoint/`, captured from the untouched scripts before the lift and checked in.

## Changes Since Previous Plan Version

All sixteen findings addressed; four changed the design.

* **v0-F1** golden fixtures captured pre-lift and checked in; tests never depend on `dev/`.
* **v0-F2** V4 asserts the measured `worst3` numbers as the acceptance gate.
* **v0-F3** test #3 gains an oracle, conservation and abstention accounting.
* **v0-F4** the lift now specifies cost, pooling, upsampling, markers, connectivity, tie-breaking,
  anisotropy, unreachable-voxel handling, and the omitted consolidation kernel.
* **v0-F5** `min_share` has no default; the 0.02 used this session is documented as unestablished.
* **v0-F6** whole-anchor totals become a hashed input artifact; ROI-local fractions are rejected.
* **v0-F7** consolidation is fail-closed on executor-recorded evidence.
* **v0-F8** `forbid_merge` uses symbolic territory refs resolved by a validated binding step.
* **v0-F9** subset replaced by exact conservation/abstention accounting.
* **v0-F10** each invariant given an executable definition; volatile fields excluded from hashes.
* **v0-F11** CLI enters through `runtime`; adapter lives there too.
* **v0-F12** an explicit `io.py` with blockwise reads, hashing, scope masks, append-only records.
* **v0-F13** adapter format identified during implementation plus a round-trip test.
* **v0-F14** `annotate` added; descriptor fields for anchor IDs, counts and scope specified.
* **v0-F15** repair gated to contact scopes; bridges recorded as `certified_unrepaired`.
* **v0-F16** per-command artifact contracts defined.
* **v0-F17** the subpackage stays private; the public-API snapshot is expected to be unchanged.
=========================== END artifacts/plan_v1.md ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
