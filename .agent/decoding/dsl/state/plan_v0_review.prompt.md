You are reviewing an implementation plan for a CCC run. You are the CODER: you will implement it,
so review it for executability.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge the plan on:
1. Executability. The task spec is long and prescriptive; does the plan actually cover it, or does
   it silently drop required pieces (the five CLI modes, all twelve tests, the four action types,
   the seven verification invariants, the design document contents)?
2. Placement. The plan puts the package inside `connectomics/decoding/checkpoint/` rather than a
   new top-level `checkpoint/`, arguing that `dev/` is gitignored (so nothing there is testable)
   and that the repo enforces dependency direction with static AST tests. Is that sound, and does
   it create a circular or illegal dependency?
3. The numerical lift. Reusable code currently lives in gitignored `dev/zebrafinch/` scripts that
   produced every measured number in this line of work. Is the lift plan safe? Is test #12 as
   specified sufficient to catch a regression, given the plan itself notes it must pin behaviour
   BEFORE the refactor?
4. Whether the safety properties are actually enforceable as written -- particularly the executor
   refusing `consolidate_same_anchor` while an unexecuted `split_by_anchor` covers an overlapping
   scope, and the "output territory is a subset of the input component" postcondition.
5. Anything that would let this framework pass its tests while being, in substance, a repackaging
   of mechanisms that already measured exactly 0.0000 on real data.

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

=========================== BEGIN artifacts/plan_v0.md ===========================
# Plan v0

## Summary

Build the checkpoint DSL as `connectomics/decoding/checkpoint/` — inside the existing `decoding`
package rather than a new top-level one — with the numerical kernels lifted into
`connectomics/decoding/nucleus.py` and the `dev/zebrafinch/` scripts rewired to import them so
their CLI behaviour is preserved. That placement is not cosmetic: `dev/` is **gitignored**, so no
test can cover code that stays there, and the repo enforces a dependency direction
(`decoding → {config, data, utils}`) with static AST tests, which a new top-level package would
have to be added to explicitly.

Four layers, no shortcuts between them: immutable `Descriptor` records; a `Certificate` that is a
hard assertion from a trusted anchor; an ordered `ActionSpec` list; and an `Executor` that
validates preconditions before touching anything. The policy that maps descriptors to actions is
one deterministic function with a single condition (`anchor.distinct_count >= 2`), isolated so a
future agent can replace it while still being unable to mutate a segmentation.

The discriminating test is #3 — **one atomic supervoxel overlapping two nuclei**. Region-level edge
rejection provably cannot fix that case (measured: one supervoxel carried 100.0% of the
contamination), so if `split_by_anchor` is not genuinely voxel-level, this framework is a
re-packaging of three mechanisms that already measured exactly 0.0000.

## Scope

In scope: the schema, engine, registry, actions, verification, serialization, CLI, the
`nucleus_anchor` operator, the numerical lift, the twelve tests, and the design document.

Out of scope, per the task's non-goals: morphology/junction/artifact detectors, glia
classification, semantic labels in the core schema, any agent loop, changes to the global
agglomeration algorithm beyond a constraint adapter, and the neurite-bridge error class.

## Proposed Changes

### A. Placement and the guardrail question

```
connectomics/decoding/nucleus.py            numerical kernels, arrays in / arrays out, no I/O
connectomics/decoding/checkpoint/
    schema.py          EntityRef, Descriptor, Certificate, ActionSpec, CheckpointPlan,
                       CheckpointResult; stdlib dataclasses + explicit validation
    serialization.py   canonical JSON, content-derived stable IDs, JSONL append-only records
    registry.py        operator registration by name (function-call style, matching
                       `decoding/registry.py`, NOT a decorator — see CLAUDE.md)
    actions.py         typed action schemas + the executor
    verification.py    the seven invariants of task section D
    engine.py          sequential pipeline runner over a pass specification
    operators/base.py  the operator protocol
    operators/nucleus_anchor.py
scripts/checkpoint.py                       thin CLI: describe | plan | apply | verify | run
docs/segmentation_checkpoint_dsl.md
tests/unit/test_checkpoint_*.py
```

Before writing code, run `tests/unit/test_v3_guardrails.py` and
`tests/unit/test_public_api_snapshot.py` to see what a new subpackage trips. Expect the public-API
snapshot to need an explicit entry; do **not** widen the dependency direction to make an import
work — if `checkpoint` needs something outside `{config, data, utils}`, that is a design signal to
move the dependency, not to edit the guardrail.

### B. The numerical lift (the part that makes tests possible)

`dev/zebrafinch/nucleus_competitive_split.py` and `flood_contact_units.py` mix I/O, pooling and
numerics. Lift exactly these pure functions into `connectomics/decoding/nucleus.py`:

| function | contract |
|---|---|
| `anchor_histogram(labels, anchors)` | `{component: {anchor: voxels}}`, one pass |
| `qualifying_anchors(hist, min_share)` | anchors above a share of **their own total mass** |
| `conflicting_components(hist, min_share)` | components with >= 2 qualifying anchors |
| `competitive_territories(cost, markers, mask)` | seeded watershed; returns territory labels |
| `shared_anchor_mass(labels, anchors, tol)` | the strict, any-mass contamination test |
| `pool_min / pool_max` | the pooling used to build cost and mask |

All take arrays and return arrays or plain dicts — no CloudVolume, no h5py, no paths. The dev
scripts then import them and keep their current CLI surface; a regression test asserts the lifted
functions reproduce the scripts' current numbers on a small fixture (task test #12).

`min_share` defaults to **0.02** and this is a documented, measured default, not a new invention:
an absolute floor previously let a 1,396-voxel leak (0.3% of that anchor) trigger splitting a
healthy component in half.

### C. Schema and identity

Dataclasses plus a `validate()` on each type — the repo uses OmegaConf/dataclasses, and pydantic
is not a dependency, so introducing it for this alone is out.

Stable IDs are `sha256` over the record's canonical JSON with the id field excluded, truncated to
16 hex chars. That makes IDs content-derived, so determinism (test #10) is structural rather than
a property to be tested for separately. `schema_version` is an explicit field on every serialized
record.

Descriptor keys are namespaced strings, not an enum. This task computes only:

```
anchor.distinct_count        anchor.overlap_fraction        component.volume_voxels
```

`anchor.overlap_fraction` is per-anchor and is a fraction of **that anchor's** mass.

### D. Certificate and policy

One certificate: `distinct_anchor_identity_conflict`, `strength: hard`, emitted iff
`anchor.distinct_count >= 2`. The condition is expressed in the minimal safe syntax
(`field / operator / value`) — no `eval`, no expressions in YAML.

The anchor contract is documented in the schema module, in the task's own words: one nucleus does
not certify that the component is pure, and **absence of a nucleus asserts nothing**. The second
half is what keeps this from silently becoming a glia classifier.

### E. Actions, in the order the task fixes

`split_by_anchor` → `forbid_merge` → `consolidate_same_anchor` → `rebuild_local_rag`.

The ordering is load-bearing and must be enforced by the executor, not just emitted in order:
consolidating before the split would union every component holding anchor N — and while those
components are fused through the conflicting one, that re-joins exactly what the split separated.
The executor refuses `consolidate_same_anchor` if the plan contains an unexecuted
`split_by_anchor` for an overlapping scope.

`split_by_anchor` safety properties, checked as postconditions:
* every output territory is a **subset** of the input component (never absorbs unrelated voxels);
* the affected voxel scope is recorded exactly;
* voxels the flood does not reach stay unassigned rather than being forced to a winner.

`forbid_merge` writes a canonical framework-level constraint manifest, plus an adapter to whatever
the global decoder consumes. Keep the canonical form independent of ABISS so the manifest survives
a backend change.

### F. What must NOT be claimed

My own design note (`paper_NM_decoding/context/claude_code_decoding-watershed.md` §7) argues that
restricting the flood to the sub-graph greedy would merge makes post-hoc equivalent to in-pipeline.
The task explicitly forbids claiming that equivalence. **The task wins.** State the restriction as
a safety property — the operator never merges anything the agglomeration did not — and drop the
equivalence claim from the code, the docs, and (separately, outside this run) that design note.

## Files and Areas

| File | Purpose |
|---|---|
| `connectomics/decoding/nucleus.py` | lifted numerical kernels, no I/O |
| `connectomics/decoding/checkpoint/*.py` | schema, serialization, registry, actions, verification, engine |
| `connectomics/decoding/checkpoint/operators/nucleus_anchor.py` | the one operator |
| `scripts/checkpoint.py` | thin CLI |
| `dev/zebrafinch/nucleus_competitive_split.py`, `flood_contact_units.py` | rewired to import the kernels; CLI unchanged |
| `docs/segmentation_checkpoint_dsl.md` | design document |
| `tests/unit/test_checkpoint_*.py` | the twelve tests |

**Do not touch** the 57 files already dirty in the working tree (`state/run_start.status`); they
are unrelated in-flight work. Do not create git commits.

## Verification Plan

Twelve tests on small synthetic 3D arrays, run with `python -m pytest tests/unit/test_checkpoint_*.py -q`:

1. one component / one nucleus → descriptors emitted, no certificate, no split;
2. one component / two nuclei → hard certificate, split, pairwise cannot-link, no output component
   holds both anchors;
3. **one atomic supervoxel spanning two nuclei** → region-level edge rejection is asserted
   insufficient, and a voxel-level split is planned and applied. *This is the test that
   distinguishes the framework from the three mechanisms that measured 0.0000;*
4. multiple fragments on one nucleus → consolidation planned only after exclusion, and no
   distinct-anchor territories joined;
5. minority contamination → detected via relative anchor mass; the verifier must not rely on
   dominant-segment sharing (that is precisely how `nucleus_fusion_audit.py` scored 0 on a run
   that had the bug);
6. sub-threshold overlap → no certificate;
7. scope safety → labels outside the ROI partition-equivalent to input;
8. idempotence → second run is a no-op;
9. serialization round trip → all typed fields preserved;
10. determinism → identical inputs give identical plans and IDs;
11. failed precondition → executor refuses when the input or config hash no longer matches;
12. regression → the dev scripts' current numbers are reproduced by the lifted kernels.

Plus: `python -m pytest tests/unit/test_v3_guardrails.py tests/unit/test_v2_boundaries.py
tests/unit/test_public_api_snapshot.py -q` must pass, since a new subpackage is exactly what those
guard.

Then a real dry run and full run on the `worst3` crop, with the exact commands in the final
report — the acceptance criterion asks for them explicitly.

## Risks and Questions

* **[major] The `dev/` lift is where regressions hide.** Those scripts produced the numbers this
  whole line of work rests on. Test #12 must pin the current output on a fixture *before* the
  refactor, not after, or it will happily pin the bug.
* **[major] "Do not claim equivalence" vs. my own design note.** Resolved above in favour of the
  task, but it means the note is now inconsistent and should be corrected separately.
* **[major] The V3 guardrails may reject the placement.** Better to discover that in the first
  hour than after the package is written; the plan puts that check first.
* **[minor] Scope-safety test #7 needs a partition comparison, not a label comparison** — the
  split renumbers, so equality of labels is the wrong assertion.
* **[minor] `rebuild_local_rag`** has no obvious repository utility to call. If none exists,
  implement it as an explicit invalidation record rather than inventing a graph rebuild.
* **Question:** the task lists `hold_edge` / `release_edge` as schema placeholders. I read that as
  types defined and rejected at execution time, so a plan containing one fails loudly rather than
  being silently ignored. Flag if you read it differently.

## Changes Since Previous Plan Version

Initial plan.
=========================== END plan_v0.md ===========================

Repository conventions you must respect are in CLAUDE.md; the relevant extracts:
## Architecture Philosophy

The codebase follows a clean separation of concerns:
- **PyTorch Lightning**: Orchestration layer (training loop, distributed training, mixed precision, callbacks, logging)
- **MONAI**: Domain toolkit (medical image models, transforms, losses, metrics)
- **Hydra/OmegaConf**: Modern configuration management (type-safe, composable configs)

**Key Principle:** Lightning is the outer shell, MONAI is the inner toolbox. No reimplementation of training loops or domain-specific tools.

### V2/V3 Architecture Contract

The codebase enforces an explicit contract from the v2/v3 refactor:

1. **One canonical owner per concept.** No backward-compatibility shims, no facade re-exports, no duplicate import paths.
2. **Strict config.** Unknown top-level keys **raise** at load time. Removed fields raise. `getattr(cfg.x, "y", default)` ghost reads on undeclared fields are forbidden.
3. **Stages are separate.** Pipeline = `train → infer → decode → evaluate → tune`. Each stage has its own package and its own entry function. Combined test-mode is a thin wrapper that calls stage APIs in sequence.
4. **Dependency direction:** `config → utils → data → models → metrics`; `training → {config, data, models, metrics}`; `inference → {config, data, models}`; `decoding → {config, data, utils}`; `evaluation → {config, data, metrics}`; `runtime → {config, training, inference, decoding, evaluation}`. Static AST tests in `tests/unit/test_v3_guardrails.py` enforce this.
5. **Public API is explicit and small.** `tests/unit/test_public_api_snapshot.py` asserts exact `__all__` membership.

## Agent Design Principles

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
