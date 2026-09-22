You are the CODER for a CCC run. Implement plan_v2 (below). It is the TERMINAL plan and is
authoritative. If something in it is wrong while implementing, implement the correct thing and say
so explicitly in "Risks and Unknowns" -- do not silently deviate.

You reviewed plan_v0 (16 majors) and plan_v1 (18 majors). plan_v2 accepts all of them and makes
every deferred decision. Your two sharpest findings are now design constraints:
  * a bounded ROI can leave the anchors joined through the untouched exterior -- so the operator
    must PROVE containment or downgrade its claim to `local_only`;
  * there is no existing cannot-link consumer format, so export is documented as write-only.

## Hard constraints

1. Repo root: /projects/weilab/weidf/lib/pytorch_connectomics (baseline 6ad67866).
   The working tree already has 57 modified files of UNRELATED in-flight work
   (`state/run_start.status`). Do not touch, revert or commit them. Create no git commits.
2. `dev/` is gitignored. Nothing under it can be tested in a clean checkout -- that is why the
   plan promotes three thin CLIs into `scripts/` and lifts kernels into the tracked tree. Leave the
   `dev/` copies working and unmodified in observable behaviour.
3. Capture the golden fixtures BEFORE refactoring anything. If you refactor first, test #12 becomes
   a tautology comparing the new code with itself.
4. Do not claim post-hoc splitting is equivalent to online constrained agglomeration.
5. Do not add semantic labels (`is_glia`, `is_neuron`, `cell_type`) to the core schema.
6. No `eval`, no expressions in YAML, no new heavyweight dependencies (pydantic is NOT a repo dep;
   use dataclasses + explicit validation).
7. Environment: `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u`.
   HDF5 on /projects needs HDF5_USE_FILE_LOCKING=FALSE.
8. Do not claim a command passed unless you ran it and saw the output. NOT RUN is the correct
   report for anything you did not run -- EXCEPT V4, where the plan states NOT RUN is a failure of
   the golden-capture step, not a licence to pass on synthetic evidence.

## What to produce

Implement the plan, run the tests, then write, at exactly this path:

    /projects/weilab/weidf/lib/pytorch_connectomics/.agent/decoding/dsl/artifacts/code_v0.md

with EXACTLY these level-2 sections, in order, each exactly once:

    # Code v0
    ## Overview
    ## What Changed
    ## Implementation Details
    ## Files Changed
    ## Git Baseline
    ## Verification
    ## Review Focus
    ## Risks and Unknowns
    ## Changes Since Previous Code Version

"## Files Changed" needs a table headed exactly `| File | Purpose |` then `|---|---|`.
"## Git Baseline" needs `run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd` and
`current_head: <git rev-parse HEAD>`.
"## Changes Since Previous Code Version" must contain exactly: Initial implementation.
"## Verification" must give, per check, the command run and its actual result, or NOT RUN.
The final artifact must also list the exact dry-run and full-run commands for the `worst3` crop.

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

=========================== BEGIN artifacts/plan_v2.md ===========================
# Plan v2

## Summary

Terminal plan. Every deferred decision is made here; nothing is left to implementation taste.

Five findings changed the design again, and two of them are corrections to claims I made rather
than gaps to fill:

* **Invariant 1 contradicted the design.** Certified-but-unrepaired bridge components still hold
  two anchors, so a global "no output component holds two anchors" invariant fails by construction.
  The invariant is now scoped to *repaired* components, with unrepaired ones enumerated explicitly.
* **A bounded ROI can leave the anchors joined outside it.** Voxels outside the ROI keep the
  original id, so the keeper territory stays connected to everything beyond the box — including
  any part of the other anchor's soma that extends past the margin. This affects the runs already
  executed in this session, not just the plan. The operator must now *prove* containment or
  downgrade its claim.
* **There is no existing cannot-link consumer format.** ABISS has `nuc_cuts.data` as an *output*
  and `NUC_PATH` as a *mask input*; nothing consumes a constraint list. v1 promised an adapter to
  "whatever the global decoder consumes" — that consumer does not exist. Say so.
* **`dev/` CLIs cannot satisfy a regression contract** they are invisible to in a clean checkout.
* **Contact eligibility is now a checked-in artifact**, not a selector the operator re-derives.

## Scope

Unchanged in what is built (the DSL, one `nucleus_anchor` operator, the CLI, twelve tests, the
design document). Changed in what is *claimed*: separation is asserted only for components whose
repair ROI provably contains them, and constraint export is documented as write-only because no
consumer format exists.

## Decisions (previously deferred)

| question | decision |
|---|---|
| contact eligibility | **caller-supplied immutable artifact** `contact_scopes.json`: `{seg_id, anchor_ids, provenance, gap_um}`. This session's nearest-anchor-gap computation produced it; it is checked in as data, not re-derived in the operator. |
| `min_share` | **no default**; required CLI argument, recorded in the config hash |
| cannot-link consumer | **none exists.** Ship the canonical manifest only. No adapter, no adapter test. Document that export is currently write-only and name what a consumer would have to accept. |
| connectivity | **6-connectivity**, asserted in a test |
| tie-breaking | lowest anchor id wins; asserted with a deliberately tied fixture |
| anchor totals | computed by `describe`, written as a **separate hashed artifact**, then a *required input* to `plan`/`apply`/`verify`. One producer, one consumer, no ambiguity |
| dev CLI regression | the three needed CLIs are **promoted** to `scripts/` as thin wrappers over the lifted kernels. `dev/` copies are left untouched. The contract becomes testable |

## Proposed Changes

### A. Scope containment, and the claim it licenses (v1-F18)

Before planning a split, compute whether the repair ROI fully contains the parent component.

* **contained** → the plan may assert separation, and invariant 1 applies;
* **not contained** → emit `scope.containment = partial` on the descriptor, still split, but record
  `separation_claim = local_only` and **exclude the component from invariant 1**, because the
  anchors may remain joined through the untouched exterior.

A verifier check confirms it directly: no territory that received a *new* id may touch the ROI
boundary face where the parent component continues. This is checkable and cheap.

*This applies retroactively to the 16 whole-volume units already running; their "separated" result
is a local claim until containment is confirmed.*

### B. Invariant 1, corrected (v1-F1)

> For every component in the **repaired set**, no output component contains more than one
> qualifying anchor. Components in the `certified_unrepaired` set are enumerated in the result with
> their reason and are excluded by construction, not by omission.

The result carries both sets explicitly so a reader cannot mistake silence for success.

### C. V4 becomes a mandatory, non-vacuous, identity-checked gate (v1-F2, F3, F4, F7)

* the `worst3` sub-ROI is **checked in** (`tests/fixtures/checkpoint/worst3_roi/`), so V4 has no
  "data unavailable" escape and the section B / V4 inconsistency disappears;
* **abstention is capped**: `abstained <= 5%` of the parent component, and each territory must
  receive `>= 1%` — an implementation cannot abstain its way to a passing contamination number;
* **exact identities asserted**, not counts: the repaired set must equal the checked-in contact
  scope list element-for-element, and the unrepaired set must equal the 16 bridge ids;
* the measured numbers stay as the gate: 115,479 -> 5,071, 1 -> 0, dominance >= 0.99.

### D. Golden capture as a reviewable artifact, not a procedure (v1-F6)

A standalone `tools/capture_checkpoint_goldens.py` with its own docstring stating it must run
against unmodified sources, plus a `MANIFEST.sha256` over both the inputs and the emitted goldens.
The test asserts the manifest matches. Ordering becomes auditable rather than a promise.

### E. The lift, with values not categories (v1-F11)

Connectivity 6; tie-break lowest-anchor-id; channels: `min` over the first three affinity channels
after convention/sigmoid restore, with the channel indices explicit in config; pooling `min` for
cost and `max` for masks; nearest upsampling with the +/- factor error documented; unreachable
voxels abstain and are counted. Each is pinned by a fixture, not prose.

### F. Constraint lifecycle (v1-F12)

One `(component, anchor)` can yield several pieces. So `forbid_merge` binds to the **anchor
territory set**, not a single piece id, and consolidation is what collapses a set to one id.
Ordering: split -> record pieces -> consolidate within anchor -> *then* install `forbid_merge`
between the consolidated territories. v1 installed constraints before consolidation, which left
them pointing at ids that consolidation then removed.

### G. Determinism means execution, not planning (v1-F13)

Invariant 7 compares the **sha256 of the output segmentation** across two independent `apply` runs
from the same frozen plan, in addition to plan-JSON equality. Volatile fields stay outside both
hashes.

### H. `annotate`, fully typed (v1-F17)

`annotate(target: EntityRef, key: str, value: JSONScalar, note: str|None)`. Execution appends to
the record and mutates nothing. Serialized identically to other actions; rejected if `key` is not
namespaced.

### I. I/O with a test that can fail (v1-F15)

A test wraps the volume reader and **asserts no single read exceeds a configured byte budget**, so
a whole-volume read fails the suite rather than merely being discouraged. Per-action timing and
affected-voxel counts are asserted present in the result record.

### J. Test #3 through the serialized boundary (v1-F9, F10)

Test #3 runs `describe -> plan -> apply` through **serialized artifacts on disk**, not in-process
calls, and asserts exact expected territory and abstention masks plus a fixed nonzero count of
assigned non-anchor voxels. "Meaningfully assigned" is replaced by that number.

## Files and Areas

As v1, plus: `tools/capture_checkpoint_goldens.py`; `tests/fixtures/checkpoint/{worst3_roi,
contact_scopes.json,MANIFEST.sha256}`; `scripts/{nucleus_split,nucleus_anchor_merge,
nucleus_contamination}.py` (promoted thin CLIs). No adapter module — there is no consumer.

## Verification Plan

V1 golden capture (auditable manifest) → V2 the twelve tests with #3 and #12 as above → V3 repo
guardrails unmodified → V4 the mandatory measured gate with identity and abstention checks → V5
containment check → V6 byte-budget I/O test → V7 two-apply output-hash determinism.

**No `NOT RUN` is acceptable for V4.** If the checked-in ROI is missing, that is a failure of the
golden-capture step, not a licence to pass on synthetic evidence.

## Risks and Questions

* **[major] Containment may fail on real data.** If most of the 16 units are `partial`, the
  headline claim weakens from "separated" to "locally separated", and the honest fix is a larger
  ROI or a whole-component repair — at real cost. Better to discover it in the verifier than in a
  paper.
* **[major] No cannot-link consumer exists**, so the export is currently write-only. The framework
  is still worth building — the manifest is the contract a consumer will implement against — but
  the acceptance criterion "durable and exportable to the later global pass" is only half
  demonstrable today, and the docs must say which half.
* **[major] Promoting the dev CLIs is a real change** to a tracked surface. Keep them thin, keep
  `dev/` intact, and do not alter observable behaviour.
* **[minor] The checked-in `worst3` ROI adds binary fixtures to the repo.** Keep it minimal —
  the smallest ROI that still reproduces the contamination numbers within tolerance.

## Changes Since Previous Plan Version

All eighteen findings addressed; five changed the design.

* **v1-F1** invariant 1 scoped to the repaired set; unrepaired enumerated.
* **v1-F2, F7** `worst3` ROI checked in; V4 mandatory; the fixture inconsistency removed.
* **v1-F3** abstention capped and per-territory minimum imposed.
* **v1-F4** exact conflict identities asserted, not 16/16 counts.
* **v1-F5** contact eligibility is a checked-in artifact.
* **v1-F6** golden capture is a reviewable tool plus a hash manifest.
* **v1-F8** dev CLIs promoted to `scripts/` so the regression contract is testable.
* **v1-F9, F10** test #3 goes through serialized artifacts with exact masks and counts.
* **v1-F11** connectivity, tie-break and channel handling given values.
* **v1-F12** constraints bind to anchor territory sets and install after consolidation.
* **v1-F13** determinism compares output-segmentation hashes across two applies.
* **v1-F14** no adapter; the manifest is canonical and the absence of a consumer is documented.
* **v1-F15** byte-budget I/O test that can fail.
* **v1-F16** anchor totals: produced by `describe`, required input thereafter.
* **v1-F17** `annotate` fully typed.
* **v1-F18** scope containment computed, asserted, and allowed to downgrade the claim.
=========================== END artifacts/plan_v2.md ===========================

=========================== BEGIN artifacts/plan_v1_review.md ===========================
# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v1_review.review.raw.md`. Eighteen major findings, none softened.

v1 fixed v0's circular oracle, ROI-local anchor fractions, bridge over-repair and invented
threshold, but the tightened plan exposed deeper contract holes: an invariant that the design
contradicts by construction, a measured gate with a `NOT RUN` escape and a vacuous-abstention
loophole, a constraint lifecycle that installs cannot-links against ids consolidation later
removes, and a bounded ROI that can leave the anchors joined through the untouched exterior.

## Findings

- [major] Certified-but-unrepaired bridge conflicts contradict invariant 1. Those components will still contain multiple qualifying anchors, so verification must either fail or silently exempt them. The plan must define partial-success semantics: repaired scopes satisfy the invariant, unrepaired certificates remain explicitly unresolved, and the checkpoint result cannot report them as verified repairs.

- [major] V4 is not actually a gate if missing data yields `NOT RUN`. That permits completion with only synthetic evidence. The required `worst3` input must be checked in or V4 must be a mandatory external acceptance step whose absence prevents declaring the implementation complete.

- [major] V4 still permits vacuous “repair.” Conservation allows arbitrary abstention, so an implementation could remove or abstain most contaminated voxels and improve sharing/dominance. The gate needs pinned assigned/abstained counts or an exact expected output partition/hash. The misplaced-voxel tolerance must be fixed before implementation, not documented after observing results.

- [major] The arm0_96 gate asserts only `16/16` counts. An implementation could repair the wrong 16 conflicts. It must assert the exact conflict identities/scopes classified as contact versus bridge, plus that no bridge-class scope was mutated.

- [major] Contact eligibility remains an unresolved design choice: caller-supplied scopes or a gap selector. If using supplied scopes, their immutable format and provenance must be specified. If using a selector, its exact distance definition, threshold, units, tie handling, and tests must be fixed. This is a decision required before coding, particularly because V4 depends on it.

- [major] The golden ordering is procedural, not enforceable or auditable as written. A genuine oracle requires a separately reviewable baseline-capture commit or manifest containing untouched script hashes, repository revision, exact commands/configuration, environment/version information, input hashes, and output arrays/hashes. Merely saying “captured before the lift” cannot prove that it was.

- [major] The fixture story is internally inconsistent. Section B says a real `worst3` sub-ROI is checked in, while V4 says the required crop may be unavailable. It must identify whether the checked-in real fixture is sufficient to reproduce all V4 numbers; otherwise the golden regression and measured gate are different datasets and both contracts need explicit boundaries.

- [major] Rewiring `dev/zebrafinch/*.py` “last” cannot satisfy the existing-CLI regression contract in a clean checkout because those files are gitignored. The plan must either leave those scripts untouched and test only the lifted behavior, or define a tracked mechanism that preserves and tests their CLI behavior.

- [major] Test #3 has a strong numerical oracle, but its integration boundary is ambiguous. It must exercise serialized describe → plan → apply through the real executor, explicitly provide one atomic fragment/supervoxel ID, assert that it becomes multiple anchor-driven territories, and compare exact assigned and abstained masks. A kernel-only test would still allow the production executor to perform no repair.

- [major] “Meaningfully assigned” is not executable. Replace it with exact expected territory and abstention masks, including a fixed nonzero count of non-anchor component voxels assigned to each expected territory.

- [major] The numerical lift remains underspecified despite being called fully specified. Connectivity has no selected value, tie-breaking has no rule, channel/store conventions are conditional, pooling origin and boundary padding are unspecified, and upsampling alignment/cropping is undefined. These choices affect exact territories and must be pinned before golden capture and coding.

- [major] Symbolic constraint binding has a lifecycle hole. One `(component, anchor)` may produce multiple pieces; `forbid_merge` is installed before consolidation; consolidation and RAG rebuilding may then change IDs. The plan must define stable territory identity, pair expansion, and rebinding so the final decoder-consumable constraints still reference every final distinct-anchor territory.

- [major] Invariant 7 is only nominally addressed. Comparing canonical plan JSON proves deterministic planning, not deterministic execution. Repeated application from the same frozen inputs must compare output partition, abstention mask, territory bindings, constraint manifest, action outcomes, and graph invalidation/rebuild result, excluding only timing and timestamps.

- [major] The real cannot-link format remains explicitly deferred to implementation. That format determines constraint identity, serialization, and replay semantics, so it must be discovered and selected before finalizing the schema. Similarly, `rebuild_local_rag` still has no chosen backend, executable postcondition, or focused test.

- [major] Scalable I/O is still mostly a file-list promise. There is no test requiring bounded reads or rejecting a whole-volume read, no concrete corrected-volume/delta representation that avoids a full copy, and no atomic output/append-log contract. A fake blockwise store that fails on unbounded access should cover this.

- [major] The anchor-total command contract is inconsistent. Section D treats totals as a required input, while sections D/K say `describe` computes and writes them even though descriptors need them during that same operation. Define a separate streaming precompute artifact or make `describe` explicitly consume an existing totals artifact, with the nucleus-volume hash and reference extent included.

- [major] v0-F14 is only nominally addressed. “Add `annotate`” does not define its typed parameters, execution behavior, or serialization. The representation of per-anchor IDs, local counts, authoritative totals, overlap fractions, and `certified_unrepaired` status also remains unspecified.

- [major] Bounded-ROI repair can leave an exterior bridge through the unchanged portion of the original component. The plan must require full containment of the conflicting parent component or define interface relabeling and global connectivity verification; outside-scope partition equivalence alone does not prevent the repaired territories from remaining connected outside the ROI.

## Questions

1. Will eligibility use a checked-in authoritative contact-scope artifact, or a deterministic gap selector? If the latter, what exact threshold and distance definition?
2. Which exact checked-in data and predeclared tolerance make V4 mandatory and reproducible, including expected abstention?
3. What consumer format and stable identity model will keep cannot-links valid through consolidation and local RAG rebuilding?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v1_review.md ===========================

