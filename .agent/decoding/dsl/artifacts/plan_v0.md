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
