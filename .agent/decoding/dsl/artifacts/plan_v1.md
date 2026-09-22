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
