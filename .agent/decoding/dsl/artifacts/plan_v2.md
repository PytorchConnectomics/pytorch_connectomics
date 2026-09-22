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
