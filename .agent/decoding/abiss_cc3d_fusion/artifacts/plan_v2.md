# Plan v2

## Summary

All twelve `plan_v1_review` findings accepted. Three of them changed the design rather than the
wording, and one exposed an actual error in v1's safety invariant.

**The mask problem is resolved by construction, not by a veto (finding 2).** The fusion
partition will be built only from CC3D decoded under the **ABISS keep-mask**. A GT-free
voxelwise mask comparison runs first and, by a rule fixed here with no "material" judgement,
**every chunk with a nonzero mask difference is re-decoded** under `tissue_border_keep_mask_full.zarr`.
Only chunks with a bit-zero difference reuse the running store, where the two foregrounds are
identical by construction. The tier10-masked store from array 2855306 keeps exactly one role:
the historical tier10 reproduction. This costs a measurement before it costs compute, and the
measurement tells us the compute bill before we pay it.

**v1's one-anchor invariant was wrong, and finding 5 is why.** v1 built a candidate graph from
raw claims and rejected any connected component holding ≠1 anchor. That conflates the *candidate*
graph with the *resulting label* graph: assigning fragment `f` to anchor `A` while `B` also
claims `f` does not union `A` and `B` — `B` is untouched — so v1 rejected exactly the cases the
margin rule exists to adjudicate, while the reviewer correctly notes the rejection was
simultaneously unreachable on valid extractor output. v2 replaces it with the rule the task
actually specifies: abstention is a property of the **supporting CC3D component** (a component
overlapping ≥2 established anchors never emits an edge), multi-host fragments are adjudicated
per policy, and the single-anchor property of the resulting partition is a checked invariant
rather than the selection mechanism.

**Everything GT-touching now happens after the freeze (finding 1)**, with one stated exception:
asserting the published baseline `0.4443760423975249`, which yields one bit against a constant
already declared in the manifest. The tier10 human-GT reproduction moves after Stage D, and the
GT-scored version of the mask bridge is deleted — the re-decode decision depends only on GT-free
mask arithmetic.

Unchanged and still load-bearing: substrate is **`arm0_win144`**, honest local control
**0.775832**, threshold 0.70 on **stored** values.

## Scope

As plan_v1, plus: a GT-free mask-difference measurement and conditional re-decode; GT-free
generation of all three candidate families before any evaluator overlay; a dense-store fidelity
gate; and a split of diagnostics into GT-free proposal generation and post-evaluation scoring.

Removed from scope, explicitly rather than silently (finding 4): "incompatible centerlines" and
"glia/lamellar morphology" suspect detectors, which have no frozen definition available here;
the split suspect detector is limited to the two definitions below. Distance-2 affinity
aggregation is removed (finding 7). The GT-scored mask bridge control is removed (finding 1).

## Proposed Changes

Stage order, with the firewall marked:

```text
A   gt_free      identity manifest                                    no GT
C0  gt_free      voxelwise mask difference, all 726 chunks            no GT
C1  gt_free      conditional re-decode under the ABISS keep-mask      no GT
C2  gt_free      dense-store fidelity gate                            no GT
C3  gt_free      evidence extraction + global ABISS table             no GT
C4  gt_free      candidate generation: assign / face / cut families   no GT
C5  gt_free      targeted affinity features                           no GT
D   gt_free      FREEZE policies, assignments, hashes                 no GT
B1  evaluation   baseline constant assertion (one bit)                may also run earlier
B2  evaluation   ten-chunk tier10 reproduction of 0.775832            after D
E   evaluation   audit rows, directional oracles, Gate A, frozen NERL, funnel
F   evaluation   diagnostics, suspect-detector precision, composition, visual probes
G   evaluation   results.md written last
```

### Stage A — identity manifest (findings 7, 12)

As plan_v1: all 726 chunk files enumerated individually (HDF5 headers only) with shape, dtype,
channel count, size, mtime and windowed sha256; tag-resolution evidence; ABISS `info` verbatim
with shape, dtype, origin, voxel size, mip list, decode parameters; both mask identities; the
CC3D recipe and `(chunk_ordinal<<32)|local_label` convention.

Leakage audit (finding 12) is **structural first**: every `gt_free/*` artifact must validate
against a declared JSON schema whose key set is fixed in this plan, with `expected_base_nerl`
the single allowlisted constant, carrying a `declared_constant: true` sibling flag. The keyword
scan runs afterwards as a secondary check, with that one key excepted by path, not by regex.

### Stage C0 — voxelwise mask difference (finding 2)

`stage_c0_maskdiff.py`, GT-free, sharded. For each of the 726 chunks, materialise both masks on
the chunk's native grid: `M_t10` (FFN tissue 18-18-20, 2× XY upsample, out-of-bounds pad
KEEP=1 — the recipe the running store used) and `M_abiss` (`tissue_border_keep_mask_full.zarr`,
native, out-of-bounds pad **0**). Record `n_extra = |M_t10 ∧ ¬M_abiss|`, `n_missing =
|M_abiss ∧ ¬M_t10|`, per-chunk keep fractions, and whether the chunk reaches past x>10664 or
y>10912. Writes `gt_free/mask_difference.json`.

This settles empirically the claim v1 asserted without evidence — that only ring chunks differ.
`M_abiss` is `tissue AND border`, so `M_t10 ⊇ M_abiss` is expected but not assumed; `n_missing`
is measured, and any `n_missing > 0` is a contract violation that stops the run.

### Stage C1 — conditional re-decode (finding 2)

**Fixed rule, no threshold and no judgement:** every chunk with `n_extra > 0` is re-decoded from
affinity under `M_abiss` into `…grid1008_halo72.cc3d_t070_abissmask.chunks`. Chunks with
`n_extra == 0` are reused from the 2855306 store, where the foreground is identical by
construction; that identity is verified by hash for a deterministic sample and by the C2 gate.
The fusion partition is read **exclusively** from the ABISS-mask store. The tier10-masked store
is used only by B2.

Cost is bounded by the C0 measurement and reported before C1 runs: if `n_extra > 0` on most
chunks this is a full re-decode of comparable cost to array 2855306 (~3 h at 16 shards), and
that is the price of a correct fusion partition rather than a corrected-after-the-fact one.

### Stage C2 — dense-store fidelity gate (finding 3)

For 12 deterministically selected chunks (lowest `sha256(chunk_key)`, including `z2_y10_x7` and
four ring chunks), re-decode from affinity + mask in-process and assert the stored partition
equals the fresh one **exactly up to relabeling**: identical foreground mask, and a bijection
between stored and fresh label ids verified by the `(stored, fresh)` pair count equalling both
label counts. Provenance and file count are not accepted as evidence of contents.

### Stage C3 — evidence extraction (findings 3, 6)

Reads, per chunk: the ABISS-mask CC3D labels, the ABISS crop, **both masks explicitly** (so
`mask_removed` is distinguishable from `below_threshold` — a zero label alone cannot tell them
apart), and the nucleus volume. Emits per component: namespaced id, voxel count, bbox, centroid,
face touches, ring/mask-boundary flags, every overlapping nonzero ABISS label with counts and
fractions, nucleus contact, and compact face slabs. Atomic tmp+rename per shard with a `.done`
record; reduction requires 726/726 and refuses duplicates.

**Nucleus semantics, predeclared** (finding 6): `yl_cb_80nm_neuron_v2.h5` is read at its native
80 nm grid; a component voxel at affinity-grid `(z,y,x)` maps by
`floor(z*20/80), floor(y*9/80), floor(x*9/80)` — nearest-neighbour, no interpolation; a component
`touches_nucleus` when ≥1000 of its voxels map to a nonzero instance, and the set of instance
ids is recorded per component. ABISS-side nucleus counts are computed the same way per ABISS
label, and are never inherited from the earlier nucleus audit.

Global table `gt_free/abiss_segment_stats.npz`: voxels, bbox, chunk span, face touch, ring
contact, nucleus instance set, multi-nucleus flag.

**Band classification, predeclared with closed semantics** (finding 6). Intervals are half-open
`[lo, hi)`; a label is an **anchor** iff `voxels >= A and diag >= D`; **mid fragment** iff
`M_lo <= voxels < M_hi and not anchor`; **dust** iff `voxels < M_lo`; and
**large_low_extent** iff `voxels >= A and diag < D` — a fifth, explicitly non-assignable class
that v1 left unclassified. Dust is **diagnostic-only and never assignable as a fragment**.

| setting | A (vox) | D (bbox diag) | M_lo | M_hi |
|---|---:|---|---:|---:|
| tight | 3×10⁶ | 10 µm | 3×10⁴ | 3×10⁶ |
| central | 1×10⁶ | 5 µm | 1×10⁴ | 1×10⁶ |
| loose | 3×10⁵ | 2.5 µm | 3×10³ | 3×10⁵ |

The volume arithmetic (a 150 nm-radius, 10 µm cylinder ≈ 4.36×10⁵ voxels at 9×9×20 nm) is
presented as a **reference-caliber interpretation** of the bands, not as a physical validation
of the bbox thresholds or the tight/loose multipliers (finding 8).

### Stage C4 — GT-free candidate generation, all three families (finding 4)

All three families exist before any evaluator overlay, so Gate A can score all three.

**Assign family.** An edge `(fragment f, anchor a, support s)` is emitted only when the
supporting CC3D component `k` overlaps **exactly one** established anchor — a `multi_anchor`
component never emits an edge, in any policy. `s = |f ∩ k| / |f|`.

**Face family.** For each shared chunk face, match components across the face by IoU on the
shared plane. An edge is emitted only on **mutual best match** with `IoU >= 0.50` and a margin
`>= 0.15` over the runner-up on both sides. Deduplicated deterministically by sorted
`(chunk_ordinal, cc3d_id)` pair.

**Cut family.** Suspects are defined by a frozen GT-free detector with exactly two rules: an
ABISS label carrying **≥2 distinct nucleus instances**, or **voxels ≥ 3×10⁷**. A cut candidate
is a CC3D boundary inside a suspect that separates its two nucleus claims. The undefined
"incompatible centerlines" and "glia/lamellar" detectors are dropped from this run.

**Gate A, frozen** (finding 4): pass iff a directional family shows **≥ +0.01**
candidate-restricted NERL oracle gain, **or** the candidate set contains a repair for **≥1 of
the three ERL-visible arm0_96 false merges**. The vague "important break" clause is removed.

### Stage C5 — targeted affinity features (findings 6, 7)

Processes the **union** of candidates required by every policy and every robustness neighbour,
so no variant needs a later read.

**Interface definition, predeclared** (finding 7): for axis `c ∈ {0,1,2}`, where channel `c` is
the edge along the input array's axis `c`, the interface edge set is every 6-adjacent pair
`(v, v+e_c)` with one voxel in the fragment and the other in the anchor; the edge value is
`aff[c][v]` at the lower-index voxel. P4 uses the mean and p10 over **that directed edge set
only**. Distance-2 neighbourhoods and channel `max` are removed. A candidate whose interface
edge set is empty is `no_direct_contact` and abstains in P4+, reported separately. The opposite
index convention `aff[c][v+e_c]` is computed and reported as a diagnostic column so a convention
error is visible rather than silent, and a synthetic test asserts the channel-to-axis mapping
through the same code path the decode uses.

0.75-persistence is recomputed inside the candidate bbox under the ABISS mask.

### Stage D — freeze (findings 1, 6)

Materialises `gt_free/policy_<name>.json` and `gt_free/assignments_<name>.npz` with
`gt_free: true`, `frozen_before_evaluation: true`, `parameters`, `implementation_sha256`, and a
companion `proposal_sha256` recorded before any evaluator run — enforced by the existing
`arm096_evaluate_frozen.py` contract.

Ladder P1–P6 as plan_v1 (P1 none; P2 all one-anchor; P3 support ≥0.50 and ≥1000 component
voxels on the anchor; P4 + interface mean ≥0.70 and p10 ≥0.50; P5 + margin ≥0.20 and
fragment/anchor ratio ≤0.25; P6 + vetoes on nucleus contact, multi-nucleus anchor, ring, mask
boundary), plus `P6+persist`.

**Robustness neighbours** (finding 6): exactly six **one-factor-at-a-time** variants off the
central band — support ∈ {0.40, 0.60}, interface mean ∈ {0.65, 0.75}, margin ∈ {0.15, 0.25}.
Not a Cartesian product, and not crossed with the bands; the three bands are reported
separately at the central thresholds.

**Resolver, corrected** (finding 5). Per policy:

1. R1: an edge exists only if its supporting CC3D component is `one_anchor`. This, not the
   candidate graph, is where multi-anchor evidence is refused.
2. R2: aggregate a fragment's multiple edges to the same anchor by max support.
3. R3: if a fragment retains edges to ≥2 distinct anchors — necessarily from different CC3D
   components — then P2–P4 abstain with reason `multi_host` (they contain no margin gate, so
   they cannot adjudicate), and P5+ accept the best iff `margin >= 0.20`, else abstain.
   Tie-break deterministically by `(support desc, anchor voxels desc, anchor label id asc)`.
4. R4: emit the assignment map fragment → anchor. Anchor and fragment roles are read from the
   immutable pre-assignment table, so an assigned fragment is never an anchor or a relay; one
   hop only.
5. R5: **checked invariant, not selection**: every resulting component contains exactly one
   established anchor; no anchor is a fragment in any accepted edge; no accepted fragment
   appears as an anchor. Violation is a hard failure, not an abstention.

Every step is a per-fragment aggregation plus a global assertion, so the output is
order-independent by construction; the permutation suite proves it on constructed inputs that
are reachable, unlike v1's vacuous cases.

### Stages B, E, F, G — evaluation (findings 1, 8, 9, 11)

`B1` asserts `score_lut(graph, lut, 50) == 0.4443760423975249` within `1e-10`; it is the single
evaluator read permitted before the freeze because it compares against a constant already
declared in Stage A and returns one bit. `B2` — the ten-chunk end-to-end tier10 reproduction of
**0.775832** under the tier10 recipe, with per-chunk agreement against the archived CSV rows —
runs **after** Stage D.

`E` computes the seven partition rows (importing the pure helpers from
`cc3d/compare_cc3d_abiss.py`, its `main()` and defaults untouched), the three directional
oracles over the Stage-C4 populations with GT deciding correctness only, Gate A, and the frozen
policy NERL for P1–P6, `P6+persist`, the six neighbours and the three bands, with the full
evaluation contract per row and the fusion funnel.

`F` scores the diagnostics that must follow evaluation: suspect-detector precision for Gate C,
face-continuation diagnostics, the visual-probe manifests
(`evaluation_gt/visual_probes/`, ≥3 representatives each for correct, incorrect, multi-anchor,
mask-boundary, chunk-boundary, suspected-split, no-benefit), and the composition matrix.

**Composition gate, strengthened** (finding 9). Before scoring against
`frozen_endpoint_merges.npz`: require an exact substrate digest — the ABISS `info` sha256 and
the `reports/arm096_lut/s*.npz` shard digest set must match those the prior map was evaluated
against; label-universe subset and baseline agreement are necessary but explicitly **not
sufficient**, since a different segmentation revision can satisfy both. If that provenance is
not recoverable, the composed row is reported as an **unsafe diagnostic** and never as a frozen
GT-free result. Conflict semantics are stated, not deferred: order 1 (EC then fusion) resolves
EC pairs by union-find, then maps each fragment to the EC-resolved root of its anchor; order 2
(fusion then EC) applies the fusion map, then EC pairs over the resulting labels; the conflict
set is every label appearing as a fragment in our map and in an EC pair, enumerated. If either
order produces a component with ≥2 established anchors, that row is **rejected as
invariant-violating** and reported as such, not scored as a policy.

`G` writes `evaluation_gt/results.md` **last**, after F, answering primary questions 1–6 in
order including on a Gate-A negative path, and cross-tabbing CC3D disagreement against the three
ERL-visible false merges.

## Files and Areas

Under `dev/zebrafinch/abiss_cc3d_fusion/`: `README.md`, `common.py`, `stage_a_identity.py`,
`stage_c0_maskdiff.py`, `stage_c1_redecode.py`, `stage_c2_storegate.py`, `stage_c3_evidence.py`,
`stage_c3_reduce.py`, `stage_c4_candidates.py`, `stage_c5_affinity.py`, `stage_d_freeze.py`,
`stage_b_fidelity.py`, `stage_e_evaluate.py`, `stage_f_diagnostics.py`, `stage_g_report.py`,
`tests/test_*.py`, `sbatch_stage_c0.sh`, `sbatch_stage_c1.sh`, `sbatch_stage_c3.sh`.

Read-only dependencies as plan_v1, plus `tissue_border_keep_mask_full.zarr` and
`yl_cb_80nm_neuron_v2.h5`. Environment: `pytc`, `HDF5_USE_FILE_LOCKING=FALSE`, repo-root cwd.

## Verification Plan

Tests 1–12 as plan_v1, with these replacements and additions:

* **T2** asserts both mask contracts explicitly: `M_t10` pads KEEP=1 and `M_abiss` pads 0 on a
  synthetic ring chunk, and `n_missing == 0`.
* **T5/T6** are restated against the corrected rule: a `one_anchor` component emits edges; a
  `multi_anchor` component emits none, in all three bands.
* **T13 resolver permutations** (finding 5), now on reachable inputs: a fragment claimed by two
  anchors via two distinct CC3D components abstains under P2–P4 and is margin-adjudicated under
  P5; a component overlapping two anchors emits nothing; inconsistent chunk-local hosts; cycles
  and duplicated edges; assigned fragments never become anchors or relays; ≥20 random edge
  permutations must give identical output and identical reason codes.
* **T14 interface orientation** (finding 7): on a synthetic two-label volume with a known
  boundary, the directed edge set and `aff[c][v_lower]` value are asserted, and the opposite
  convention is shown to differ — so a silent convention flip cannot pass.
* **T15 alignment** (finding 10): elementwise equality on a **non-symmetric** subvolume between
  our crop and a CloudVolume read at global XYZ probes computed by an independent inline
  arithmetic implementation in the test, not by the loader under test. Histogram comparison is
  removed as a primary check.
* **T16 band boundaries** (finding 6): labels exactly at `A`, `M_lo`, `M_hi` land in exactly one
  class; `large_low_extent` is non-assignable; dust is never a fragment.
* **T17 store fidelity** (finding 3): a stored partition equals a fresh decode up to relabeling.
* **T18 schema/leakage** (finding 12): a `gt_free/*` artifact carrying any key outside the
  declared schema fails; `expected_base_nerl` passes only with `declared_constant: true`.

Data-side order: A → C0 (report `n_extra` distribution and the resulting re-decode bill before
spending it) → C1 → C2 → C3 (726/726) → C4 → C5 → **D freeze** → B1/B2 → E → F → G, with the
leakage audit before D and again before E, and `README.md` recording every skipped stage and the
gate that stopped it. Stage C shards ≤16 concurrent, not co-scheduled with another
affinity-streaming array.

## Risks and Questions

* **C0 may bill a full re-decode.** If `n_extra > 0` on most chunks, C1 costs another ~3 h and
  ~50 GB. That is the correct price; the alternative is a candidate graph whose topology was
  fixed by the wrong mask, which no downstream veto can repair. C0 is cheap and reports the bill
  first.
* **Array 2855306 is still the right thing to have launched**: it is required for B2, it bounds
  the re-decode cost by giving the mask-comparison a concrete counterfactual, and its chunks are
  reused wherever `n_extra == 0`.
* **The corrected resolver is more permissive than v1's** — it no longer rejects a whole
  candidate component when two anchors claim one fragment. It is nonetheless the task's stated
  rule, and R5 remains a hard-failure invariant. Reviewers should check R1/R3 specifically.
* **Expected result remains small or null.** Prior label-algebra fusions all lost;
  `frozen_endpoint_merges` gained +0.000129. An empty P6 with a populated P2 is an informative
  outcome, not a failure to report.
* Open question unchanged: a Gate-A pass on the *cut* family only routes to Gate C and the
  nuclei/glia task; no dense splitter is built here.

## Changes Since Previous Plan Version

1. **F1** — all evaluator work moved after the Stage-D freeze, except B1's one-bit assertion
   against a constant declared in Stage A. B2 moved after D. The GT-scored mask bridge is
   deleted; the re-decode decision is GT-free mask arithmetic with no "material" threshold.
2. **F2** — the fusion partition is built exclusively from an ABISS-keep-mask CC3D store.
   Stage C0 measures the voxelwise mask difference on all 726 chunks and Stage C1 re-decodes
   every chunk with `n_extra > 0`. The `touches_ring` veto is no longer load-bearing, and the
   "only ring chunks differ" assertion is replaced by measurement.
3. **F3** — Stage C3 reads both masks and the nucleus volume explicitly, so `mask_removed` is
   distinguishable from `below_threshold`; Stage C2 adds a store-fidelity gate proving stored
   contents equal a fresh decode up to relabeling.
4. **F4** — face and cut candidate families are now generated GT-free in Stage C4 with frozen
   definitions (mutual-best IoU ≥0.50 with ≥0.15 margin; suspects = ≥2 nucleus instances or
   ≥3×10⁷ voxels). Undefined suspect detectors are dropped explicitly. Gate A's vague clause is
   replaced by "repairs ≥1 of the three ERL-visible false merges".
5. **F5** — the invariant is corrected. Abstention is a property of the supporting CC3D
   component (R1), multi-host fragments abstain in P2–P4 and are margin-adjudicated in P5+ (R3),
   and single-anchor-per-component is a checked post-hoc invariant (R5) rather than the
   selection mechanism. Permutation tests now use reachable inputs.
6. **F6** — half-open intervals, the new `large_low_extent` class, dust declared
   diagnostic-only, six one-factor robustness neighbours off the central band only, the affinity
   pass covering the union over all variants, and full nucleus resolution/mapping semantics.
7. **F7** — the interface statistic is a directed 6-adjacent edge set with a declared channel
   and index convention; distance-2 and channel `max` removed; the opposite convention reported
   as a diagnostic and covered by test T14.
8. **F8** — the cylinder arithmetic is restated as a reference-caliber interpretation, not a
   physical validation of the bands.
9. **F9** — composition requires an exact substrate digest (ABISS `info` and LUT shard hashes);
   subset plus baseline equality are explicitly declared insufficient; both composition orders
   are defined; invariant-violating compositions are rejected rather than scored.
10. **F10** — B5 replaced by elementwise comparison on a non-symmetric subvolume with
    independent inline coordinate arithmetic; histogram checks demoted.
11. **F11** — diagnostics split into GT-free generation (C4) and post-evaluation scoring (F);
    `results.md` is written last, in Stage G, after the Gate-C suspect-detector precision exists.
12. **F12** — the leakage audit is schema-first with `expected_base_nerl` allowlisted by path
    and flagged `declared_constant: true`; keyword scanning is secondary.
