# Lesson: nucleus competition v2 — split first, then forbid re-merging

2026-08-14. This note distinguishes the current arm0 nucleus-aware ABISS decode (labelled
`arm0_96` throughout this file) from the earlier nucleus-veto and post-hoc prototypes. The
full-volume v2 decode is still in progress, so this is a method/error lesson, not a final
score report.

[corrected 2026-08-17: the `arm0_96` name is a misnomer for the whole-volume and hard7
numbers below. Those runs read
`outputs/nisb_base_banis_plus_zebrafinch_heavy/20260726_114349/test_step=00200000/0/raw_x1_ch0-1-2_chunked-raw_cs1008x1008x1008_halo72x72x72_zebrafinch_chunk_raw_grid1008_halo72.h5.chunks`
— the `aff_path` of `wholevol_arm096_fullmask.yaml` and `wholevol_arm096_nuc_competitive_v2.yaml`,
and the default `abiss_tuning.yaml` `affinity_path` inherited by the `hard7_nucleus` rung —
which is the window-144 inference; `tier10_matrix_jobs.txt` tags exactly this store
`t10_arm0_win144`, while true `[48,96,96]` output lives in the sibling
`_win48x96x96.h5.chunks` store. That native store held only the fixed ten tier-10 chunks
(2026-07-28) until slurm array 2861372 completed all 726 on 2026-08-16, so no whole-volume
window-96 number predates 2026-08-16. The two paired gates in the final section are
correctly attributed: "adversarial win144" is rung `z4_nucleus_matchguard` (default win144
path) and "native arm0_96" is rung `z4_nucleus_matchguard_win96` (explicit `_win48x96x96`
path). All measured numbers stand; only the window attribution changes.]

## Executive conclusion

The previous version asked only:

> May these two existing watershed/agglomeration objects merge?

The current version asks two separate questions in the correct order:

1. Which soma owns each voxel inside a watershed object that already contains multiple
   touching nuclei?
2. After that assignment creates distinct territories, may ABISS merge those territories
   again?

The answers use two different operators:

```text
competitive seeded watershed = creates the missing boundary
nucleus cannot-link          = preserves that boundary during agglomeration
```

This is why the new version can address the failure where nucleus cores looked separated
but soma material remained falsely merged. A merge veto cannot split one existing node.
Competition can.

## What “the last version” means

There were two predecessors, and they should not be conflated.

### 1. ABISS nucleus veto only

The `wholevol_nuc_arm1ft` run added the nucleus instance volume to mean-edge
agglomeration. Every supervoxel received one of three records:

- `NONE`: no qualifying nucleus signal;
- `PROPER(id)`: one nucleus dominates its tagged voxels;
- `CONFLICT`: multiple nucleus identities are already mixed.

ABISS rejected a proposed union between distinct `PROPER` identities and also protected a
`CONFLICT` object from merging with `PROPER` or another `CONFLICT`. This successfully
removed 33 dominant-nucleus fusion pairs according to the original audit, but it could
only act between region-graph nodes that already existed.

That result was useful but over-interpreted. The original
`nucleus_fusion_audit.py` sampled each nucleus's dominant final segment. Different dominant
segments prove that the nucleus cores differ; they do not prove that all soma material is
topologically separated. Minority shell contamination or a shared untagged soma object can
remain while every nucleus has a clean-looking dominant label.

The stronger `nucleus_shell_contamination.py --tol 0.0` audit exposed this blind spot.

### 2. Post-hoc competitive split prototype

`nucleus_competitive_split.py` demonstrated the correct operator on a completed
segmentation: flood from nucleus seeds inside one fused parent and assign each voxel to one
seed. On the `worst3` crop it reduced misplaced nucleus-mask voxels from 115,479 to 5,071
(-95.6%), removed the shared cross-nucleus segment, and left the five clean controls
unchanged.

That prototype proved that competition can place a boundary even when affinity is
confidently wrong. It was not yet a complete ABISS deployment:

- it operated after the final segmentation rather than on the watershed substrate;
- materializing a whole corrected volume required a large rewrite;
- checkpoint cannot-links were exported as
  `write_only_no_decoder_consumer`, so no downstream decoder enforced them;
- a later attachment or agglomeration step could therefore reconnect territories unless it
  separately understood the exported constraint.

The current version promotes the proven post-hoc operator into the decode and connects its
output to a real constraint consumer.

## Why nuclei could look separated while somas stayed merged

The nucleus mask is a sparse identity marker, not a complete soma segmentation. Three
failure mechanisms made “nuclei separated” weaker than it sounded.

### Dominant-label blindness

A nucleus can have 99% of its marker mass in a unique segment while its remaining marker
mass, soma shell, or proximal material participates in a shared object. Dominance then
looks excellent even though the cell bodies are not cleanly separated.

This is exactly the important contrast between the two affinity arms:

- `arm1_ft` showed both exclusion and anchoring problems: marker mass was spread and shared;
- `arm0` (the whole-volume run labelled `arm0_96`, in fact window-144 affinity) mainly showed
  an exclusion problem: dominance was 0.9905–1.0000, yet whole somata still shared segments.

For that arm, 32 final segments contained at least two qualifying nuclei, involving 74 of
465 neurons (15.9%). The misplaced-marker fraction was only 0.85%, understating the
instance-level defect by roughly twentyfold. Report fused-neuron incidence, not dominance
alone.

[corrected 2026-08-17: these figures come from `wholevol_arm096_fullmask`, whose `AFF_PATH` is
`.../raw_x1_ch0-1-2_chunked-raw_cs1008x1008x1008_halo72x72x72_zebrafinch_chunk_raw_grid1008_halo72.h5.chunks`;
that store is the window-144 inference (t10_arm0_win144), not window-96. Genuine window-96
affinity was not computed whole-volume until 2026-08-16, slurm 2861372. Numbers unchanged.]

### The representational floor

In the worst measured case, one watershed supervoxel already held 100% of the mixed
nucleus-tagged mass: 28,576,381 tagged voxels inside a 243-million-voxel object spanning 88
atomic chunks. At the region-graph level it was one node before mean-edge agglomeration
started.

No merge ordering, multicut, mutex rule, nucleus edge veto, or other graph method can divide
the interior of one node. The old veto had no candidate edge on which to act. This is why
three earlier in-pipeline variants changed the measured result by exactly zero:

1. nucleus-first merge ordering;
2. conflict-clause merge veto;
3. a global per-supervoxel nucleus table.

### A veto says “do not join,” not “where to cut”

The fused somata have an affinity bottleneck near 0.999 and remain connected at thresholds
0.9, 0.99, and 0.999. There is no weak membrane surface for ordinary agglomeration to find.
A veto can retain whatever boundary the watershed happened to produce, but it cannot invent
the missing boundary. Seed competition supplies the missing identity prior.

## Current v2 pipeline

The canonical order is:

```text
arm0 affinity (window-144 store, historically labelled arm0_96) + keep mask
  -> ABISS watershed L0..L5
  -> global watershed remap
  -> nucleus contact detection and competitive growth
  -> sparse territory overlay while building atomic RAGs
  -> nucleus-aware mean-edge agglomeration L0..L5
  -> final agglomeration remap

The base watershed precomputed volume is not rewritten. The competition stage writes a
small manifest and pooled territory arrays. `cut_chunk_agg.py` overlays those labels on each
watershed cutout before the atomic region graph is constructed. Thus agglomeration sees a
refined watershed even though `WS_PATH` remains an immutable base artifact.

### Step 1: find real multi-nucleus targets

`nucleus_competition.py` scans nucleus instance mask v2 and maps each nucleus to watershed
IDs. A nucleus qualifies in a watershed object only when that object contains at least 2%
of that nucleus's own total mask mass (`NUC_MIN_SHARE=0.02`). The relative threshold avoids
splitting a healthy soma because of a tiny mask leak.

Only watershed IDs with at least two qualifying nuclei become candidates.

### Step 2: separate contacts from neurite bridges

Not every multi-nucleus segment is a soma contact. In the earlier final `arm0_96`
segmentation (the `wholevol_arm096_fullmask` run), the 32 cases divided evenly:

- 16 soma-contact cases below an 8 µm surface-gap threshold;
- 16 neurite-bridge cases, with nuclei as far as 80.6 µm apart.

[corrected 2026-08-17: the `arm0_96` name is a misnomer. That run's `aff_path`
(`.../20260726_114349/test_step=00200000/0/raw_x1_ch0-1-2_..._grid1008_halo72.h5.chunks`,
`wholevol_arm096_fullmask.yaml:40`) is the **window-144** inference — `build_tier10_jobs.py`
keys it `arm0_win144` and `tier10_matrix_jobs.txt` tags it `t10_arm0_win144`; the store's
filename encodes chunk size and halo only, never the window. Every whole-volume `arm0_96`
figure in this note comes from that store and is therefore window-144 affinity. Genuine
window-96 affinity (the separate `_win48x96x96` store) was not computed whole-volume until
2026-08-16, slurm array 2861372.]

V2 estimates each nucleus center and equivalent-sphere radius from its instance mask. Nuclei
whose signed surface gap is below `NUC_CONTACT_UM=8.0` form a contact unit. Only contact
units compete. Disconnected/far groups are written to `bridges_left_untouched`; drawing a
nearest-nucleus boundary halfway along a neurite would be biologically arbitrary.

Bridge errors still require a different signal: caliber continuity, crossness, tubeness,
or a targeted branch/glia separator. The nucleus mask detects those errors but does not
locate their cut.

### Step 3: assign the fused object to nucleus territories

Each contact unit is flooded inside its original watershed parent and a bounded repair box.
Current settings are:

```text
NUC_COMPETITION_FACTOR       4
NUC_COMPETITION_MARGIN_ZYX   [1024, 1024, 512]
affinity channels            [0, 1, 2]
cost                         1 - min(channel affinity)
connectivity                 6-connected
```

Pooling uses the minimum affinity as a conservative path cost. A seed survives pooling only
when the pooled block unanimously belongs to one nucleus. The parent mask is max-pooled, so
the flood cannot leave the object being repaired.

Every adjudicated territory receives the deterministic nucleus-owner ID declared by the
publication's `identity` block. The mint key is the nucleus alone, so one nucleus receives the
same ID under different parents and in the single-owner canonicalization table. Unadjudicated
residue deliberately retains the parent watershed ID; parent-ID absence is therefore neither
asserted nor used as an acceptance gate.

The flood geometry is contained by its input parent, but the published labeling is **not** a
refinement-only operation. Canonicalization assigns the same nucleus-owner ID to every qualified
single-owner base segment. In native96, 77 source segments consolidate into 15 labels, with one
nucleus absorbing as many as seven base segments. If a nucleus mask ever places one instance
across two genuinely different cells, this naming step silently fuses them; a region-graph
cannot-link cannot detect that fusion. The manifest ledger therefore records every retired source
and each consolidation's complete `sources` list.

Validation follows the artifact, not institutional memory: a validator may enforce what a
publication declares about its identity and residue, but may not impose a convention remembered
from another run.

### Step 4: overlay before the atomic RAG

The competition manifest pins the exact `WS_PATH`. A missing manifest fails closed, and a
manifest built from a different watershed is rejected. Each agglomeration chunk loads only
territories intersecting its cutout and replaces the selected parent voxels before writing
`seg.raw`.

This is the crucial improvement over the old veto: the region graph now contains an edge
between distinct soma territories instead of one indivisible mixed node.

### Step 5: prevent re-merging at every hierarchy level

`NucExtractor` counts nucleus instance IDs on the refined supervoxels. With the current
settings, a record becomes `PROPER(id)` when it has at least 50 tagged voxels and one identity
owns at least 60% of its tagged mass:

```text
ABISS_NUC_MIN_TAGGED  50
ABISS_NUC_DOMINANCE   0.6
```

Before accepting any mean-edge union, ABISS evaluates `nuc_can_merge`:

- `PROPER(A) + PROPER(A)` may merge;
- `PROPER(A) + PROPER(B)` is rejected for `A != B`;
- `CONFLICT + PROPER` and `CONFLICT + CONFLICT` are rejected;
- `NONE` may merge with a proper or conflict object, allowing untagged branches to grow;
- all accepted records are joined and propagated through composite chunks.

Rejected edges are written to `nuc_cuts.data`. The rule is applied before semantic and size
heuristics and is independent of the affinity value. A 0.999 edge cannot override a distinct
nucleus identity.

## Is this the semantic-channel cut?

The code path is similar in shape but stronger in semantics.

Both systems aggregate side information per region and veto an edge when two confident
regions disagree. The important differences are:

| | Semantic cut | Nucleus v2 |
|---|---|---|
| signal | dense class counts | sparse instance identities |
| pre-RAG split | none | competitive territory overlay |
| when edge veto applies | only below semantic affinity threshold | every candidate edge |
| uncertainty behavior | low signal or low dominance permits merge | explicit `NONE` / `PROPER` / `CONFLICT` algebra |
| identity scope | class disagreement | individual nucleus cannot-link |
| output audit | `sem_cuts.data` | `nuc_cuts.data` |

The pre-RAG competition has no semantic-channel analogue. That is the part that breaks the
old representational floor.

## Error modes and non-guarantees

V2 provides a stronger guarantee, but only within the information supplied to it.

### Nucleus-mask errors

- A missed nucleus creates no seed and therefore no protected identity.
- Two cells mislabeled as one nucleus cannot be separated by an instance-ID cannot-link.
- A split/duplicate nucleus can induce an unnecessary soma cut.
- Bad mask geometry changes the equivalent-sphere contact test.

Nucleus mask v2 is therefore part of the model, not infallible ground truth.

### Contact/bridge classification errors

The 8 µm rule uses centers and equivalent-sphere radii. Elongated, clipped, or inaccurate
nuclei can be classified incorrectly. A false contact can cut proximal neuropil; a false
bridge leaves a soma fusion unrepaired. The manifest must report both classes for review.

### Boundary placement is prior-driven

Affinity is confidently wrong in these burst errors. Seeded watershed will always produce
an assignment boundary, but the exact surface is not evidence of a real membrane. Factor-4
pooling further quantizes it. The reliable claim is cell identity separation, not
voxel-perfect soma morphology.

### Exclusion does not imply anchoring

V2 prevents distinct nucleus territories from merging. It does not force every disconnected
piece belonging to one nucleus to merge. False splits and proximal-branch attachment remain
separate problems. The required order is still exclusion before anchoring: anchoring through
an unresolved shared blob would reconnect all identities.

### Untagged growth is intentionally permissive

`NONE` objects may attach to a protected identity. This is needed to recover neurites, but it
means the cannot-link alone does not decide ownership of every untagged branch. If one
untagged object bridges two identities, edge order and the first accepted attachment matter;
the second identity is then blocked. This preserves exclusion but may assign the bridge to
the wrong cell.

### Far neurite/glia bridges are intentionally unsolved

V2 abstains on far nucleus groups. It should not be reported as a general false-merge or glia
splitter. Those cases need a separate, morphology-aware correction that can justify where a
branch or glial hub should be cut.

### Scope and pooling failures fail closed

Missing unanimous seeds, overlapping repair scopes for one parent, generated-ID collisions,
an incompatible watershed manifest, or invalid nucleus coordinates abort the stage. Silent
fallback to unconstrained agglomeration would invalidate the experiment.

## Full-volume launch incident: zero-filled affinity from an incomplete NFS listing

The first full-volume attempt failed at watershed L1, but the L1 merge was only where the
damage became visible. Five reproduced failures each contained one L0 child whose stored
boundary affinity was partly or entirely zero. For example, child `0_9_6_8` had an all-zero
`aff_i_2`; the verified `wholevol_arm096_fullmask` baseline (window-144 affinity) and a fresh
direct read of the same source seam both had 225,452 nonzero and 36,692 zero values.
[corrected 2026-08-17: despite the `arm0_96` name, every `wholevol_arm096_*` run reads
`.../grid1008_halo72.h5.chunks`, which is the window-144 inference (`t10_arm0_win144`);
genuine window-96 affinity (`..._grid1008_halo72_win48x96x96.h5.chunks`) was not computed
whole-volume until 2026-08-16, slurm array 2861372.]

This was not an OOM or a stochastic ABISS merge failure. The affected children crossed a
1008-voxel HDF5 chunk seam. `_ChunkedH5Array` discovered source chunks with `os.listdir()`
independently in every worker. Under the 1,520-process L0 fanout over NFS, an incomplete
directory listing could omit a real chunk. The adapter interpreted the absent filename as
an absent spatial cell and silently left that part of the requested array as zero. The L0
watershed remained locally well-formed, but its saved boundary contract no longer matched
its neighbor, so composite merging correctly rejected it at L1.

The production adapter now derives the complete filename grid deterministically from the
configured ABISS `BBOX`, the known chunk shape, and the required
`chunk_z0_y0_x0.h5` probe. A missing required file reaches `h5py` and fails the task instead
of synthesizing zero affinity. Directory listing remains only a non-production fallback.
Tests prohibit `listdir` on the configured path, exercise a two-file seam, and require a
missing grid member to raise.

The recovery rule is strict:

> If a source-read defect may have changed L0 watershed contents, do not resume or reuse
> those L0 artifacts. Cancel the dependent chain and prepare a fresh output namespace.

Accordingly, the contaminated arm0 and arm2 chains were canceled and preserved for audit.
Fresh `v2` namespaces were prepared, and an end-to-end atomic seam probe matched the
verified baseline payload hash exactly
(`2825d0fd0cf39ee60013fa9bf43dd5531d864c5b68580fefc9d49bab40aeabbc`) before either full
chain was submitted.

This incident also exposed a provenance weakness: a Git commit alone did not identify the
runtime because ABISS commonly executes edited scripts and locally rebuilt, untracked
binaries. Replay manifests now pin both Git HEAD and a SHA-256 digest of all ABISS runtime
scripts plus executable build outputs. Preparation must be rerun after any runtime edit or
rebuild. Do not edit those files while a submitted chain is using them.

## Controlled hard7 result: the boundary is created, then reconnected

Before interpreting the full-volume jobs, the current arm0_96 parameters were run on seven
fixed hard chunks in three paired arms:

1. current arm0_96 ABISS control;
2. the same parameters plus a chunk-local, 26-connected CC3D nucleus veto;
3. the same parameters plus competitive pre-RAG nucleus territories and the veto.

The mask was cut from `yl_cb_80nm_neuron_v2.h5` at the exact chunk bboxes. CC3D was applied
per source nucleus ID, keeping only its largest connected component so that disconnected
islands could not become invented instances. This is a controlled decoder comparison: affinity,
FFN mask, ABISS parameters, chunks, and scoring are identical between arms.

| arm | hard7 human mt50 linear NERL | hard7 human mt1 linear NERL | fused source/CC3D pairs |
|---|---:|---:|---:|
| control | 0.7788734050 | 0.7714437401 | 1 |
| local nucleus veto | 0.7788734050 | 0.7714437401 | 1 |
| competitive split + veto | 0.7788734050 | 0.7714437401 | 1 |

The pseudo-label score moves by only about `1e-5`. All three arms therefore have the same
human NERL and the same single fusion on `z4_y6_x1`.

This is not because competition failed to identify the case. In `z4_y6_x1`, watershed
segment `72057662824513538` contains CC3D/source nuclei 611 and 651. The corrected
competition run assigns separate deterministic territories, touches about 22.03 million
voxels in the overlay, and records a 2.132 um surface gap. Yet the final segmentation still
places both nuclei in segment `72128031635904131`, with exactly the control NERL
(`0.8611928694`). The veto and competition arms merely change the final label count
(29,315 control; 30,109 veto; 29,954 competition).

The likely bypass is later than the RAG operator. `match_chunks.cpp::process_nucs` applies
the chunk remap and then joins nucleus records; when two different nucleus records collide it
logs `nuc: load_conflict_collisions` but does not call `nuc_can_merge`. The corrected z4 run
logs one such collision. The mean agglomerator does call `nuc_can_merge`, but by then it
cannot undo a remap performed during internal chunk matching. Thus the experiment establishes
that the current pre-RAG boundary reaches ABISS, but does **not** establish preservation
through the complete hierarchy.

One false start must remain excluded from the comparison. Replacing the local runner payload
with a competition manifest did not recompute its stage list, so the nominal competition arm
initially skipped `competitive_nucleus_growth`. The valid rerun supplies an explicit manifest
and voxel size and recomputes the default stages. A decoder-arm name is not provenance;
record and verify the executed stage list.

Decision: do not promote these nucleus parameters yet. First make chunk matching honor the
same cannot-link constraint, then rerun `z4_y6_x1` alone. Promotion requires distinct final
segments for nuclei 611 and 651, zero load-conflict collision for that pair, unchanged output
on the six abstaining controls, and no mt1/dominance regression. Only then rerun hard7 and
advance to larger regions.

## What must be measured when the run completes

Do not declare success from nucleus dominance or NERL alone.

1. Before scoring, verify that chunk matching preserves nucleus cannot-links:
   - count `load_conflict_collisions` at every hierarchy level;
   - report remaps rejected by the nucleus constraint;
   - require the known z4 pair to remain in distinct final segments.
2. Inspect the competition manifest:
   - multi-nucleus watershed IDs;
   - repaired contact units;
   - bridges left untouched;
   - seed/territory voxel counts and repair boxes.
3. Run `nucleus_shell_contamination.py --tol 0.0` on the final segmentation.
4. Report at least:
   - fused final segments;
   - fused nucleus pairs;
   - neurons participating in a fusion;
   - contact and bridge cases separately;
   - per-nucleus dominant fraction and number of segments needed for 90% mass.
5. Verify already-clean nuclei and outside-repair material are unchanged.
6. Count nucleus-rejected RAG edges from `nuc_cuts.data` across hierarchy levels.
7. Score canonical funlib NERL at merge threshold 50 against the unchanged
   `wholevol_arm096_fullmask` baseline (`0.444376`) — window-144 affinity despite the
   `arm096` name.
   [corrected 2026-08-17: that run's `AFF_PATH` (`seg_arm096_fullmask/.../param`) is
   `…_grid1008_halo72.h5.chunks`, which `tier10_matrix_jobs.txt` labels `t10_arm0_win144`;
   genuine window-96 affinity (`…_win48x96x96.h5.chunks`) was not computed whole-volume until
   2026-08-16, slurm array 2861372. The number and any A/B against it are unaffected — only
   the window attribution carried by the name.]
8. Inspect per-skeleton regressions; a global gain can hide over-splitting.

The prior support audit found that soma-contact cases carry 34,615 of 500,845 skeleton
nodes (6.91%), so the full-volume correction can affect NERL. Still, soma-fusion incidence
is the direct metric for the biological constraint and must be reported alongside ERL.

## Status and artifacts

As of 2026-08-15, the corrected arm0 ch0-2 full-volume decode is operationally complete.
Recovery nucleus job `2856392` completed in 7 h 20 min; constrained agglomeration and final
remap jobs `2856393` through `2856399` all completed with zero failed final shards. The
competition stage found nine multi-nucleus watershed objects, classified all nine as contact
units, and wrote nine territory overlays. All expected watershed, agglomeration, and remap
DONE counts are complete.

Canonical funlib NERL at merge threshold 50 is `0.468030323127`, versus `0.444376` for the
`wholevol_arm096_fullmask` baseline: `+0.023654323127` absolute and `+5.323%` relative, at 0.996734
node coverage. Both runs read the identical `AFF_PATH`
(`.../raw_x1_ch0-1-2_chunked-raw_cs1008x1008x1008_halo72x72x72_zebrafinch_chunk_raw_grid1008_halo72.h5.chunks`),
so the A/B delta stands; the affinity is window-144, not window-96, despite the `arm096` name.
[corrected 2026-08-17: this store is the window-144 inference (t10_arm0_win144); genuine
window-96 affinity was not computed whole-volume until 2026-08-16, slurm 2861372] This is a real whole-volume score for the completed artifact, but it does not
clear the biological promotion gate above. The current chunk-matching path still lacks a
`nuc_can_merge` veto; therefore the shell-contamination audit, collision counts, and known z4
pair remain required before attributing the gain to preserved soma separation.

The independent arm2 ch3-5 decode uses its own v2 namespace. Do not infer its result from
the arm0 score.

Current implementation and launch configs:

- `wholevol_arm096_nuc_competitive_v2.yaml`
- `wholevol_arm2mix_r10_nuc_competitive_v2.yaml`
- `lib/abiss/scripts/nucleus_competition.py`
- `lib/abiss/scripts/nucleus_overlay.py`
- `lib/abiss/scripts/cut_chunk_agg.py`
- `lib/abiss/src/seg/NucExtractor.hpp`
- `lib/abiss/src/seg/Types.h`
- `lib/abiss/src/agg/mean_aggl.cpp`
- `tests/unit/test_abiss_nucleus_competition.py`
- `lib/abiss/tests/test_nuc_algebra.cpp`
- `lib/abiss/tests/test_nuc_agg.py`
- `abiss_tuning/fixed_tiers/reports/hard7_nucleus/summary.json`
- `reports/abiss_nucleus_cc3d_hard7_0814.md`

Previous methods and diagnostics:

- `nucleus_competitive_split.py`
- `nucleus_split_wholevol.py`
- `nucleus_fusion_audit.py`
- `nucleus_shell_contamination.py`
- `conflict_bottleneck.py`
- `lesson_abiss.md`, L126

The shortest correct summary is:

> V1 protects RAG boundaries that already exist. V2 creates the missing soma boundary inside
> a fused watershed object, but current chunk matching can remap the two territories together
> before enforcing the nucleus constraint. Hierarchy-wide protection is not yet established.

## Superseding implementation: identity must survive every representation boundary

The preceding summary describes the first competitive implementation and its initial
failure. The current implementation is stricter. It treats nucleus identity as an invariant
that must survive four different representations, not as one extra edge test in mean-edge
agglomeration.

The important differences from the last version are:

1. **Mask support is measured over the actual mip-0 footprint.** The old manifest sampled
   one high-resolution center per 80-nm nucleus voxel. The RAG sees nearest-neighbour blocks
   of `4 x 8 x 8 = 256` voxels, so center sampling could qualify the wrong watershed/owner
   pair. The manifest now accumulates exact blockwise histograms over that expanded footprint.
   `ABISS_NUC_MIN_TAGGED=1024` consequently means four source-mask voxels; the former value
   50 was smaller than a single expanded voxel.
2. **Owner identity is global only for a real repair.** A zero-repair publication is still a
   complete publication and still runs ownership filtering and single-owner canonicalization;
   it is a label no-op only when those tables are empty. For owners participating in a competitive repair,
   sparse tags below the global `NUC_MIN_SHARE` qualification are removed, every qualified
   soma piece for one owner receives the *same* deterministic protected ID, and each flooded
   territory receives that ID plus a dense owner tag. Different owners still receive different
   IDs. This prevents cross-child aliasing without fragmenting one soma into a protected label
   per watershed fragment.
3. **Agglomeration preserves identity-bearing representatives.** A tagged component remains
   the representative when it absorbs an untagged component. Nucleus cannot-links are checked
   on every mean-edge merge, and hierarchy matching carries nucleus state. This fixed the
   original `load_conflict_collisions` failure, including the 611/651 reproduction.
4. **Array layout is explicit.** CloudVolume cutouts are commonly Fortran ordered. Flattening
   a channel view with default C-order `reshape` produced a copy, so logs reported filtering
   and canonicalization while `seg.raw` and `nuc.raw` remained unchanged. Overlay code now
   mutates explicit C working arrays and always writes both channels back to the original
   cutouts. A Fortran-order regression test covers this path.
5. **The final remap replays the same declared overlay.** The RAG path used the competitive labels,
   but the old final writer reloaded the original `WS_PATH`. A hard gate could therefore show
   zero hierarchy collisions, distinct top-level owner records, and a rejected nucleus edge,
   yet fuse 611 and 651 in the materialized segmentation. `cut_chunk_remap.py` now applies the
   same competition overlay before the aggregation remap table is evaluated.

The fifth issue is qualitatively different from the earlier false merge. Internal graph
state was already correct: the failed diagnostic retained owner 1 as segment
`1202455102220122803`, owner 4 as `1217835787581195885`, and one final nucleus-rejected edge.
Only the output writer reintroduced the fusion. This is why collision logs alone cannot be
the promotion gate.

The required proof now crosses the full pipeline:

- the manifest uses exact expanded-footprint support and globally qualified owners;
- a zero-repair manifest completes ownership filtering and canonicalization, and the native gate
  remains unchanged because it declares neither protected owners nor canonicalization entries;
- every hierarchy-level `load_conflict_collisions` count is zero;
- top-level nucleus records retain distinct owners and record the expected veto;
- the materialized segmentation reports `fused_source_pairs == 0` and maps nuclei 611 and
  651 to distinct nonzero dominant labels;
- NERL is reported with the canonical funlib formula at merge threshold 50. The mt1 oracle is
  not a selection metric because a few outlier false-merge voxels should not invalidate an
  otherwise good segmentation.

The final paired gates satisfy that proof with runtime hash
`1686177c2546ffe0f3c9dd067885ab66bf41b85928fff92a2d01763010771d45`:

- The adversarial win144 gate has one repair and protected owners 1 and 4. All 18 hierarchy
  loads report zero conflicts. Source nuclei 611 and 651 have distinct dominant labels,
  dominance 0.98249 and 0.99062, and one segment each for 90% of their mass. There are zero
  fused source pairs. Canonical human mt50 linear NERL is `0.8611928694330402`, restoring the
  pre-fix score while actually separating the nuclei.
- The native arm0_96 gate has zero multi-nucleus watershed targets, zero repairs, and zero
  protected owners. All 16 RAG cutouts and all 16 final-remap cutouts take the explicit no-op
  path; all 18 hierarchy loads are conflict-free. Sources 611 and 651 retain their original
  distinct affinity-derived labels with dominance 0.98525 and 0.99044 and one segment each
  for 90% mass. Canonical human mt50 linear NERL remains `0.9647168642163645`.
- Focused Python tests cover expanded-footprint histograms, competitive ownership,
  Fortran-order writeback, protected-only filtering, shared per-owner labels, zero-repair
  no-op behavior, and final-remap replay.

In short: the last version created a soma boundary locally; the current version gives each
side a durable identity, anchors the pieces belonging to the same repaired nucleus, protects
different identities through atomic and composite graph operations, and reconstructs the
same labeled state when writing the final volume.
