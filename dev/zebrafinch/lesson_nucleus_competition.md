# Lesson: nucleus competition v2 — split first, then forbid re-merging

2026-08-14. This note distinguishes the current `arm0_96` nucleus-aware ABISS decode from
the earlier nucleus-veto and post-hoc prototypes. The full-volume v2 decode is still in
progress, so this is a method/error lesson, not a final score report.

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
- `arm0_96` mainly showed an exclusion problem: dominance was 0.9905–1.0000, yet whole
  somata still shared segments.

For `arm0_96`, 32 final segments contained at least two qualifying nuclei, involving 74 of
465 neurons (15.9%). The misplaced-marker fraction was only 0.85%, understating the
instance-level defect by roughly twentyfold. Report fused-neuron incidence, not dominance
alone.

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
arm0_96 affinity + keep mask
  -> ABISS watershed L0..L5
  -> global watershed remap
  -> nucleus contact detection and competitive growth
  -> sparse territory overlay while building atomic RAGs
  -> nucleus-aware mean-edge agglomeration L0..L5
  -> final agglomeration remap
```

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
segmentation, the 32 cases divided evenly:

- 16 soma-contact cases below an 8 µm surface-gap threshold;
- 16 neurite-bridge cases, with nuclei as far as 80.6 µm apart.

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

The largest territory retains the original watershed ID. Other territories receive
deterministic IDs above `2^60`, outside ABISS's native ID range. IDs are derived from the
parent/anchor pair, collision-checked, and stable across resume.

The important safety invariant is refinement:

> Every output territory is a subset of its input watershed object.

Competition cannot merge two previously distinct objects. It can only divide a selected
one.

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
`aff_i_2`; the verified `arm0_96` baseline and a fresh direct read of the same source seam
both had 225,452 nonzero and 36,692 zero values.

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

## What must be measured when the run completes

Do not declare success from nucleus dominance or NERL alone.

1. Inspect the competition manifest:
   - multi-nucleus watershed IDs;
   - repaired contact units;
   - bridges left untouched;
   - seed/territory voxel counts and repair boxes.
2. Run `nucleus_shell_contamination.py --tol 0.0` on the final segmentation.
3. Report at least:
   - fused final segments;
   - fused nucleus pairs;
   - neurons participating in a fusion;
   - contact and bridge cases separately;
   - per-nucleus dominant fraction and number of segments needed for 90% mass.
4. Verify already-clean nuclei and outside-repair material are unchanged.
5. Count nucleus-rejected RAG edges from `nuc_cuts.data` across hierarchy levels.
6. Score canonical funlib NERL at merge threshold 50 against the unchanged
   `wholevol_arm096_fullmask` baseline (`0.444376`).
7. Inspect per-skeleton regressions; a global gain can hide over-splitting.

The prior support audit found that soma-contact cases carry 34,615 of 500,845 skeleton
nodes (6.91%), so the full-volume correction can affect NERL. Still, soma-fusion incidence
is the direct metric for the biological constraint and must be reported alongside ERL.

## Status and artifacts

As of 2026-08-14, the corrected fresh full-volume decodes have entered the Slurm watershed
stage. The arm0 ch0-2 chain is `2855166` (watershed L0) through `2855180` (final remap); the
independent arm2 ch3-5 chain is `2855181` through `2855195`. L0 requests 19 CPUs and 180 GiB
per shard. The competition manifests and final nucleus/NERL results do not exist yet. Crop
behavior, the repaired affinity seam, and software contracts are verified; whole-volume
effectiveness remains pending.

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

Previous methods and diagnostics:

- `nucleus_competitive_split.py`
- `nucleus_split_wholevol.py`
- `nucleus_fusion_audit.py`
- `nucleus_shell_contamination.py`
- `conflict_bottleneck.py`
- `lesson_abiss.md`, L126

The shortest correct summary is:

> V1 protected boundaries that already existed. V2 creates the missing soma boundary
> inside a fused watershed object, then protects it through the complete ABISS hierarchy.
