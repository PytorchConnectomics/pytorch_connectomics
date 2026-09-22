You are reviewing revision v4 of an implementation plan for a CCC run. You are the CODER: you will
implement this plan, so review it for executability.

You reviewed plan_v0 (11 [major]), plan_v1 (G1-G6), plan_v2 (H1-H6), and plan_v3 (I1-I5). Plan_v4
responds to I1-I5.

IMPORTANT CONTEXT ON I1: you showed a single-dominant-id record cannot deliver exact cannot-link.
The human owner of this project has DECIDED, on the record, to ship the weaker guarantee
("Invariant D": no merge joins clusters whose RECORDED DOMINANT ids differ) rather than widen the
record to identity sets or add NUC_WS. That decision is final and is not open for re-litigation.
Judge plan_v4 on whether it states that weaker contract HONESTLY and IMPLEMENTS it CORRECTLY - not
on whether the contract should have been stronger.

Plan_v4's other changes: the evidence floor now applies to aggregate `total` rather than to
individual ids (renamed ABISS_NUC_MIN_TAGGED, default 50); the merge-time check calls
`nuc_can_merge` itself and aborts whenever it is false; BOTH hierarchy aborts were removed in favor
of `nuc_join` plus counters, per your I4; and V1-V10 were made concrete.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Attack it on:
1. Are I2, I3, I4, I5 really fixed? Name any that are not.
2. Is Invariant D stated accurately, and does B3/B4 actually implement exactly that and nothing
   weaker? Find any merge B3 permits that Invariant D forbids, or vice versa.
3. Does applying the floor to aggregate `total` really keep the record self-consistent? Attack it.
4. With both hierarchy aborts removed, is any invariant now unobservable in a way that matters?
5. Are V1-V10 runnable as specified by someone who has not designed this? Name any fixture,
   argument, or file whose format is still undefined.
6. Anything the revision broke that plan_v3 had right.

If the plan is ready to implement, say so plainly rather than manufacturing findings. A finding
without a concrete failing case or a concrete missing specification is not useful at this stage.

=========================== BEGIN task.md ===========================
# Task

Add a first-class **nucleus instance mask** input to `lib/abiss` (`NUC_PATH`, dtype
`uint32`) and use it to guide agglomeration, so that supervoxel clusters carrying
different nucleus ids are never merged into one segment.

The literal user request:

> "given the context above, add a feature in abiss to make use of nucleus instance mask uint32"

## Repository and Working Copy

The code lives in the **`lib/abiss` repository** (`https://github.com/PytorchConnectomics/abiss`),
which is a standalone git repo. `lib/` is gitignored by the parent `pytorch_connectomics`
repo, so abiss changes never appear in a parent diff. All CCC git baselines and diffs for
this run therefore target abiss, not pytorch_connectomics.

**Standing constraint carried over from `.agent/features/abiss_speedup/`:** build
experiments must NOT touch `lib/abiss` in place. A previous run broke a live SLURM chain by
rebuilding `lib/abiss/build/` while jobs were executing those binaries. SLURM job 2787863
(`v3mim_verify`, from the abiss_speedup bench) was running against abiss at the start of
this run.

Accordingly, this run works on an isolated clone:

```text
.agent/features/abiss_nucleus/work/abiss   (git clone of lib/abiss, detached at 3c4f562, clean, no build/)
```

The live `lib/abiss` checkout is read-only for this run. All edits, builds, and diffs happen
inside `work/abiss`.

## Background: what abiss already has

ABISS already carries a *semantic class* guidance path end-to-end, at three injection
points. The nucleus feature should parallel it, not overload it.

**P1 - pre-watershed affinity cut.** `scripts/cut_chunk_ws.py:31` calls
`augment_affinity.py:147` `mask_affinity_with_semantic_labels`, which zeroes the affinity
edge wherever the two voxels carry *different nonzero* sem labels. This test is already
identity-based (`sem_offset != sem_aligned`), not class-based. Gated by `SEMANTIC_WS`.

**P2 - per-supervoxel payload.** `src/seg/SemExtractor.hpp` accumulates a fixed 3-bin
histogram per supervoxel through a 6->3 class LUT (`sem_map`, line 61). Serialized as
`ongoing_semantic_labels.data` and carried through the chunk hierarchy by
`match_chunks.py:126`, `reduce_chunk.py:111`, `merge_chunks_me.py:59`,
`merge_chunks_overlap.py:65`.

**P3 - agglomeration veto.** `src/agg/mean_aggl.cpp:625` `sem_can_merge` requires both
clusters to have a dominant label (>= `dominant_signal_ratio` 0.6 of counts, >=
`total_signal_threshold` 100k voxels); if the dominants differ the edge is refused and
logged to `sem_cuts.data`. Applied at line 699 **only when the edge weight is <= 0.5**
(`sem_params.aff_threshold`, line 130). Cluster payload is combined by summation at lines
754-758.

**P4 - size veto.** `src/agg/mean_aggl.cpp:709-718` refuses a merge when both sides are
large. This heuristic exists precisely because the code cannot otherwise tell a real cell
body from a runaway merge.

## Why a separate `NUC_PATH` rather than reusing `SEM_PATH`

Decided with the user before this run started:

1. `semantic_t = uint8_t` (`src/seg/Types.h:25`) caps instances at 255. Widening it in
   place would change the on-disk meaning of every existing sem file.
2. `SemExtractor` is lossy by design (6 classes -> 3 bins); an instance id must survive
   verbatim.
3. The predicates genuinely differ. Sem is "different classes should not merge, and only
   for weak edges." Nucleus is "different cells must never merge, at any affinity." One
   shared `aff_threshold` forces one of them to be wrong.
4. They compose: class veto for axon/dendrite *and* instance veto for cell bodies should be
   usable simultaneously. Overloading one channel makes them mutually exclusive.
5. The sem path is already inconsistent: `match_chunks.py:122` reads `nlabels = 5` while
   `reduce_chunk.py:108` and `sem_array_t` are both 3. That OVERLAP=2 branch is broken or
   dead; building on it inherits the defect. Do not copy `process_sems` blindly.

## Design intent

Carry a fixed-width per-cluster nucleus record so every plumbing script stays a one-line
mirror of the existing sem lines:

```text
(seg_t sid, nuc_t id, size_t count, size_t total)
```

`count` = voxels of the dominant id, `total` = all nucleus-tagged voxels in the cluster.
This is exact *given* the veto, because a cluster can never legitimately end up holding two
different ids: any such merge is refused before it happens. Do not serialize a
variable-length id->count map; that is only needed for a soft veto and it breaks the
fixed-record assumption in five scripts.

The veto should be **ungated by affinity**, unlike the sem veto: an instance nucleus mask is
far more trustworthy than a semantic net prediction, and a soma-soma fusion is far more
expensive than a stray cut. Keep a minimum tagged-voxel threshold so a handful of
misassigned voxels cannot hard-split a real cell.

Optional companion, if the plan judges it in scope: a `NUC_WS` flag mirroring `SEMANTIC_WS`
at `cut_chunk_ws.py:31`, reusing the `mask_affinity_with_semantic_labels` mechanism. It
makes "one supervoxel carries at most one nucleus id" true by construction, which is what
makes the fixed-width record exact.

`traverseSegments` (`src/seg/Utils.hpp:112`) is fully variadic, so adding an extractor is
mechanical. Note that `src/seg/atomic_chunk_ME.cpp:78-104` currently duplicates the whole
call for the sem/no-sem case; a second optional payload turns that into four branches, so
prefer making the new extractor a no-op when its input file is absent, or hoist the pack.

Also note: agglomeration parameters are compile-time constants
(`src/agg/mean_aggl.cpp:121-153`); `main` only parses `argv[1]` as the agglomeration
threshold (lines 1145-1149). Any new tunable is a recompile unless the plan adds env or
argument parsing.

## Scope

**In scope:** the `lib/abiss` repository only - C++ (`src/seg/`, `src/agg/`) and the
Python/bash chunk drivers under `scripts/`.

**Out of scope:** the pytorch_connectomics wrapper. `connectomics/decoding/decoders/abiss.py`
has no `SEM_PATH` plumbing today and gains no `NUC_PATH` plumbing in this run. Also out of
scope: generating the nucleus instance mask itself, and the perinuclear-shell tagging
preprocessing step described below.

## Known caveat the implementation cannot fix

The nuclear envelope is a membrane; affinity networks put a boundary on it, so the nucleus
interior often becomes its own cluster that never joins the soma. An id parked there
constrains nothing. Correct usage tags *perinuclear cytoplasm* (dilate each instance, take
shell = dilated minus nucleus, assign id k to supervoxels dominated by shell k). That is a
preprocessing step on the nucleus volume, outside abiss and outside this run's scope, but
the feature should be documented so a user does not assume raw nucleus interiors work.

## Constraints

* Do not modify the live `lib/abiss` checkout; work only in `work/abiss`.
* Do not create git commits during this CCC run.
* Do not add runtime or build dependencies without explicit user approval
  (`AGENTS.md` Environment section, parent repo).
* Existing behavior with no `NUC_PATH` configured must be bit-identical to the current
  pipeline. This repo reproduces a Seuron provenance record; a silent change to default
  segmentation output is a hard failure.
* Keep the diff scoped. Do not opportunistically fix the `nlabels = 5` sem bug in the same
  change unless the plan argues it is required; if it is left alone, say so.

## Success criteria

1. `NUC_PATH` (plus any needed `NUC_MIP`) is accepted end-to-end: config JSON -> `set_env.py`
   -> `cut_chunk_agg.py` writes `nuc.raw` -> C++ extractor -> agglomeration veto ->
   per-chunk serialization -> chunk hierarchy reduce/merge/match.
2. Two clusters carrying different nucleus ids are never merged, at any affinity, subject to
   the minimum tagged-voxel threshold.
3. Refused nucleus edges are logged to their own cut file, mirroring `sem_cuts.data`.
4. Nucleus ids are `uint32`, not `uint8`.
5. With `NUC_PATH` absent, output is unchanged from baseline.
6. The build succeeds inside `work/abiss` and the verification evidence is real command
   output, not asserted.
=========================== END task.md ===========================

=========================== BEGIN artifacts/plan_v3.md (superseded, for reference) ===========================
# Plan v3

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, and use it as
a cannot-link constraint during agglomeration.

Plan_v2 introduced a three-state record but kept two things that the reviewer broke: an
**absolute** evidence floor, whose discards accumulate across supervoxels and reopen the
two-nucleus merge by a second route (H2), and a **detect-after-the-fact** response to conflicting
remaps, which records an invalid coalescence instead of preventing it (H3).

Plan_v3 changes both, and both changes make the design smaller rather than larger:

1. **The absolute floor is replaced by a dominance ratio.** A supervoxel with *any* nucleus
   evidence now always gets a state — PROPER if one id holds at least `dominance_ratio` of the
   tagged voxels, CONFLICT otherwise. Nothing is silently dropped to "no evidence", so H2's
   accumulation scenario cannot form. This also yields a stateable, provable contamination bound
   (Bound C below) in place of plan_v2's hand-waving. The ratio and its 0.6 default mirror
   `agglomeration_semantic_heuristic_t::dominant_signal_ratio` (`mean_aggl.cpp:132`), so the
   feature is consistent with the heuristic already in this file.
2. **Conflicting remaps abort instead of resolving to CONFLICT.** The reviewer is right that
   converting to TOP after a many-to-one remap commits an already-invalid result, and that
   plan_v1's `abort()` was the correct response to an unreachable-by-construction violation.
   Plan_v3 splits the policy explicitly: `nuc_join` keeps pure lattice semantics where mixed
   evidence is *legitimate* (extraction, cross-chunk evidence for one supervoxel), and the sites
   where a conflict can only mean a broken veto (`reduce_chunk`, `match_chunks`, merge
   propagation) abort loudly.

Plus three corrections the reviewer is straightforwardly right about: the idempotence claim is
retracted (H1), TOP may merge with BOTTOM (H4), and the id domain is kept whole by moving the
state into its own field instead of reserving `0xFFFFFFFF` (H5).

The guarantee, stated precisely:

> **Invariant N''.** No merge performed by agglomeration joins two clusters whose recorded states
> name different nuclei, or joins a CONFLICT cluster to anything carrying nucleus evidence.
>
> **Bound C.** In any cluster recorded PROPER with id *i*, the number of voxels tagged with any
> *j != i* is at most `(1 - dominance_ratio) * total`.
>
> Neither statement covers contamination *within a single watershed supervoxel*: that fusion
> predates agglomeration, is not repairable there, and is instead isolated (the supervoxel becomes
> CONFLICT) and counted.

## Scope

**In scope** (`lib/abiss`, worked in the isolated clone `work/abiss`): the record and wire types;
`nuc_join`; `NucExtractor` with dominance-based resolution and diagnostics; propagation through
the distributed hierarchy including OVERLAP=2 veto feedback; the veto in `mean_aggl.cpp`;
`NUC_PATH` plumbing with dtype/range/alignment contracts; and executable tests covering the
default path, the veto, the resolution rule, malformed input, the reviewer's H2 accumulation
scenario, and the hierarchy binaries.

**Out of scope, with reasons:** `NUC_WS` (a recall optimization once CONFLICT contains the damage;
the A4 counters size it first); `NUC_MIP` (replaced by the C5 shape assertion); the
pytorch_connectomics wrapper; generating the mask; perinuclear-shell tagging;
`scripts/reduce_chunk.py` and `scripts/match_chunks.py`, which are **dead legacy** —
`overlap_chunk_me.sh:44` invokes the C++ `$BIN_PATH/reduce_chunk` and `composite_chunk_me.sh:42`
the C++ `$BIN_PATH/match_chunks`.

## Proposed Changes

### Phase A — types, algebra, extraction

**A1. `src/seg/Types.h`** — beside `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                       // voxel dtype of nuc.raw; 0 == background

enum : uint8_t {
    NUC_STATE_NONE     = 0,   // BOTTOM: no nucleus evidence at all (total == 0)
    NUC_STATE_PROPER   = 1,   // one id holds >= dominance_ratio of the tagged voxels
    NUC_STATE_CONFLICT = 2,   // TOP: no id is dominant
};

struct __attribute__((packed)) nuc_record_t {
    uint8_t  state = NUC_STATE_NONE;
    uint32_t id    = 0;   // meaningful only when state == PROPER
    uint64_t count = 0;   // voxels backing `id`; 0 unless PROPER
    uint64_t total = 0;   // voxels carrying any nonzero id; DIAGNOSTIC ONLY, never decides
};
```

The state lives in its own field, so **the full `uint32` id domain `[1, 0xFFFFFFFF]` stays usable**
(H5). Plan_v2 reserved `0xFFFFFFFF` as a sentinel and thereby rejected a legitimate mask value;
that regression is retracted. Only `0` is reserved, which is inherent to a background label.

`total` must never be read by a decision — comment it so a later change cannot quietly promote it.

**A2. Wire format.** `std::pair` is not a layout contract. Define the record explicitly and use it
at every producer and consumer:

```cpp
struct __attribute__((packed)) nuc_wire_t {
    seg_t    sid;      // offset 0
    uint8_t  state;    // offset 8
    uint32_t id;       // offset 9
    uint64_t count;    // offset 13
    uint64_t total;    // offset 21
};
static_assert(sizeof(nuc_wire_t) == 29);
static_assert(offsetof(nuc_wire_t, sid)   == 0);
static_assert(offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id)    == 9);
static_assert(offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);
```

Matching numpy dtype, stated here so fixtures cannot drift from the binary (little-endian, the
only target this code runs on):

```python
NUC_WIRE = numpy.dtype([('sid','<u8'), ('state','u1'), ('id','<u4'),
                        ('count','<u8'), ('total','<u8')], align=False)
assert NUC_WIRE.itemsize == 29
```

**A3. `nuc_join` — one shared combination operation, with its algebra stated correctly (H1).**

```cpp
inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t r;
    r.total = a.total + b.total;
    if (a.state == NUC_STATE_CONFLICT || b.state == NUC_STATE_CONFLICT) {
        r.state = NUC_STATE_CONFLICT;
    } else if (a.state == NUC_STATE_NONE) {
        r.state = b.state; r.id = b.id; r.count = b.count;
    } else if (b.state == NUC_STATE_NONE) {
        r.state = a.state; r.id = a.id; r.count = a.count;
    } else if (a.id == b.id) {
        r.state = NUC_STATE_PROPER; r.id = a.id; r.count = a.count + b.count;
    } else {
        r.state = NUC_STATE_CONFLICT;
    }
    return r;
}
```

**Algebraic properties, corrected.** Plan_v2 claimed full idempotence; that is false, as the
reviewer showed with `a = (PROPER, 1, 5, 5)` giving `nuc_join(a,a) = (PROPER, 1, 10, 10)`.
The accurate statement:

* the **state/id projection** is the join of a flat lattice (BOTTOM `NONE`, TOP `CONFLICT`), hence
  associative, commutative, **and** idempotent on that projection;
* `count` and `total` form an additive commutative monoid, hence associative and commutative but
  **not** idempotent;
* therefore `nuc_join` on the full record is **associative and commutative, and not idempotent** —
  which is correct and sufficient, because the pipeline never presents the same evidence twice.
  Every combination site joins records derived from *disjoint* voxel sets.

V5 tests exactly this and no longer demands an unsatisfiable property.

**A4. `src/seg/NucExtractor.hpp` (new).** Modeled on `SemExtractor.hpp`; no class LUT; nullable
source so the A5 call site does not branch.

* Constructor takes `const Chunk *` (may be `nullptr`) and `dominance_ratio`.
* `collectVoxel(Coord c, Tseg segid)`: return if null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]` and, if `id != 0`, increment `m_counts[segid][id]`, where
  `m_counts` is `MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`. Untagged supervoxels never
  enter the outer map.
* `collectBoundary`, `collectContactingSurface`: empty.

`output(chunkMap, filename)` remaps supervoxel ids through `chunkMap` and merges the id→count maps
of supervoxels sharing a target, as `SemExtractor::output` does (`SemExtractor.hpp:31-51`). This is
the only place raw per-id voxel evidence is aggregated. Then, per target supervoxel:

```text
total := sum of all counts
if total == 0:                                    (NONE,     -,   0,         0)
(max_id, max_count) := largest count, ties -> smaller id      # deterministic, see below
if max_count >= dominance_ratio * total:          (PROPER,   max_id, max_count, total)
else:                                             (CONFLICT, -,   0,         total)
```

**This is the H2 fix.** Plan_v2 discarded a supervoxel's evidence entirely when it fell below an
absolute floor, recording NONE; the reviewer showed two such NONE supervoxels can accumulate 120
voxels of nucleus 1 and then legally merge with a nucleus-2 cluster. Under the dominance rule, a
supervoxel with *any* nucleus evidence never becomes NONE — A (60 voxels of id 1, dominance 1.0)
is PROPER id 1 — so the scenario cannot form. V9 tests the reviewer's exact case.

**Bound C follows directly**, and is the honest replacement for plan_v2's absolute-floor
hand-waving. Each supervoxel recorded PROPER *i* holds at most `(1 - dominance_ratio) * total_sv`
voxels of any other nucleus, by the rule above. NONE clusters hold zero tagged voxels. Merges join
only equal-id or NONE (Invariant N''), and both `count` and `total` sum, so the per-supervoxel
bound sums to the cluster bound. CONFLICT clusters make no claim.

The tie-break "smaller id" matters: `MapContainer` may be `absl::flat_hash_map`
(`CMakeLists.txt:42-49` defines `USE_ABSL_HASHMAP` when abseil is found), whose iteration order is
not stable, and a nondeterministic winner would break this pipeline's reproducibility. Note this
argmax runs **once**, inside `output()`, over complete evidence — no downstream stage ever selects
a winner, which is why `nuc_join` needs no ordering assumptions.

`dominance_ratio` comes from `ABISS_NUC_DOMINANCE` (default `0.6`), parsed in
`atomic_chunk_ME.cpp` `main`. An optional absolute floor `ABISS_NUC_MIN_VOXELS`
(**default 0, i.e. disabled**) drops ids below it *before* the totals are computed; the README must
state that raising it above 0 reintroduces exactly the accumulation leak H2 describes and voids
Bound C. That is the answer to reviewer question 1: the floor is per-supervoxel, its discards do
accumulate, and it is therefore off by default.

`output()` **always creates the file**, writing zero records when the source was null (B5).

Counters printed with a stable `nuc:` prefix so tests can grep them: supervoxels resolved
CONFLICT; supervoxels where `total > count` (minority evidence inside the dominance allowance);
and, if `ABISS_NUC_MIN_VOXELS > 0`, voxels discarded by the floor.

**A5. `src/seg/atomic_chunk_ME.cpp`.** The file duplicates the whole `traverseSegments<1>(...)`
call for the sem-present and sem-absent cases (lines 78-104). `NucExtractor` being nullable keeps
that at two branches:

* Declare the `bio::mapped_file_source` for `nuc.raw` in the same scope as `seg_file`/`aff_file`
  so it outlives the traversal. If `std::filesystem::exists("nuc.raw")`, open it and build
  `ConstChunkRef<nuc_t, 3>` with the same extents and `fortran_storage_order()` as `sem.raw`
  (lines 79-84); else leave the pointer null.
* **Unconditional** size check, never `assert`: if the mapped size differs from
  `sizeof(nuc_t)*dim[0]*dim[1]*dim[2]`, print expected and actual bytes to `stderr` and
  `std::abort()`.
* Add `nuc_extractor` to both packs; call `nuc_extractor.output(map, "ongoing_nuclei_labels.data")`
  unconditionally after the branch.
* Parse `ABISS_NUC_DOMINANCE` and `ABISS_NUC_MIN_VOXELS` here; reject a dominance ratio outside
  `(0.5, 1.0]` with a clear message (at or below 0.5 two ids could both be "dominant").

### Phase B — agglomeration

**B1. No new agglomeration parameters.** Nothing is added to `agglomeration_param_t`;
`heuristics_aff_threshold` (line 148) is untouched. The veto is parameterless — all tuning lives
at extraction.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156) and
`load_nuc(...)` beside `load_sem` (line 245), called next to line 545 on
`ongoing_nuclei_labels.data`, reading `nuc_wire_t`. Mirror `load_sem`'s index-mapping structure;
an empty or missing file yields an empty vector as `load_sem` does (lines 248-251), so every
nucleus path is skipped when the feature is unused.

Duplicate sids combine with `nuc_join`, **CONFLICT allowed**. This site is legitimate mixed
evidence, not a broken veto: `ongoing_nuclei_labels.data` is concatenated across chunks by
`merge_intermediate_outputs`, so one supervoxel spanning a chunk boundary contributes one record
per chunk over disjoint voxel sets. If those records name different ids, the supervoxel genuinely
straddles two nuclei and CONFLICT is the correct answer.

**Where combination happens, and under which policy** — the H3 split, stated so the implementer
cannot apply the wrong one:

| Site | Operation | Conflict policy | Why |
|---|---|---|---|
| `NucExtractor::output` | aggregate raw counts, resolve (A4) | CONFLICT is a valid outcome | watershed genuinely fused two nuclei |
| `load_nuc` duplicates | `nuc_join` | CONFLICT is a valid outcome | disjoint cross-chunk evidence for one supervoxel |
| `reduce_chunk` remap collision | check | **abort** | remap is agg-derived; a conflict means the veto failed |
| `match_chunks` remap collision | check | **abort** | same |
| merge propagation (B4) | `nuc_join` | **abort** on differing proper ids | B3 makes it unreachable |

**B3. The veto.**

```cpp
inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    const bool ca = (a.state == NUC_STATE_CONFLICT), cb = (b.state == NUC_STATE_CONFLICT);
    if (ca && cb) return false;                                  // two contaminated clusters
    if (ca)       return b.state == NUC_STATE_NONE;              // H4: TOP may absorb BOTTOM
    if (cb)       return a.state == NUC_STATE_NONE;
    if (a.state == NUC_STATE_NONE || b.state == NUC_STATE_NONE) return true;
    return a.id == b.id;                                         // absolute, no threshold escape
}
```

**H4 fix:** plan_v2 refused every merge touching CONFLICT, which orphaned a conflicted supervoxel
even when its only neighbours were ordinary untagged cytoplasm. The reviewer is right that this is
a containment *policy*, not a requirement — `CONFLICT ⊕ NONE = CONFLICT`, so absorbing untagged
material introduces no new nucleus identity and the cluster still refuses every proper id
afterwards. Allowing it avoids avoidable over-segmentation and shrinks the cut stream. Note that
under A4, NONE now means "zero nucleus evidence", which is what makes this provably safe.

Call site: in `agglomerate_cc`'s main loop, immediately after the frozen-edge block ending at line
697 and before the semantic check at line 699, guarded by `if (!nuc_ids.empty())`, with **no
affinity gate**. On refusal push the edge to a new `nuc_rg_vector` in `agglomeration_output_t`
(line 167), set `e.edge->w = Limits::min()`, `continue` — the shape of the semantic refusal at
lines 700-706. A nucleus refusal is logged in preference to a semantic one.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687.** That condition
already reads `!sem_counts.empty()` and changes behavior when a semantic payload is present;
replicating it for nuclei would make enabling the feature silently alter frozen-edge handling.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), same `v0`/`v1`/`s`
swap discipline, guarded by `!nuc_ids.empty()`: combine with `nuc_join`. Before combining, if both
records are PROPER with different ids, print both ids and both `seg_indices` values to `stderr`
and `std::abort()`. Plan_v2 downgraded this to a warning-plus-CONFLICT; the reviewer is right that
this commits an invalid result, and that plan_v1's abort was correct — B3 makes the state
unreachable, so reaching it is a code defect, and a loud failure beats a silently wrong
segmentation.

**B5. File lifecycle.** Creation and content are separate concerns:

* **Creation is unconditional.** `agg` opens and closes `ongoing_nuc.data`, `done_nuc.data`, and
  `nuc_cuts.data` every run regardless of `nuc_ids.empty()`; `NucExtractor::output` always creates
  `ongoing_nuclei_labels.data`. Required: the drivers `mv` these under `set -euo pipefail`, and
  `reduce_chunk.cpp`'s `read_array` aborts on a file it cannot open.
* **Content is conditional** on `!nuc_ids.empty()`.

Mirror the ongoing/done split at lines 848-857 and 880-891, and `of_sem_cuts` at 960-961 and
1050-1052.

**B6. Diagnostics.** All counters printed with a `nuc:` prefix: extraction CONFLICTs; extraction
`total > count`; floor-discarded voxels; `load_nuc` duplicate combinations producing CONFLICT.
Merge-propagation and remap conflicts do not get counters — they abort.

### Phase C — hierarchy and drivers

**C1. `src/seg/reduce_chunk.cpp`.** Add a nucleus reducer reading `nuc_wire_t`, applying the `sid`
remap from `remap.data`. When two source records remap onto the same target sid: combine with
`nuc_join` **only if that cannot mask a veto failure** — concretely, if both are PROPER with
different ids, print both sids and ids and `abort()` (H3); otherwise `nuc_join`. Emit one record
per sid. Call from line 228 with the `ongoing_nuclei_labels_%1%.data` /
`reduced_ongoing_nuclei_labels_%1%.data` pair. An absent or empty input must still produce the
output file.

Plan_v1 claimed a pure `sid` remap sufficed because that is what `reduce_sem` does; that is true
for a summable histogram and false for an id. Plan_v2 collapsed collisions to CONFLICT, which
records the fusion rather than preventing it. Plan_v3 aborts, because these remaps come from
`agg`'s own `remap.data` — merges that already passed B3 — so a conflicting collision is a
software-invariant violation, not a data condition.

**C2. `src/seg/match_chunks.cpp`.** Mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data`, writing `ongoing_nuclei_labels.data`, with
C1's collision policy.

**C3. OVERLAP=2 veto feedback.** Semantic cuts are re-applied across the overlap boundary:
`overlap_chunk_me.sh:50` copies `sem_cuts.data` to `vetoed_edges_<chunk>.data`,
`merge_chunks_me.py:62` merges them, `match_chunks.cpp:190-238` removes those edges from the region
graph. Without the equivalent, a nucleus veto made in one round is silently forgotten in the next.
At `overlap_chunk_me.sh:50`, after the existing `cp`, add
`cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. The consumer already sorts and dedups
(`match_chunks.cpp:212-214`) and the record is the same `(seg_t, seg_t)` pair, so no C++ change is
needed. T3 executes this chain.

**C4. Shell drivers.** Mirror every `ongoing_semantic_labels` / `done_sem` / `sem_cuts` line:

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40, plus the same three moves at
  the line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), the
  `done_nuc.data` move (line 64), the `nuc_cuts.data` move (line 66), and the C3 `cat`.

B5's unconditional creation is what makes every `mv` safe with no nucleus input.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"`; **not** `NUC_MIP`. The existing loop is
  `for e in env: if e in data:`, so a JSON without `NUC_PATH` exports nothing new; comment it.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49), add the `NUC_PATH` block:
  `load_data(global_param['NUC_PATH'], mip=global_param['AFF_RESOLUTION'],
  fill_missing=global_param.get('NUC_FILL_MISSING', False))`, then `cut_data` with the same
  `start_coord`/`end_coord` used for `seg.raw`, then validate, then `save_raw_data("nuc.raw", ...)`.

  **Alignment contract (replaces `NUC_MIP`):** read at `AFF_RESOLUTION`, the same mip as affinity
  and watershed, because ids must align voxel-for-voxel with `seg.raw`. Assert the nucleus cutout's
  spatial shape equals the `seg.raw` cutout's shape; raise otherwise.

  **Value and dtype contract (H5 — full domain preserved):**

  ```text
  non-integer dtype (float, bool, complex)   -> raise
  signed dtype containing a negative value   -> raise
  any value > 0xFFFFFFFF                     -> raise
  otherwise                                  -> astype(numpy.uint32), lossless
  ```

  Every nonzero uint32 id is valid, including `0xFFFFFFFF`; only `0` is background. The cast is
  mandatory because `save_raw_data` writes `data.dtype` verbatim (`cut_chunk_common.py:38-50`)
  while the binary mmaps `nuc.raw` as `nuc_t`; the same element-width trap is already documented
  there for affinity (`affinity_dtype()` / `ABISS_AFF_DTYPE`).
* `scripts/merge_chunks_me.py:59`, `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"` beside `"ongoing_semantic_labels"`.

### Phase D — documentation

`README.md` section covering: the `NUC_PATH` key with dtype and alignment contracts and the full
uint32 domain; `ABISS_NUC_DOMINANCE` (default 0.6) and `ABISS_NUC_MIN_VOXELS` (default 0, with an
explicit warning that raising it reintroduces the accumulation leak and voids Bound C); the
three-state record and what CONFLICT costs; the `nuc:` counters; the `nuc_cuts` /
`nuc_rejected_edges` outputs; **Invariant N'' and Bound C stated verbatim, including what they do
not promise**; and the caveat that the mask should tag **perinuclear cytoplasm**, not raw nucleus
interiors, because the nuclear envelope is a membrane the affinity network boundaries, so an id
parked on the interior can sit on a cluster that never joins its soma.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, state enum, `nuc_record_t`, `nuc_wire_t` + static_asserts, `nuc_join` |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, dominance resolution, counters |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw` + unconditional size check, extractor in both branches, env parsing |
| `src/agg/mean_aggl.cpp` | `load_nuc`, `nuc_can_merge`, veto, `nuc_join` propagation + abort, counters, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | nucleus reducer; abort on conflicting collision |
| `src/seg/match_chunks.cpp` | `process_nucs`; abort on conflicting collision |
| `scripts/set_env.py` | export `NUC_PATH` only |
| `scripts/cut_chunk_agg.py` | `nuc.raw` with shape, dtype, range validation |
| `scripts/atomic_chunk_me.sh`, `composite_chunk_me.sh`, `overlap_chunk_me.sh` | nucleus artifacts; `cat nuc_cuts.data >> vetoed_edges_*` |
| `scripts/merge_chunks_me.py`, `merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | feature docs, Invariant N'', Bound C, perinuclear caveat |
| `work/test/` (new, outside the shipped tree) | fixtures and drivers for V1-V10 |

Untouched on purpose: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy);
`src/ws/*`; `CMakeLists.txt` (header-only addition needs no target change — confirm during
implementation, edit only if the build proves otherwise).

## Verification Plan

All commands run inside `work/abiss`. The live `lib/abiss` is never touched and no build writes to
`lib/abiss/build/`. Every claim in `code_v0.md` quotes real command output; a step not run is
reported as not run.

**V1 — build.** `cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8` compiles clean including the A2
`static_assert`s. Report any new warning.

**V2 — default-path invariance.** This repo reproduces a Seuron provenance record, so a silent
change to no-nucleus output is a hard failure.

1. Build **baseline** binaries from `run_start_ref` in a scratch worktree of the clone into a
   separate build directory; build **modified** binaries.
2. `work/test/make_fixture.py` (fixed seed) writes `aff.raw` (float32, `(x,y,z,3)`, Fortran order),
   `seg.raw` (uint64), `chunkmap.data`, `param.txt` per `atomic_chunk_ME.cpp:31-34`. No `nuc.raw`.
3. Run `acme param.txt <tag>` then `agg <thr> input_rg.data frozen.data ns.data` under both builds
   in separate directories, staging inputs exactly as `atomic_chunk_me.sh:34-41` does
   (`mv edges_<tag>.data input_rg.data`; `cat boundary_{0..5}_<tag>.data >> frozen.data`;
   `touch ns.data ongoing_semantic_labels.data ongoing_seg_size.data`).
4. **The comparison set is exactly the set of files the baseline run produced**; every one must be
   byte-identical (`cmp`).
5. Files produced only by the modified build must exist and be **empty** (assert size 0):
   `ongoing_nuclei_labels.data`, `ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`.
6. `set_env.py` on a param JSON without `NUC_PATH` exports no `NUC_` variable; `cut_chunk_agg.py`
   against a small local HDF5/zarr volume writes no `nuc.raw`.

**V3 — the veto fires.** Add a uint32 `nuc.raw` tagging two supervoxel groups ids 1 and 2, joined
by a **high-affinity** path so the unconstrained run definitely merges them; at least 2000 tagged
voxels each. Without `nuc.raw`: one segment. With: two segments and `nuc_cuts.data` non-empty
naming the expected pair. The veto fires despite the high affinity.

**V4 — pass-through.** Only one group tagged (allowed); both tagged with the same id (allowed);
one group tagged with `0xFFFFFFFF` and the other with 1 (vetoed — proves H5, that the former top
sentinel is now an ordinary id).

**V5 — the algebra, tested as stated (H1).** Over all pairs and triples drawn from
`{NONE, PROPER id1, PROPER id2, CONFLICT}` with **non-zero counts and totals**:

* `nuc_join` is commutative and associative on the **full record**, including the reviewer's
  `(id1, id2, id2)` case in both groupings;
* idempotence is asserted on the **state/id projection only**, and the test explicitly asserts
  `nuc_join(a,a).count == 2*a.count` so the non-idempotence of the additive fields is pinned rather
  than accidentally passing on zero-count records.

**V6 — resolution and barrier.**

* dominance: a supervoxel with 5000 of id1 and 100 of id2 → PROPER id1 (0.98 >= 0.6), `total>count`
  counter reports 1; with 500 of id1 and 400 of id2 → CONFLICT (0.56 < 0.6), CONFLICT counter
  reports 1.
* barrier: the CONFLICT supervoxel must **fail** to merge with a PROPER id-3 cluster and with
  another CONFLICT cluster, and must **succeed** in merging with a NONE cluster (the H4 behavior).
* `ABISS_NUC_DOMINANCE=0.5` and `=1.1` are rejected with a clear message.

**V7 — input contracts.** uint16 accepted and written at uint32 width; a value `> 0xFFFFFFFF`
raises; float32 raises; a signed mask with a negative value raises; a shape mismatch against
`seg.raw` raises. Separately, a **truncated `nuc.raw`** makes `acme` exit nonzero with the A5
message, run against the Release build (this tests the unconditional check, not the Python guard).

**V8 — hierarchy binaries, executed with concrete commands (H6).** Each test is a script under
`work/test/` whose every command carries a comment naming the driver line it replicates.

* **T1 `reduce_chunk <tag>`** — build `ongoing_nuclei_labels_<tag>.data` with the A2 numpy dtype
  plus the `remap.data`, `ongoing_segments.data`, `done_segments.data`, `residual_rg_<tag>.data`,
  and boundary inputs the binary reads. Assert `sid`s are remapped and payloads preserved. Then a
  **conflicting many-to-one case**: two PROPER records with ids 1 and 2 remapped onto one target
  sid must make the binary **exit nonzero** with the C1 abort message. Plan_v2 asserted a TOP
  record here; per H3 that would be asserting the bug.
* **T2 `match_chunks <tag>`** — same construction against `o_ongoing_nuclei_labels.data`, including
  the same conflicting case expecting a nonzero exit.
* **T3 the real cut chain.** Run `agg` on a nucleus-conflicting fixture so it *produces*
  `nuc_cuts.data`. Then execute the C3 append **by extracting the literal line from the driver**
  rather than retyping it — e.g. `eval "$(grep -F 'cat nuc_cuts.data' scripts/overlap_chunk_me.sh)"`
  with `output_chunk` set — so the test cannot drift from the shipped driver. Then run the
  `merge_chunks_me.py` merge step for `vetoed_edges`, then `match_chunks <tag>`, and assert the
  vetoed edge is absent from the resulting region graph.
* **T4 two-chunk end-to-end, mandatory, with concrete commands.** `work/test/run_hierarchy.sh`
  builds two adjacent synthetic chunks with different nucleus ids and runs, per chunk:
  `acme param.txt <tag>`; `mv edges_<tag>.data input_rg.data`;
  `cat boundary_{0..5}_<tag>.data >> frozen.data`; `touch ns.data ongoing_semantic_labels.data
  ongoing_nuclei_labels.data ongoing_seg_size.data`;
  `agg <thr> input_rg.data frozen.data ns.data`; `split_remap chunk_offset.txt <tag>`;
  `assort <tag> <META>` — mirroring `atomic_chunk_me.sh:34-53`. Then the composite level:
  `python3 scripts/merge_chunks_me.py <json> <META>`; `mv ongoing.data localmap.data`;
  `mv residual_rg.data input_rg.data`; `meme <tag> <META>`;
  `cat new_edges.data >> input_rg.data`;
  `agg <thr> input_rg.data frozen.data ongoing_supervoxel_counts.data` — mirroring
  `composite_chunk_me.sh:37-70`. `$UPLOAD_CMD`/`$DOWNLOAD_CMD` are set to `cp -r` and
  `FILE_PATH`/`IO_SCRATCH_PATH` to local directories; `acquire_cpu_slot`/`release_cpu_slot` and the
  tar/upload steps are omitted as they are transport, not computation.

  Assert the two tagged groups remain separate in the final remap and that the constraint survives
  the second hierarchy level. **If T4 cannot be run, the run is `Status: blocked` for human
  decision — not a passing run with a caveat.**

**V9 — the H2 accumulation scenario, tested directly.** Build the reviewer's exact case:
supervoxels A and B each carrying 60 voxels of id 1, and C carrying 100 voxels of id 2, with
affinities that would merge A-B then AB-C. Assert A and B are both recorded PROPER id 1 (not NONE),
that A-B merges, and that **AB-C is vetoed**. Then re-run with `ABISS_NUC_MIN_VOXELS=100` and
assert the merge *is* allowed, documenting that the optional floor reintroduces the leak exactly as
the README warns.

**V10 — driver syntax.** `bash -n` every modified shell script.

## Risks and Questions

**R1 — Invariant N'' and Bound C are weaker than "hard cannot-link", deliberately.** Contamination
inside a single watershed supervoxel predates agglomeration. Plan_v3 contains it (CONFLICT) and
counts it. The README must say so; V6's barrier test keeps the containment honest.

**R2 — `ABISS_NUC_DOMINANCE = 0.6` is unmeasured.** Too low admits genuinely mixed supervoxels as
PROPER (weakening Bound C); too high turns ordinary mask bleed into CONFLICT barriers
(over-segmentation). It matches the semantic heuristic's existing default, which is precedent, not
evidence. The A4 counters make the effect measurable on the first real run.

**R3 — three abort sites are new hard-failure paths in long distributed runs.** All three are
unreachable if B3 is correct, which is exactly why they abort rather than paper over. But a latent
defect that previously produced a quietly wrong segmentation will now kill a multi-hour job. That
is the intended trade and should be stated in the README.

**R4 — the optional absolute floor is a documented footgun.** Default 0 keeps Bound C valid; any
nonzero value voids it. V9 tests both directions so the behavior is pinned rather than assumed.

**R5 — V2 proves invariance on a synthetic fixture, not real EM data.** A full Seuron provenance
re-run is out of scope and must be stated as such in `code_v0.md`.

**R6 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per
tagged supervoxel; negligible when nuclei cover a small voxel fraction. Worth a code comment.

**R7 — this is the terminal plan version under `p3` and it is unreviewed.** Three review rounds
each found real defects; the findings have narrowed (round 3 produced two design changes, two
contract corrections, and one documentation error) but the trend has not reached zero.

**Q1 — should `nuc_can_merge` also refuse CONFLICT-CONFLICT?** Plan_v3 refuses it, on the grounds
that two contaminated clusters merging compounds contamination with no benefit. Unlike
CONFLICT-NONE this has no over-segmentation cost, since both sides are already isolated from
proper ids.

**Q2 — is the dominance rule the right shape for real nucleus masks?** It assumes contamination is
a small fraction of a supervoxel's tagged voxels. For a perinuclear-shell tagging (the intended
usage) that holds. For raw nucleus-interior tagging it may not, which is another reason the README
must push users toward the shell.

## Changes Since Previous Plan Version

Responds to `plan_v2_review.md`. All six findings accepted.

* **H1 (false idempotence claim) — retracted and corrected.** A3 now states the accurate algebra:
  the state/id projection is a flat-lattice join (associative, commutative, idempotent); `count` and
  `total` form an additive monoid (associative, commutative, **not** idempotent); so the full
  record is associative and commutative only. That is sufficient because every combination site
  joins records over disjoint voxel sets. V5 tests the properties that hold and explicitly pins
  `nuc_join(a,a).count == 2*a.count` so the non-idempotence cannot pass accidentally on zero-count
  records.
* **H2 (sub-floor evidence accumulates) — fixed by replacing the absolute floor with a dominance
  ratio.** Under A4 a supervoxel with any nucleus evidence is never NONE, so the reviewer's
  `A(60 of id1) + B(60 of id1) -> NONE, then merge with C(id2)` cannot form: A and B are both
  PROPER id 1 and the AB-C merge is vetoed. This also produces **Bound C**, a provable cluster-level
  contamination bound that plan_v2 could not state. The absolute floor survives only as an opt-in
  `ABISS_NUC_MIN_VOXELS` defaulting to **0**, documented as voiding Bound C. V9 tests the
  reviewer's exact scenario in both configurations. Reviewer question 1 answered: yes, the floor is
  per-supervoxel and its discards accumulate, which is why it is off by default.
* **H3 (detection is not prevention) — fixed by aborting instead of resolving.** The reviewer is
  right that plan_v1's abort was correct for an unreachable invariant violation. B2's table now
  splits the policy explicitly: `nuc_join` keeps pure lattice semantics where mixed evidence is
  legitimate (extraction, cross-chunk evidence for one supervoxel), while `reduce_chunk`,
  `match_chunks`, and merge propagation **abort** on conflicting proper ids, because those remaps
  derive from `agg`'s own veto-guarded output. T1/T2 now assert a nonzero exit rather than
  asserting the TOP record, which as the reviewer noted was asserting the bug.
* **H4 (TOP-BOTTOM isolation unnecessary) — accepted.** B3 now allows CONFLICT to merge with NONE.
  Since A4 makes NONE mean "zero nucleus evidence", absorbing it introduces no identity and the
  cluster still refuses every proper id. This removes the avoidable orphaning and shrinks the cut
  stream. Reviewer question 3 answered. CONFLICT-CONFLICT remains refused (Q1).
* **H5 (reserved sentinel breaks the uint32 domain) — fixed by a separate state field.** A1 moves
  the state into `uint8_t state`, so every nonzero uint32 id including `0xFFFFFFFF` is valid; C5
  rejects only values above the uint32 range. V4 tests `0xFFFFFFFF` as an ordinary id. Reviewer
  question 2 answered: no reservation is needed and none is taken.
* **H6 (verification not executable) — fixed by rewriting V8 with concrete commands.** T1/T2 name
  the exact binary invocation and every input file, and assert the abort. T3 extracts the C3 append
  line **from the driver itself** so the test cannot drift from shipped code. T4 lists the full
  per-chunk and composite command sequences with the driver lines they mirror, the `cp -r`
  substitution for `$UPLOAD_CMD`/`$DOWNLOAD_CMD`, and the expected second-level artifact, and
  remains mandatory. V5 no longer demands the unsatisfiable property. V9 is new and covers the
  accumulation case that had no test.
=========================== END artifacts/plan_v3.md ===========================

=========================== BEGIN artifacts/plan_v3_review.md (your prior review) ===========================
# Plan v3 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v3_review.review.raw.md`.

`READY: no`, five findings, all `[major]`. Reviewer verdict: "H1 and H5 are fixed. H4 is fixed only
when the optional floor is disabled. H2, H3, and H6 are not fully fixed. At the default settings,
Bound C's arithmetic induction is sound, but it is too weak to establish the requested cannot-link
guarantee."

This round is different in kind from the previous three. Rounds 1-3 found defects that a better
plan could fix. Round 4 identifies a **representational impossibility**: with a fixed-width
record that stores one dominant id, no parameter setting delivers both noise tolerance and exact
cannot-link. The reviewer states it directly — "The current representation offers no setting that
provides both noise tolerance and the claimed guarantee" — and demonstrates both horns.

I1 is the load-bearing finding and is not a planning error that plan_v4 can absorb; it is a
question about what the feature promises. I3, I4, and I5 are ordinary defects with clear fixes.

## Findings

**I1 [major] — Invariant N'' permits an agglomeration merge joining distinct nuclei, even with the
floor disabled.** With `dominance_ratio = 0.6`:

```text
A: 60 voxels id1, 40 voxels id2  -> PROPER id1
B: 60 voxels id1, 40 voxels id3  -> PROPER id1
```

B3 allows `A + B` because both records name id1, yet the merge newly joins material from nuclei 2
and 3. The plan's disclaimer about contamination within a single watershed supervoxel does not
cover this cross-supervoxel fusion.

Bound C still holds exactly after the merge — 80 minority voxels equals `(1 - 0.6) * 200` — so
**Bound C is mathematically compatible with violating the cannot-link requirement.** Invariant N''
protects only recorded *dominant* identities, not all nucleus identities the clusters carry.

**I2 [major] — H2 is fixed only at the default; the supported positive floor invalidates Bound C
and reopens the original leak.** With `ABISS_NUC_MIN_VOXELS = 100`, a supervoxel of `100 id1 +
99 id2` drops id2 before `total` is computed, producing `PROPER id1, count=100, total=100`. The
actual minority count of 99 exceeds `(1 - 0.6) * 100 = 40`, and the record contradicts A1's own
definition of `total` as all nonzero-tagged voxels. The original H2 case remains reachable exactly
as V9 concedes. *"A warning that an advertised option voids the guarantees is not a fix."*

The same option makes the new CONFLICT-NONE allowance unsafe:

```text
X: 100 id1 + 100 id2  -> CONFLICT
Y: 99 id3             -> NONE (after the floor)
```

B3 allows `X + Y`, joining a CONFLICT cluster to real nucleus evidence. Plan_v2's full barrier
would have rejected this, so H4's fix is a **regression** in this configuration.

Conversely with the floor disabled, a single erroneous tagged voxel in an otherwise untagged
supervoxel makes it PROPER and creates an absolute veto — failing the plan's own stated
requirement that a handful of misassigned voxels must not hard-split a cell.

**I3 [major] — H3 is incomplete: the abort sites cover only differing PROPER ids, not every merge
B3 forbids.** `CONFLICT + PROPER id7` and `CONFLICT + CONFLICT` are equally forbidden by B3, and a
collision of either kind is equally evidence the veto was bypassed. C1/C2 instead apply `nuc_join`
and silently emit CONFLICT, and B4 aborts only on two differing PROPER records. Each invariant
check should reject **every pair for which `nuc_can_merge` is false**, rather than duplicating one
branch of that predicate.

**I4 [major] — the `match_chunks` abort is reachable on legitimate data.** B2 itself recognizes
that disjoint chunk portions of one boundary-spanning supervoxel can resolve to different PROPER
ids, and that coalescing them should yield CONFLICT. The same data reaches C2 when boundary
matching canonicalizes two chunk-local sids to one. C2 would abort merely because the identity
reconciliation happens in `process_nucs` rather than as duplicate input to `load_nuc`. No proof or
provenance rule distinguishes the two cases. As specified this is a **data-reachable
denial-of-service** on a long distributed run, not an invariant check. C1 may be justified for a
purely agg-derived remap; C2 is not established.

**I5 [major] — H6 remains unfixed; V8 is still not executable without reconstructing the hierarchy
protocol.** Specifically:

* V1 gives `cmake ..` but never creates or enters a build directory.
* T1/T2 list required files but not their binary layouts, contents, boundary filenames, or
  complete fixture construction; the coder must derive them from the binaries.
* T3 deliberately does not execute `overlap_chunk_me.sh`; evaluating a grepped `cat` line cannot
  test its working directory, surrounding file creation, variable setup, or control flow. The
  subsequent `merge_chunks_me.py` invocation, arguments, JSON, and staging remain unspecified.
* T4 stops the atomic sequence around line 53 and **omits C4's essential later move**
  `ongoing_nuc.data -> ongoing_nuclei_labels_<chunk>.data`, so the merge step it is meant to test
  lacks the per-chunk nucleus payload entirely.
* T4 leaves `<tag>`, `<META>`, `<json>`, directory topology, chunk offsets, and fixture formats
  undefined.
* V5 asks for records drawn from a set including NONE and CONFLICT "with non-zero counts and
  totals", contradicting A1's canonical NONE record and CONFLICT's required zero count.

## Questions

1. Is the intended guarantee being reduced from "all distinct mask ids cannot merge" to "different
   recorded dominant ids cannot merge"? The dominance representation can only provide the latter.
2. Should weak evidence remain represented in a separate state or exact identity set, so it can
   avoid single-voxel hard vetoes without becoming NONE?
3. What proves that C2's many-to-one mappings are exclusively B3-vetted agglomeration remaps rather
   than legitimate cross-chunk identity reconciliation?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v3_review.md ===========================

=========================== BEGIN artifacts/plan_v4.md (UNDER REVIEW) ===========================
# Plan v4

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, and use it to
guide agglomeration.

Plan_v3_review established that a fixed-width record holding one dominant id **cannot** deliver
exact cannot-link (finding I1), and that no floor setting gives both noise tolerance and that
guarantee. The user decided the contract on 2026-07-30: **ship the weaker, honestly-stated
guarantee** rather than widen the record to identity sets or take on the watershed path.

Plan_v4 therefore stops claiming what the representation cannot do:

> **Invariant D (dominant-id cannot-link).** No merge performed by agglomeration joins two clusters
> whose recorded dominant nucleus ids differ, or joins a CONFLICT cluster to a cluster carrying a
> recorded dominant id.
>
> **What Invariant D does not promise.** It constrains *recorded dominant* identities, not every
> nucleus id present in a cluster's voxels. Minority evidence inside the dominance allowance is
> not tracked, so the following merge is permitted and is a known limitation, not a bug:
>
> ```text
> A: 60 voxels id1, 40 voxels id2  -> PROPER id1
> B: 60 voxels id1, 40 voxels id3  -> PROPER id1     # A+B is allowed, joining nuclei 2 and 3
> ```
>
> It also does not cover contamination within a single watershed supervoxel, which predates
> agglomeration; such supervoxels are recorded CONFLICT and counted.

That limitation is acceptable for the intended usage — perinuclear-shell tagging, where a
supervoxel's tagged voxels come from one nucleus — and pathological only for masks where a single
supervoxel is ~40% contaminated by a *different* nucleus than its neighbour is. The README must
carry the counterexample verbatim so nobody reads the feature as an absolute guarantee.

**Bound C** is retained as a genuine quantitative property, with its status corrected: it bounds
minority *mass*, and the reviewer correctly showed it does **not** imply cannot-link. It is
reported as a diagnostic bound, never as a safety argument.

The remaining three findings are fixed mechanically: the invariant check now mirrors the whole
`nuc_can_merge` predicate rather than one branch (I3); the abort is confined to the single site
where provenance is unambiguous, removing the data-reachable denial-of-service (I4); and the
verification plan is made actually runnable (I5).

## Scope

**In scope** (`lib/abiss`, in the isolated clone `work/abiss`): record and wire types; `nuc_join`;
`NucExtractor` with dominance resolution and diagnostics; propagation through the distributed
hierarchy including OVERLAP=2 veto feedback; the veto in `mean_aggl.cpp`; `NUC_PATH` plumbing with
dtype, range, and alignment contracts; documentation carrying Invariant D and its stated limits;
and executable tests.

**Out of scope, decided:** identity sets and `NUC_WS` — the user chose the dominant-id contract
over both. `NUC_WS` remains the natural follow-up if the A4 counters show mixed supervoxels are
common on real data, and the counters exist to answer exactly that. Also out: `NUC_MIP` (replaced
by the C5 shape assertion), the pytorch_connectomics wrapper, mask generation, perinuclear-shell
tagging, and `scripts/reduce_chunk.py` / `scripts/match_chunks.py`, which are **dead legacy** —
`overlap_chunk_me.sh:44` invokes the C++ `$BIN_PATH/reduce_chunk` and `composite_chunk_me.sh:42`
the C++ `$BIN_PATH/match_chunks`.

## Proposed Changes

### Phase A — types, algebra, extraction

**A1. `src/seg/Types.h`** — beside `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                       // voxel dtype of nuc.raw; 0 == background

enum : uint8_t {
    NUC_STATE_NONE     = 0,   // no usable nucleus evidence
    NUC_STATE_PROPER   = 1,   // one id holds >= dominance_ratio of the tagged voxels
    NUC_STATE_CONFLICT = 2,   // tagged, but no id is dominant
};

struct __attribute__((packed)) nuc_record_t {
    uint8_t  state = NUC_STATE_NONE;
    uint32_t id    = 0;   // meaningful only when state == PROPER
    uint64_t count = 0;   // voxels backing `id`; 0 unless PROPER
    uint64_t total = 0;   // ALL voxels carrying any nonzero id; DIAGNOSTIC ONLY, never decides
};
```

The state has its own field, so the **full uint32 id domain `[1, 0xFFFFFFFF]` stays usable**; only
`0` is reserved, which is inherent to a background label.

`total` counts every nonzero-tagged voxel with **no exception** — no filter, no floor, subtracts
nothing. Plan_v3 let the floor drop ids before computing `total`, which contradicted this
definition (I2). Comment the field so a later change cannot quietly filter or promote it.

**A2. Wire format.** `std::pair` is not a layout contract. Define it explicitly and use it at every
producer and consumer:

```cpp
struct __attribute__((packed)) nuc_wire_t {
    seg_t    sid;      // offset 0
    uint8_t  state;    // offset 8
    uint32_t id;       // offset 9
    uint64_t count;    // offset 13
    uint64_t total;    // offset 21
};
static_assert(sizeof(nuc_wire_t) == 29);
static_assert(offsetof(nuc_wire_t, sid)   == 0);
static_assert(offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id)    == 9);
static_assert(offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);
```

Matching numpy dtype for every fixture, so tests cannot drift from the binary (little-endian, the
only target this code runs on):

```python
NUC_WIRE = numpy.dtype([('sid','<u8'), ('state','u1'), ('id','<u4'),
                        ('count','<u8'), ('total','<u8')], align=False)
assert NUC_WIRE.itemsize == 29
```

**A3. `nuc_join`.**

```cpp
inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t r;
    r.total = a.total + b.total;
    if (a.state == NUC_STATE_CONFLICT || b.state == NUC_STATE_CONFLICT) {
        r.state = NUC_STATE_CONFLICT;
    } else if (a.state == NUC_STATE_NONE) {
        r.state = b.state; r.id = b.id; r.count = b.count;
    } else if (b.state == NUC_STATE_NONE) {
        r.state = a.state; r.id = a.id; r.count = a.count;
    } else if (a.id == b.id) {
        r.state = NUC_STATE_PROPER; r.id = a.id; r.count = a.count + b.count;
    } else {
        r.state = NUC_STATE_CONFLICT;
    }
    return r;
}
```

Algebra, stated accurately: the **state/id projection** is a flat-lattice join and is therefore
associative, commutative, and idempotent on that projection; `count` and `total` form an additive
commutative monoid and are associative and commutative but **not** idempotent. So `nuc_join` on
the full record is **associative and commutative, and not idempotent** — sufficient, because every
combination site joins records derived from **disjoint voxel sets**.

**A4. `src/seg/NucExtractor.hpp` (new).** Modeled on `SemExtractor.hpp`; no class LUT; nullable
source so A5 does not branch.

* Constructor takes `const Chunk *` (may be `nullptr`), `dominance_ratio`, and `min_tagged`.
* `collectVoxel(Coord c, Tseg segid)`: return if null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]` and, if `id != 0`, increment `m_counts[segid][id]`, where
  `m_counts` is `MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`. Untagged supervoxels never
  enter the outer map.
* `collectBoundary`, `collectContactingSurface`: empty.

`output(chunkMap, filename)` remaps supervoxel ids through `chunkMap` and merges the id→count maps
of supervoxels sharing a target, as `SemExtractor::output` does (`SemExtractor.hpp:31-51`). This is
the only place raw per-id evidence is aggregated. Then, per target supervoxel:

```text
total := sum of ALL counts                                        # exact, never filtered
if total == 0:                                (NONE,     -, 0,         0)
if total <  min_tagged:                       (NONE,     -, 0,         total)   # too little to judge
(max_id, max_count) := largest count, ties -> smaller id
if max_count >= dominance_ratio * total:      (PROPER,   max_id, max_count, total)
else:                                         (CONFLICT, -, 0,         total)
```

**The floor is applied to `total`, never to individual ids** — this is the I2 fix. Plan_v3 dropped
sub-floor *ids* before summing, which made `total` a lie and broke Bound C's arithmetic. Applying
it to the aggregate keeps `total` exact and the record self-consistent; the cost is that a
supervoxel below the floor is recorded NONE, which under Invariant D is a documented recall
limitation rather than a broken guarantee.

`min_tagged` (`ABISS_NUC_MIN_TAGGED`, default **50**) exists to stop a single bleed voxel in an
otherwise untagged supervoxel from becoming PROPER and casting an absolute veto that hard-splits a
real cell — the failure plan_v3 could not avoid with a floor of 0. `dominance_ratio`
(`ABISS_NUC_DOMINANCE`, default **0.6**, matching
`agglomeration_semantic_heuristic_t::dominant_signal_ratio` at `mean_aggl.cpp:132`) handles bleed
inside an already-tagged supervoxel. Reject a ratio outside `(0.5, 1.0]` with a clear message; at
or below 0.5 two ids could both qualify.

The tie-break "smaller id" matters: `MapContainer` may be `absl::flat_hash_map`
(`CMakeLists.txt:42-49` defines `USE_ABSL_HASHMAP` when abseil is found), whose iteration order is
unstable, and a nondeterministic winner would break reproducibility. This argmax runs **once**,
over complete evidence; no downstream stage selects a winner, which is why `nuc_join` needs no
ordering assumptions.

Counters, printed with a stable `nuc:` prefix so tests can grep them: supervoxels resolved
CONFLICT; supervoxels with `total > count` (minority inside the dominance allowance); supervoxels
recorded NONE because `total < min_tagged`, plus the voxel sum they represent. The last two are the
measurement that decides whether `NUC_WS` and identity sets are worth a follow-up run.

`output()` **always creates the file**, writing zero records when the source was null (B5).

**A5. `src/seg/atomic_chunk_ME.cpp`.** The file duplicates the whole `traverseSegments<1>(...)`
call for the sem-present and sem-absent cases (lines 78-104); a nullable `NucExtractor` keeps that
at two branches.

* Declare the `bio::mapped_file_source` for `nuc.raw` in the same scope as `seg_file`/`aff_file` so
  it outlives the traversal. If `std::filesystem::exists("nuc.raw")`, open it and build
  `ConstChunkRef<nuc_t, 3>` with the same extents and `fortran_storage_order()` as `sem.raw`
  (lines 79-84); else leave the pointer null.
* **Unconditional** size check, never `assert`: if the mapped size differs from
  `sizeof(nuc_t)*dim[0]*dim[1]*dim[2]`, print expected and actual bytes to `stderr` and
  `std::abort()`.
* Add `nuc_extractor` to both packs; call `nuc_extractor.output(map, "ongoing_nuclei_labels.data")`
  unconditionally after the branch.
* Parse `ABISS_NUC_DOMINANCE` and `ABISS_NUC_MIN_TAGGED` here.

### Phase B — agglomeration

**B1. No new agglomeration parameters.** Nothing is added to `agglomeration_param_t`;
`heuristics_aff_threshold` (line 148) is untouched. All tuning lives at extraction.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156) and
`load_nuc(...)` beside `load_sem` (line 245), called next to line 545 on
`ongoing_nuclei_labels.data`, reading `nuc_wire_t`. Mirror `load_sem`'s index-mapping structure; an
empty or missing file yields an empty vector as `load_sem` does (lines 248-251), so every nucleus
path is skipped when the feature is unused. Duplicate sids combine with `nuc_join`; CONFLICT is a
legitimate outcome here, because `ongoing_nuclei_labels.data` is concatenated across chunks and one
supervoxel spanning a boundary contributes one record per chunk over disjoint voxels.

**Combination sites and their conflict policy** — revised per I4:

| Site | Operation | Conflict policy |
|---|---|---|
| `NucExtractor::output` | aggregate raw counts, resolve (A4) | CONFLICT is a valid outcome |
| `load_nuc` duplicates | `nuc_join` | CONFLICT is a valid outcome |
| `reduce_chunk` remap collision | `nuc_join` | CONFLICT is a valid outcome; **counted** |
| `match_chunks` remap collision | `nuc_join` | CONFLICT is a valid outcome; **counted** |
| merge propagation (B4) | `nuc_join` | **abort** if `!nuc_can_merge(a,b)` |

**The abort is now confined to B4** — the one site where provenance is unambiguous, because the
pair has just been tested by `nuc_can_merge` a few lines earlier in the same function. Plan_v3 also
aborted in `reduce_chunk` and `match_chunks`; the reviewer showed (I4) that `match_chunks`
legitimately canonicalizes two chunk-local sids of **one** boundary-spanning supervoxel onto one
sid, so aborting there is a data-reachable denial-of-service on a long distributed run, and nothing
in the plan distinguishes that case from an agg-derived remap. Rather than assert a provenance rule
that has not been proven, both hierarchy sites now combine safely and **count** collisions; the
counter is the observable that would reveal a genuine veto failure.

**B3. The veto.**

```cpp
inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    const bool ca = (a.state == NUC_STATE_CONFLICT), cb = (b.state == NUC_STATE_CONFLICT);
    if (ca && cb) return false;
    if (ca)       return b.state == NUC_STATE_NONE;
    if (cb)       return a.state == NUC_STATE_NONE;
    if (a.state == NUC_STATE_NONE || b.state == NUC_STATE_NONE) return true;
    return a.id == b.id;
}
```

Call it in `agglomerate_cc`'s main loop immediately after the frozen-edge block ending at line 697
and before the semantic check at line 699, guarded by `if (!nuc_ids.empty())`, with **no affinity
gate**. On refusal push the edge to a new `nuc_rg_vector` in `agglomeration_output_t` (line 167),
set `e.edge->w = Limits::min()`, `continue` — the shape of the semantic refusal at lines 700-706. A
nucleus refusal is logged in preference to a semantic one.

CONFLICT may absorb NONE. Under a nonzero `min_tagged`, a NONE cluster can hide sub-floor evidence,
so this allowance can join a contaminated cluster to weak evidence — explicitly inside what
Invariant D does not promise, and the alternative (plan_v2's full barrier) orphans conflicted
supervoxels next to ordinary cytoplasm for no gain against the contract we are shipping.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687.** That condition
already reads `!sem_counts.empty()` and changes behavior when a semantic payload is present;
replicating it would make enabling nuclei silently alter frozen-edge handling.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), same `v0`/`v1`/`s`
swap discipline, guarded by `!nuc_ids.empty()`. Before combining, evaluate
`nuc_can_merge(a, b)`; if it is **false**, print both records and both `seg_indices` values to
`stderr` and `std::abort()`. Then combine with `nuc_join`.

This is the I3 fix. Plan_v3 aborted only on two differing PROPER records, which duplicated one
branch of the predicate and let `CONFLICT + PROPER` and `CONFLICT + CONFLICT` through — pairs B3
equally forbids. Reusing the predicate itself makes the check exhaustive by construction and keeps
the two in sync if either changes.

**B5. File lifecycle.** Creation and content are separate:

* **Creation is unconditional.** `agg` opens and closes `ongoing_nuc.data`, `done_nuc.data`, and
  `nuc_cuts.data` every run regardless of `nuc_ids.empty()`; `NucExtractor::output` always creates
  `ongoing_nuclei_labels.data`. Required: the drivers `mv` these under `set -euo pipefail`, and
  `reduce_chunk.cpp`'s `read_array` aborts on a file it cannot open.
* **Content is conditional** on `!nuc_ids.empty()`.

Mirror the ongoing/done split at lines 848-857 and 880-891, and `of_sem_cuts` at 960-961 and
1050-1052.

**B6. Diagnostics.** All with a `nuc:` prefix: extraction CONFLICTs; extraction `total > count`;
extraction below-`min_tagged` supervoxels and their voxel sum; `load_nuc` duplicate combinations
producing CONFLICT; `reduce_chunk` and `match_chunks` remap collisions producing CONFLICT. Only
B4's violation aborts.

### Phase C — hierarchy and drivers

**C1. `src/seg/reduce_chunk.cpp`.** Add a nucleus reducer reading `nuc_wire_t`, applying the `sid`
remap from `remap.data`, combining collisions with `nuc_join` and counting those that yield
CONFLICT. Emit one record per sid. Call from line 228 with the `ongoing_nuclei_labels_%1%.data` /
`reduced_ongoing_nuclei_labels_%1%.data` pair. An absent or empty input must still produce the
output file.

**C2. `src/seg/match_chunks.cpp`.** Mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data`, writing `ongoing_nuclei_labels.data`, with
C1's collision handling.

**C3. OVERLAP=2 veto feedback.** Semantic cuts are re-applied across the overlap boundary:
`overlap_chunk_me.sh:50` copies `sem_cuts.data` to `vetoed_edges_<chunk>.data`,
`merge_chunks_me.py:62` merges them, `match_chunks.cpp:190-238` removes those edges from the region
graph. Without the equivalent, a nucleus veto made in one round is silently forgotten in the next.
At `overlap_chunk_me.sh:50`, after the existing `cp`, add
`cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. The consumer already sorts and dedups
(`match_chunks.cpp:212-214`) and the record is the same `(seg_t, seg_t)` pair, so no C++ change is
needed.

**C4. Shell drivers.**

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  **`mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group)** — the
  reviewer flagged (I5) that omitting this from T4 leaves the merge step without a nucleus payload,
  so it is called out here and exercised in T4.
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40, plus the same four moves at the
  line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), the
  `done_nuc.data` move (line 64), the `nuc_cuts.data` move (line 66), and the C3 `cat`.

B5's unconditional creation is what makes every `mv` safe with no nucleus input.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"`; **not** `NUC_MIP`. The existing loop is
  `for e in env: if e in data:`, so a JSON without `NUC_PATH` exports nothing new; comment it.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49), add the `NUC_PATH` block:
  `load_data(global_param['NUC_PATH'], mip=global_param['AFF_RESOLUTION'],
  fill_missing=global_param.get('NUC_FILL_MISSING', False))`, then `cut_data` with the same
  `start_coord`/`end_coord` used for `seg.raw`, then validate, then `save_raw_data("nuc.raw", ...)`.

  **Alignment contract (replaces `NUC_MIP`):** read at `AFF_RESOLUTION`, the same mip as affinity
  and watershed, because ids must align voxel-for-voxel with `seg.raw`. Assert the nucleus cutout's
  spatial shape equals the `seg.raw` cutout's shape; raise otherwise.

  **Value and dtype contract:**

  ```text
  non-integer dtype (float, bool, complex)   -> raise
  signed dtype containing a negative value   -> raise
  any value > 0xFFFFFFFF                     -> raise
  otherwise                                  -> astype(numpy.uint32), lossless
  ```

  Every nonzero uint32 id is valid, including `0xFFFFFFFF`; only `0` is background. The cast is
  mandatory because `save_raw_data` writes `data.dtype` verbatim (`cut_chunk_common.py:38-50`)
  while the binary mmaps `nuc.raw` as `nuc_t`; the same element-width trap is documented there for
  affinity (`affinity_dtype()` / `ABISS_AFF_DTYPE`).
* `scripts/merge_chunks_me.py:59`, `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"` beside `"ongoing_semantic_labels"`.

### Phase D — documentation

`README.md` section covering: the `NUC_PATH` key with dtype, range, and alignment contracts and the
full uint32 domain; `ABISS_NUC_DOMINANCE` (0.6) and `ABISS_NUC_MIN_TAGGED` (50) and what each
controls; the three-state record; the `nuc:` counters and what decision they inform; the
`nuc_cuts` / `nuc_rejected_edges` outputs; the B4 abort as a hard-failure path; and **Invariant D
reproduced verbatim including the `A: 60/40, B: 60/40` counterexample and the within-supervoxel
caveat.** Also the usage caveat that the mask should tag **perinuclear cytoplasm**, not raw nucleus
interiors, because the nuclear envelope is a membrane the affinity network boundaries, so an id
parked on the interior can sit on a cluster that never joins its soma.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, state enum, `nuc_record_t`, `nuc_wire_t` + static_asserts, `nuc_join` |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, dominance + `min_tagged` resolution, counters |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw` + unconditional size check, extractor in both branches, env parsing |
| `src/agg/mean_aggl.cpp` | `load_nuc`, `nuc_can_merge`, veto, `nuc_join` propagation + predicate-based abort, counters, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | nucleus reducer; `nuc_join` on collision + counter |
| `src/seg/match_chunks.cpp` | `process_nucs`; `nuc_join` on collision + counter |
| `scripts/set_env.py` | export `NUC_PATH` only |
| `scripts/cut_chunk_agg.py` | `nuc.raw` with shape, dtype, range validation |
| `scripts/atomic_chunk_me.sh`, `composite_chunk_me.sh`, `overlap_chunk_me.sh` | nucleus artifacts incl. the `ongoing_nuc.data` move; `cat nuc_cuts.data >> vetoed_edges_*` |
| `scripts/merge_chunks_me.py`, `merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | feature docs, Invariant D verbatim with its counterexample |
| `work/test/` (new, outside the shipped tree) | fixtures and drivers for V1-V10 |

Untouched on purpose: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy);
`src/ws/*`; `CMakeLists.txt` (header-only addition needs no target change — confirm during
implementation, edit only if the build proves otherwise).

## Verification Plan

All commands run inside `work/abiss`. The live `lib/abiss` is never touched and no build writes to
`lib/abiss/build/`. Every claim in `code_v0.md` quotes real command output; a step not run is
reported as not run.

**V1 — build.** Explicitly creating and entering the build directory (I5):

```bash
cd work/abiss && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8
```

Must compile clean including the A2 `static_assert`s. Report any new warning.

**V2 — default-path invariance.** This repo reproduces a Seuron provenance record, so a silent
change to no-nucleus output is a hard failure.

1. Build **baseline** binaries from `run_start_ref` in a scratch worktree of the clone into a
   separate build directory; build **modified** binaries.
2. `work/test/make_fixture.py` (fixed seed) writes, for a chunk of shape `(64,64,64)`:
   `aff.raw` (float32, `(64,64,64,3)`, Fortran order), `seg.raw` (uint64, Fortran order),
   `chunkmap.data` (empty), and `param.txt` whose three lines are `offset[0..2]`, `dim[0..2]`, and
   `ac_offset`, per `atomic_chunk_ME.cpp:31-34`. No `nuc.raw`.
3. Stage and run exactly as `atomic_chunk_me.sh:34-41` does, under both builds in separate
   directories: `acme param.txt <tag>`; `mv edges_<tag>.data input_rg.data`;
   `for i in 0 1 2 3 4 5; do cat boundary_${i}_<tag>.data >> frozen.data; done`;
   `touch ns.data ongoing_semantic_labels.data ongoing_seg_size.data`;
   `agg 0.25 input_rg.data frozen.data ns.data`. `<tag>` is the literal string `0_0_0_0`, matching
   the driver's chunk-tag form.
4. **The comparison set is exactly the set of files the baseline run produced**; every one must be
   byte-identical (`cmp`).
5. Files produced only by the modified build must exist and be **empty** (assert size 0):
   `ongoing_nuclei_labels.data`, `ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`.
6. `set_env.py` on a param JSON without `NUC_PATH` exports no `NUC_` variable; `cut_chunk_agg.py`
   against a small local HDF5 volume writes no `nuc.raw`.

**V3 — the veto fires.** Add a uint32 `nuc.raw` tagging two supervoxel groups ids 1 and 2, joined by
a **high-affinity** path so the unconstrained run definitely merges them; at least 2000 tagged
voxels each, well above `min_tagged`. Without `nuc.raw`: one segment. With: two segments and
`nuc_cuts.data` non-empty naming the expected pair. The veto fires despite the high affinity.

**V4 — pass-through.** Only one group tagged (allowed); both tagged with the same id (allowed); one
group tagged `0xFFFFFFFF` and the other `1` (vetoed — proves the full uint32 domain works).

**V5 — the algebra, over records the record type actually permits (I5).** The test set is
`{ NONE(0,0,0), NONE(0,0,t>0), PROPER(id1,c>0,t>=c), PROPER(id2,c>0,t>=c), CONFLICT(0,0,t>0) }` —
CONFLICT and NONE always carry `count == 0`, per A1, so plan_v3's "non-zero counts and totals" for
every state is corrected. Over all pairs and triples: `nuc_join` is commutative and associative on
the **full record**, including the `(id1, id2, id2)` case in both groupings; idempotence is
asserted on the **state/id projection only**, with an explicit assertion that
`nuc_join(a,a).count == 2*a.count` so the non-idempotence of the additive fields is pinned.

**V6 — resolution and barrier.**

* dominance: `5000 id1 + 100 id2` → PROPER id1 (0.98 ≥ 0.6), `total>count` counter 1;
  `500 id1 + 400 id2` → CONFLICT (0.56 < 0.6), CONFLICT counter 1; `total` equals 900 in the second
  case, proving the floor never subtracts from `total`.
* `min_tagged`: a supervoxel with 10 tagged voxels → NONE with `total == 10`, below-floor counter 1.
* barrier: the CONFLICT supervoxel **fails** to merge with a PROPER id-3 cluster and with another
  CONFLICT cluster, and **succeeds** with a NONE cluster.
* `ABISS_NUC_DOMINANCE=0.5` and `=1.1` are rejected with a clear message.

**V7 — input contracts.** uint16 accepted and written at uint32 width; a value `> 0xFFFFFFFF`
raises; float32 raises; a signed mask with a negative value raises; a shape mismatch against
`seg.raw` raises. Separately a **truncated `nuc.raw`** makes `acme` exit nonzero with the A5
message, against the Release build.

**V8 — the B4 abort.** Construct records that make `nuc_can_merge` false and force them into merge
propagation via a unit harness, and assert `agg` aborts with both records printed — for
`PROPER id1 + PROPER id2`, `CONFLICT + PROPER`, **and** `CONFLICT + CONFLICT`. The last two are the
cases plan_v3 let through (I3) and must be covered explicitly.

**V9 — hierarchy binaries, executed with fully specified fixtures (I5).** Each test is a script
under `work/test/`, every command carrying a comment naming the driver line it replicates. Fixture
layouts are given by the A2 numpy dtype plus, for the other inputs, the structs the binaries read:
`remap_data_t<seg_t>` = `[('os','<u8'),('ns','<u8')]` (`reduce_chunk.cpp:24`), `size_data_t` =
`[('sid','<u8'),('size','<u8')]`, and boundary files named `boundary_<face>_<tag>.data` for faces
0-5 as written by `BoundaryExtractor::output`. The coder must confirm each layout against the
source before building the fixture and record the confirmation in `code_v0.md`.

* **T1 `reduce_chunk <tag>`** — build `ongoing_nuclei_labels_<tag>.data` plus `remap.data`,
  `ongoing_segments.data`, `done_segments.data`, `residual_rg_<tag>.data`, and the six boundary
  files. Assert `sid`s are remapped and payloads preserved. Then a conflicting many-to-one case:
  two PROPER records with ids 1 and 2 remapped onto one target sid must produce exactly one record
  for that sid with `state == CONFLICT`, and the collision counter must report 1. Per I4 this is no
  longer an abort.
* **T2 `match_chunks <tag>`** — same construction against `o_ongoing_nuclei_labels.data`, including
  the same conflicting case, asserting CONFLICT and the counter — explicitly covering the
  boundary-reconciliation scenario the reviewer identified as legitimate.
* **T3 the real cut chain.** Run `agg` on a nucleus-conflicting fixture so it produces
  `nuc_cuts.data`. Then **attempt to run `scripts/overlap_chunk_me.sh` itself** with a stubbed
  environment (`UPLOAD_CMD`/`DOWNLOAD_CMD` set to `cp -r`, `FILE_PATH`/`IO_SCRATCH_PATH` pointed at
  local directories, `OVERLAP=2`). If it cannot run — `init.sh`, redis, or a cpu-slot dependency —
  record the exact failing command and its output in `code_v0.md`, then run a fallback that
  executes the command sequence extracted from the driver, and **diff the extracted sequence
  against the driver** to prove no drift. Report which of the two paths actually ran. Then the
  `merge_chunks_me.py` step: `python3 scripts/merge_chunks_me.py <json> <META>` where `<json>` is
  the chunk descriptor written by the fixture script (keys `bbox`, `ac_offset`, `boundary_flags`,
  `mip_level`, `children`, as read by `chunk_utils.read_inputs`) and `<META>` is the empty string
  when no extra metadata is configured. Finally `match_chunks <tag>`, asserting the vetoed edge is
  absent from the resulting region graph.
* **T4 two-chunk end-to-end, mandatory.** `work/test/run_hierarchy.sh` builds two adjacent
  `(64,64,64)` chunks tagged with different nucleus ids under `work/test/run/{chunk_a,chunk_b}` and
  runs, per chunk, mirroring `atomic_chunk_me.sh:34-60`: `acme param.txt <tag>`;
  `mv edges_<tag>.data input_rg.data`; the six `cat boundary_*` lines;
  `touch ns.data ongoing_semantic_labels.data ongoing_nuclei_labels.data ongoing_seg_size.data`;
  `agg 0.25 input_rg.data frozen.data ns.data`; `split_remap chunk_offset.txt <tag>`;
  `assort <tag> ""`; `mv residual_rg.data residual_rg_<tag>.data`;
  `mv ongoing_segments.data ongoing_supervoxel_counts_<tag>.data`;
  **`mv ongoing_nuc.data ongoing_nuclei_labels_<tag>.data`** — the move plan_v3 omitted, without
  which the composite step has no nucleus payload at all. Tags are `0_0_0_0` and `0_1_0_0`.

  Then the composite level under `work/test/run/composite`, mirroring `composite_chunk_me.sh:37-70`:
  `python3 scripts/merge_chunks_me.py <json> ""`; `mv ongoing.data localmap.data`;
  `mv residual_rg.data input_rg.data`; `meme <tag> ""`; `cat new_edges.data >> input_rg.data`;
  the six `cat boundary_*` lines; `agg 0.25 input_rg.data frozen.data
  ongoing_supervoxel_counts.data`.

  Assert the two tagged groups remain separate in the final remap and that the constraint survives
  the second hierarchy level. **If T4 cannot be run, the run is `Status: blocked` — not a passing
  run with a caveat.**

**V10 — driver syntax.** `bash -n` every modified shell script.

## Risks and Questions

**R1 — the shipped guarantee is Invariant D, not exact cannot-link.** The I1 counterexample is
permitted by design and must appear verbatim in the README. Anyone reading the feature as absolute
will be wrong. This is a deliberate, user-approved contract choice, not an oversight.

**R2 — `min_tagged = 50` discards evidence, and those discards accumulate.** Under Invariant D that
is a recall limitation rather than a broken guarantee, which is the whole reason the contract was
narrowed. The below-floor counter and its voxel sum measure the cost on real data.

**R3 — `ABISS_NUC_DOMINANCE = 0.6` is unmeasured.** It matches the semantic heuristic's existing
default, which is precedent, not evidence. Too low weakens Bound C; too high turns ordinary bleed
into CONFLICT barriers. The counters make the effect visible on the first real run.

**R4 — the B4 abort is a new hard-failure path.** It is unreachable if B3 is correct, which is why
it aborts rather than papers over; but a latent defect that previously produced a quietly wrong
segmentation will now kill a long job. Stated in the README. Per I4, the two hierarchy sites do
**not** abort, so the data-reachable DoS is gone.

**R5 — V2 proves invariance on a synthetic fixture, not real EM data.** A full Seuron provenance
re-run is out of scope and must be stated in `code_v0.md`.

**R6 — T3's real-driver path may still be unrunnable.** The plan requires attempting it, reporting
the exact blocker, and proving the fallback matches the driver by diff. That is the honest ceiling
without building a redis/cloud harness.

**R7 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per
tagged supervoxel; negligible when nuclei cover a small voxel fraction. Worth a code comment.

**Q1 — is `min_tagged = 50` the right default?** It trades the single-bleed-voxel veto against
discarded weak evidence. Both directions are counted, so the first real run answers it.

**Q2 — should the hierarchy collision counters be promoted to a warning-per-occurrence?** Counts
alone may be too coarse to debug a genuine veto failure. Deferred; an env-gated id list is the
natural follow-up.

## Changes Since Previous Plan Version

Responds to `plan_v3_review.md`. All five findings accepted; the contract question the reviewer
raised (its question 1) was decided by the user rather than by the planner.

* **I1 (Invariant N'' permits joining distinct nuclei) — accepted as a limitation, not fixed.** The
  reviewer is right that a single-dominant-id record cannot deliver exact cannot-link, and right
  that Bound C is compatible with violating it. The user chose on 2026-07-30 to ship the weaker
  guarantee rather than widen the record to identity sets or add `NUC_WS`. Plan_v4 replaces
  Invariant N'' with **Invariant D**, states plainly what it does not promise, carries the
  reviewer's `A: 60/40, B: 60/40` counterexample verbatim into the README, and demotes Bound C from
  a safety argument to a diagnostic bound.
* **I2 (the floor invalidates Bound C and contradicts `total`) — fixed.** The floor now applies to
  the aggregate `total`, never to individual ids, so `total` remains exactly "all nonzero-tagged
  voxels" as A1 defines it and the record is self-consistent. The parameter is renamed
  `ABISS_NUC_MIN_TAGGED` to reflect the changed meaning, and defaults to 50 rather than 0, because
  the reviewer also showed a floor of 0 lets one stray voxel create an absolute veto that
  hard-splits a cell. That a below-floor supervoxel becomes NONE is now a documented recall
  limitation under Invariant D rather than a broken guarantee — which is exactly what narrowing the
  contract bought.
* **I3 (abort covers only one branch of the predicate) — fixed.** B4 now calls
  `nuc_can_merge(a, b)` itself and aborts whenever it is false, so `CONFLICT + PROPER` and
  `CONFLICT + CONFLICT` are covered by construction and the check cannot drift from the predicate.
  V8 tests all three cases.
* **I4 (the `match_chunks` abort is data-reachable) — fixed by removing both hierarchy aborts.** The
  reviewer showed boundary matching legitimately canonicalizes two chunk-local sids of one
  supervoxel onto one sid, and that nothing in the plan distinguishes that from an agg-derived
  remap. Rather than assert an unproven provenance rule, `reduce_chunk` and `match_chunks` now
  combine with `nuc_join` and **count** conflicting collisions; the counter is the observable that
  would expose a real veto failure. The abort survives only at B4, where the pair was tested by
  `nuc_can_merge` a few lines earlier and provenance is unambiguous. T1/T2 assert CONFLICT plus the
  counter instead of a nonzero exit.
* **I5 (V8 not executable) — fixed point by point.** V1 creates and enters the build directory.
  V9's fixtures name their exact binary layouts (`nuc_wire_t`, `remap_data_t`, `size_data_t`,
  `boundary_<face>_<tag>.data`) with an instruction to confirm each against the source. T3 now
  attempts the **real** `overlap_chunk_me.sh` with a stubbed environment, reports the exact blocker
  if it fails, and proves the fallback matches by diff; the `merge_chunks_me.py` invocation, its
  JSON keys, and `<META>` are specified. T4 adds the omitted
  `mv ongoing_nuc.data ongoing_nuclei_labels_<tag>.data` — without which the composite step it
  exists to test carried no nucleus payload — and fixes concrete tags, chunk shapes, directory
  topology, and the threshold. V5's test set is corrected to records the type actually permits,
  since NONE and CONFLICT always carry `count == 0`.
=========================== END artifacts/plan_v4.md ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then any
questions, then the final READY: line.
