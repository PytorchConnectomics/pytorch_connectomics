You are reviewing revision v5 of an implementation plan for a CCC run. You are the CODER: you will
implement this plan.

You reviewed plan_v0 (11 [major]), v1 (G1-G6), v2 (H1-H6), v3 (I1-I5), v4 (J1-J5). Plan_v5 responds
to J1-J5.

FIXED CONTEXT, NOT OPEN FOR RE-LITIGATION:
- The human owner decided to ship the WEAKER guarantee ("Invariant D": no merge joins clusters
  whose RECORDED DOMINANT ids differ). Identity sets and NUC_WS are declined for this run. Judge
  whether plan_v5 states that contract honestly and implements exactly it - not whether it should
  have been stronger.
- You already confirmed I3 and I4 fixed in plan_v4; those designs are carried forward unchanged.

Plan_v5's changes: NONE records now carry total == 0 (the J1 fix), which is claimed to restore a
"Closure" property stated as a theorem with a proof; Invariant D gains a third clause covering
CONFLICT+CONFLICT; two false honesty claims are withdrawn; the B4 abort test is replaced by a
standalone unit binary; and a fixture layout table resolves every binary format from source.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End with a line: READY: yes   or   READY: no

Attack it on:
1. Are J1-J5 really fixed? Name any that are not.
2. ATTACK THE CLOSURE PROOF. Is "state == PROPER => count >= dominance_ratio * total, and
   state != PROPER => count == 0" actually preserved by every reachable operation - extraction,
   nuc_join, load_nuc duplicates, reduce_chunk/match_chunks collisions, and merge propagation?
   Find a counterexample if one exists. Check integer/floating comparison edge cases too.
3. Does B3 now implement EXACTLY Invariant D's three clauses - no more, no less?
4. Is the layout table correct and complete enough that someone unfamiliar with this protocol can
   build the fixtures and run V1-V10?
5. Anything the revision broke that plan_v4 had right.

If the plan is ready to implement, SAY SO PLAINLY. Do not manufacture findings. A finding without a
concrete failing case or a concrete missing specification is not useful.

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

=========================== BEGIN artifacts/plan_v4.md (superseded, for reference) ===========================
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

=========================== BEGIN artifacts/plan_v4_review.md (your prior review) ===========================
# Plan v4 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v4_review.review.raw.md`.

`READY: no`, five findings, all `[major]`. Reviewer verdict: "I3 and I4 are fixed in the
implementation design. I2 remains broken at cluster propagation, and I5 remains incomplete.
Invariant D's principal prohibitions are enforced, but B3 is stricter than the stated contract."

Confirmed fixed: **I3** (the merge-time check now reuses the whole predicate) and **I4** (both
hierarchy aborts removed; the reviewer explicitly confirms the denial-of-service is gone and that
removing them does not make Invariant D unobservable, since forbidden pairs are caught immediately
in B3/B4).

Three findings land on plan_v4's own new material. J1 is the substantive one: moving the floor to
aggregate `total` fixed extraction but broke **merge closure** — a `NONE` record carrying nonzero
`total` poisons the dominance property of any `PROPER` record it joins. J2 and J3 are both accuracy
failures in how the contract is stated, which matters more than usual here because the contract
being honestly stated is the entire point of the user's decision.

## Findings

**J1 [major] — the aggregate floor keeps `total` exact only at extraction; `nuc_join` preserves
neither the record semantics nor Bound C.** With defaults:

```text
A: 49 voxels id2  -> NONE,      count=0,  total=49
B: 50 voxels id1  -> PROPER id1, count=50, total=50
```

B3 permits `A + B`, and `nuc_join` produces `PROPER id1, count=50, total=99`. Id1 is 50.5% of the
tagged voxels, below the required 60%, and the 49 minority voxels exceed Bound C's `0.4 * 99`. If A
had instead contained id1, `count=50` would also stop meaning "all voxels backing id1". Two
49-voxel NONE records likewise stay NONE with `total=98` even though their aggregate clears the
floor.

The reviewer's conclusion: I2 is not fixed, and **v4 broke the merge-closed state semantics that v3
had when its floor was disabled**.

**J2 [major] — B3/B4 do not implement exactly Invariant D.** `CONFLICT + CONFLICT` involves no pair
of differing recorded dominant ids and neither side carries a recorded dominant id, so Invariant D
as written *permits* it — but B3 refuses it and B4 mirrors that refusal. No Invariant-D-forbidden
pair is permitted, so the code is strictly stronger than the contract, but the extra refusal is an
undocumented containment policy with an over-segmentation cost. Either add the clause to Invariant D
or allow the merge.

**J3 [major] — the statement of Invariant D's limits is itself not honest.** Two specific errors:

* The claim that contamination within a supervoxel yields CONFLICT is contradicted by A4 and by the
  plan's own example: `60 id1 + 40 id2` is PROPER, and a mixed supervoxel below 50 tagged voxels is
  NONE.
* The claim that the limitation is pathological only near 40% contamination is false.
  `99 id1 + 1 id2` and `99 id1 + 1 id3` are both PROPER id1 and may merge, joining mask identities
  2 and 3. The limitation applies at *any* contamination level.

The "~40%" qualification must go.

**J4 [major] — V8 cannot reach the condition it claims to test.** Through normal `agg` execution
B3 rejects every pair for which `nuc_can_merge` is false, so such a pair cannot reach B4. "Force
them into merge propagation via a unit harness" specifies neither an injectable hook nor an
extracted propagation function, while simultaneously requiring that `agg` itself abort. A concrete
test seam, or a harness build and invocation, is required.

**J5 [major] — the hierarchy verification remains under-specified.** Concretely:

* V1 says all commands start inside `work/abiss` and then runs `cd work/abiss`, targeting a
  nonexistent nested directory.
* T1 does not define the binary layout or required contents of `residual_rg_<tag>.data` or the six
  boundary files.
* T2 says "same construction" without enumerating every file `match_chunks` consumes.
* T3 gives no positional invocation for `overlap_chunk_me.sh`, no exact fallback extraction
  boundaries, and leaves its tag, JSON path and values, and staging topology unspecified.
* T4 leaves the composite `<json>` and `<tag>` unresolved and never names the final remap file or
  its format for the concluding assertion.

The instruction to derive formats from source is useful implementation work but does not make these
tests runnable as specified by someone unfamiliar with the protocol.

## Questions

1. Is `CONFLICT + CONFLICT` intentionally forbidden? If so, should that clause be added to
   Invariant D?
2. Should Bound C be dropped, or should joins involving `NONE(total > 0)` transition conservatively
   so PROPER records continue satisfying their stated dominance semantics?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v4_review.md ===========================

=========================== BEGIN artifacts/plan_v5.md (UNDER REVIEW) ===========================
# Plan v5

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, and use it to
guide agglomeration under the contract the user fixed on 2026-07-30: **dominant-id cannot-link**,
honestly stated.

Plan_v4 moved the evidence floor from per-id to aggregate `total` to keep the record
self-consistent at extraction. The reviewer showed (J1) that this fixed extraction and broke
**merge closure**: a `NONE` record carrying nonzero `total` poisons the dominance property of any
`PROPER` record it joins — `NONE(total=49) + PROPER(id1, 50, 50)` yields `PROPER(id1, 50, 99)`,
where id1 is 50.5% of tagged voxels against a required 60%.

Plan_v5's fix is one line of semantics with a provable consequence:

> **A `NONE` record carries `count == 0` and `total == 0`.** A supervoxel below the floor
> contributes no *usable* evidence, so it contributes nothing to the record; its mass is reported in
> the diagnostic counters instead of being smuggled into `total`.

That restores a closure property strong enough to state as a theorem:

> **Closure.** Every reachable record satisfies
> `state == PROPER  =>  count >= dominance_ratio * total`, and `state != PROPER => count == 0`.
>
> *Proof.* Extraction establishes it: NONE gets `(0,0)`, PROPER by construction, CONFLICT has
> `count == 0`. `nuc_join` preserves it: a NONE operand contributes `(0,0)` so the other side's
> inequality carries unchanged; two same-id PROPER operands add both sides of the inequality; every
> other combination yields CONFLICT with `count == 0`. ∎
>
> **Bound C** follows immediately: in any PROPER cluster, voxels tagged with a different nucleus
> number `total - count <= (1 - dominance_ratio) * total`.

Plan_v5 also corrects two accuracy failures the reviewer found in how the contract was *stated*
(J2, J3), which matters more than usual because honest statement is the whole point of the user's
decision; makes the invariant-tripwire testable via a real seam rather than an unreachable path
(J4); and pins every fixture layout the hierarchy tests need (J5).

## Scope

**In scope** (`lib/abiss`, in the isolated clone `work/abiss`): record and wire types; `nuc_join`;
`NucExtractor`; propagation through the distributed hierarchy including OVERLAP=2 veto feedback;
the veto in `mean_aggl.cpp`; `NUC_PATH` plumbing; documentation carrying the contract verbatim; and
executable tests.

**Out of scope, decided by the user:** identity sets and `NUC_WS`. Both remain the natural
follow-up if the extraction counters show mixed or sub-floor supervoxels are common on real data;
the counters exist to answer exactly that. Also out: `NUC_MIP` (replaced by the C5 shape
assertion), the pytorch_connectomics wrapper, mask generation, perinuclear-shell tagging, and
`scripts/reduce_chunk.py` / `scripts/match_chunks.py`, which are **dead legacy** —
`overlap_chunk_me.sh:44` invokes the C++ `$BIN_PATH/reduce_chunk` and `composite_chunk_me.sh:42`
the C++ `$BIN_PATH/match_chunks`.

## Proposed Changes

### The contract, stated accurately (J2, J3)

This text goes verbatim into `README.md` and into a comment above `nuc_can_merge`.

> **Invariant D.** No merge performed by agglomeration:
> 1. joins two clusters whose recorded dominant nucleus ids differ; or
> 2. joins a CONFLICT cluster to a cluster carrying a recorded dominant id; or
> 3. **joins two CONFLICT clusters.**
>
> **What this does not promise.** The record tracks one dominant identity per cluster. **Minority
> identities are not tracked at all, at any contamination level.** Two clusters both recorded
> PROPER id1 may merge no matter which other nuclei their voxels contain:
>
> ```text
> A: 99 voxels id1, 1 voxel id2   -> PROPER id1
> B: 99 voxels id1, 1 voxel id3   -> PROPER id1     # A+B allowed; joins identities 2 and 3
> ```
>
> What *is* bounded is minority **mass**, not the minority identity set: by Bound C, minority voxels
> in a PROPER cluster number at most `(1 - dominance_ratio) * total`.
>
> Nor does a mixed supervoxel necessarily become CONFLICT: `60 id1 + 40 id2` is PROPER id1 at the
> default ratio, and any supervoxel with fewer than `min_tagged` tagged voxels is NONE regardless of
> how mixed it is.

Clause 3 is new (J2): plan_v4's code refused CONFLICT+CONFLICT while its stated contract permitted
it. The refusal is intentional — merging two contaminated clusters compounds contamination with no
benefit, and both are already isolated from proper ids so it costs no additional
over-segmentation — so the clause is added rather than the refusal removed. This answers the
reviewer's question 1.

Plan_v4's "pathological only near 40% contamination" claim and its "contamination within a
supervoxel results in CONFLICT" claim are both **withdrawn** as false (J3).

### Phase A — types, algebra, extraction

**A1. `src/seg/Types.h`** — beside `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                       // voxel dtype of nuc.raw; 0 == background

enum : uint8_t {
    NUC_STATE_NONE     = 0,   // no usable evidence; count == 0 AND total == 0
    NUC_STATE_PROPER   = 1,   // count >= dominance_ratio * total
    NUC_STATE_CONFLICT = 2,   // tagged, no dominant id; count == 0
};

struct __attribute__((packed)) nuc_record_t {
    uint8_t  state = NUC_STATE_NONE;
    uint32_t id    = 0;   // meaningful only when state == PROPER
    uint64_t count = 0;   // voxels backing `id`; 0 unless PROPER
    uint64_t total = 0;   // usable tagged voxels; 0 when state == NONE   <-- J1
};
```

The full uint32 id domain `[1, 0xFFFFFFFF]` stays usable; only `0` is reserved as background.

**`total` counts usable tagged voxels, and is 0 for NONE.** This is the J1 fix and the only
semantic change from plan_v4. Sub-floor mass is *not* recorded here — it is reported by the
below-floor counters (A4). Document the Closure property above the struct, because every other part
of the design leans on it.

**A2. Wire format.** `std::pair` is not a layout contract; define it explicitly and use it at every
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
static_assert(offsetof(nuc_wire_t, sid) == 0 && offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id) == 9 && offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);
```

**A3. `nuc_join`.** Unchanged from plan_v4:

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

With A1's `NONE => total == 0`, the NONE branches now add 0 to `total`, which is exactly what makes
Closure hold. Under plan_v4's semantics they added the hidden sub-floor mass, which is what broke
it.

Algebra: the **state/id projection** is a flat-lattice join — associative, commutative, idempotent
on that projection. `count` and `total` are an additive commutative monoid — associative and
commutative, **not** idempotent. So `nuc_join` on the full record is associative and commutative
only, which is sufficient because every combination site joins records over **disjoint voxel sets**.

**A4. `src/seg/NucExtractor.hpp` (new).** Modeled on `SemExtractor.hpp`; no class LUT; nullable
source so A5 does not branch.

* Constructor takes `const Chunk *` (may be `nullptr`), `dominance_ratio`, `min_tagged`.
* `collectVoxel(Coord c, Tseg segid)`: return if null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]` and, if `id != 0`, increment `m_counts[segid][id]`
  (`MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`). Untagged supervoxels never enter it.
* `collectBoundary`, `collectContactingSurface`: empty.

`output(chunkMap, filename)` remaps supervoxel ids through `chunkMap` and merges the id→count maps
of supervoxels sharing a target, as `SemExtractor::output` does (`SemExtractor.hpp:31-51`). Then per
target supervoxel:

```text
tagged := sum of all counts
if tagged == 0 or tagged < min_tagged:        (NONE,     -,      0,         0)     # J1: total 0
(max_id, max_count) := largest count, ties -> smaller id
if max_count >= dominance_ratio * tagged:     (PROPER,   max_id, max_count, tagged)
else:                                         (CONFLICT, -,      0,         tagged)
```

`min_tagged` (`ABISS_NUC_MIN_TAGGED`, default **50**) stops a single bleed voxel in an otherwise
untagged supervoxel from becoming PROPER and casting a veto that hard-splits a real cell.
`dominance_ratio` (`ABISS_NUC_DOMINANCE`, default **0.6**, matching
`agglomeration_semantic_heuristic_t::dominant_signal_ratio` at `mean_aggl.cpp:132`) handles bleed
inside an already-tagged supervoxel. Reject a ratio outside `(0.5, 1.0]` with a clear message.

Tie-break by smaller id: `MapContainer` may be `absl::flat_hash_map` (`CMakeLists.txt:42-49`), whose
iteration order is unstable, and a nondeterministic winner would break reproducibility. This argmax
runs **once**, over complete evidence; no downstream stage selects a winner.

Counters, printed with a stable `nuc:` prefix: `nuc: conflict_sv=<n>`;
`nuc: minority_sv=<n>` (PROPER with `total > count`); `nuc: subfloor_sv=<n>`;
**`nuc: subfloor_voxels=<n>`** — the last is now the *only* record of sub-floor mass, since A1 keeps
it out of `total`, and together with `subfloor_sv` it is the measurement that decides whether
identity sets or `NUC_WS` are worth a follow-up run.

`output()` **always creates the file**, writing zero records when the source was null (B5).

**A5. `src/seg/atomic_chunk_ME.cpp`.** The file duplicates the `traverseSegments<1>(...)` call for
the sem-present and sem-absent cases (lines 78-104); a nullable extractor keeps that at two
branches.

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

**B1. No new agglomeration parameters.** `agglomeration_param_t` and `heuristics_aff_threshold`
(line 148) are untouched; all tuning lives at extraction.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156) and
`load_nuc(...)` beside `load_sem` (line 245), called next to line 545 on
`ongoing_nuclei_labels.data`, reading `nuc_wire_t`. Mirror `load_sem`'s index mapping; an empty or
missing file yields an empty vector (lines 248-251) so every nucleus path is skipped when unused.
Duplicate sids combine with `nuc_join`; CONFLICT is legitimate here, because
`ongoing_nuclei_labels.data` is concatenated across chunks and one boundary-spanning supervoxel
contributes one record per chunk over disjoint voxels.

| Site | Operation | Conflict policy |
|---|---|---|
| `NucExtractor::output` | aggregate raw counts, resolve (A4) | CONFLICT valid |
| `load_nuc` duplicates | `nuc_join` | CONFLICT valid |
| `reduce_chunk` collision | `nuc_join` | CONFLICT valid; **counted** |
| `match_chunks` collision | `nuc_join` | CONFLICT valid; **counted** |
| merge propagation (B4) | `nuc_join` | **abort** if `!nuc_can_merge(a,b)` |

**B3. The veto.**

```cpp
inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    const bool ca = (a.state == NUC_STATE_CONFLICT), cb = (b.state == NUC_STATE_CONFLICT);
    if (ca && cb) return false;                          // Invariant D clause 3
    if (ca)       return b.state == NUC_STATE_NONE;      // clause 2
    if (cb)       return a.state == NUC_STATE_NONE;      // clause 2
    if (a.state == NUC_STATE_NONE || b.state == NUC_STATE_NONE) return true;
    return a.id == b.id;                                 // clause 1
}
```

Each branch is annotated with the Invariant D clause it implements, so the code and the contract
cannot drift (J2). Call it in `agglomerate_cc`'s main loop immediately after the frozen-edge block
ending at line 697 and before the semantic check at line 699, guarded by `if (!nuc_ids.empty())`,
with **no affinity gate**. On refusal push the edge to a new `nuc_rg_vector` in
`agglomeration_output_t` (line 167), set `e.edge->w = Limits::min()`, `continue` — the shape of the
semantic refusal at lines 700-706. A nucleus refusal is logged in preference to a semantic one.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687**, which already
reads `!sem_counts.empty()`; replicating it would make enabling nuclei silently alter frozen-edge
handling.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), same `v0`/`v1`/`s`
swap discipline, guarded by `!nuc_ids.empty()`. Evaluate `nuc_can_merge(a, b)`; if **false**, print
both records and both `seg_indices` values to `stderr` and `std::abort()`. Then combine with
`nuc_join`.

This abort is **unreachable through `agg` by construction** — B3 rejects the same pairs a few lines
earlier. It is an invariant tripwire against a future edit that breaks that correspondence, not a
runtime data path. V8 tests the predicate directly and verifies the call site by inspection; it does
**not** claim `agg` can be made to abort on data (J4).

**B5. File lifecycle.** Creation is unconditional — `agg` opens and closes `ongoing_nuc.data`,
`done_nuc.data`, `nuc_cuts.data` every run regardless of `nuc_ids.empty()`, and
`NucExtractor::output` always creates `ongoing_nuclei_labels.data`. Required, because the drivers
`mv` these under `set -euo pipefail` and `reduce_chunk.cpp`'s `read_array` aborts on a file it
cannot open. Content is conditional on `!nuc_ids.empty()`. Mirror the ongoing/done split at lines
848-857 and 880-891 and `of_sem_cuts` at 960-961 and 1050-1052.

**B6. Diagnostics.** All `nuc:`-prefixed: the four A4 extraction counters, `load_nuc` duplicate
combinations producing CONFLICT, and `reduce_chunk` / `match_chunks` collisions producing CONFLICT.
Only B4's violation aborts.

### Phase C — hierarchy and drivers

**C1. `src/seg/reduce_chunk.cpp`.** Add a nucleus reducer reading `nuc_wire_t`, applying the `sid`
remap from `remap.data`, combining collisions with `nuc_join` and counting those yielding CONFLICT.
Emit one record per sid. Call from line 228 with `ongoing_nuclei_labels_%1%.data` /
`reduced_ongoing_nuclei_labels_%1%.data`. An absent or empty input must still produce the output.

**C2. `src/seg/match_chunks.cpp`.** Mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data`, writing `ongoing_nuclei_labels.data`, with
C1's collision handling. Per the reviewer's I4, this site must **not** abort: boundary matching
legitimately canonicalizes two chunk-local sids of one supervoxel onto one sid.

**C3. OVERLAP=2 veto feedback.** `overlap_chunk_me.sh:50` copies `sem_cuts.data` to
`vetoed_edges_<chunk>.data`, `merge_chunks_me.py:62` merges them, `match_chunks.cpp:190-238` removes
those edges from the region graph. Without the equivalent, a nucleus veto made in one round is
silently forgotten in the next. After the existing `cp` at line 50, add
`cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. The consumer already sorts and dedups
(`match_chunks.cpp:212-214`) and the record is the same `(seg_t, seg_t)` pair.

**C4. Shell drivers.**

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40 plus the same four moves at the
  line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), the
  `done_nuc.data` move (line 64), the `nuc_cuts.data` move (line 66), and the C3 `cat`.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"`; not `NUC_MIP`. The loop is `for e in env: if e in
  data:`, so a JSON without `NUC_PATH` exports nothing new; comment it.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49) add the `NUC_PATH` block:
  `load_data(global_param['NUC_PATH'], mip=global_param['AFF_RESOLUTION'],
  fill_missing=global_param.get('NUC_FILL_MISSING', False))`, `cut_data` with the same
  `start_coord`/`end_coord` as `seg.raw`, validate, `save_raw_data("nuc.raw", ...)`.

  Alignment: read at `AFF_RESOLUTION` and assert the nucleus cutout's spatial shape equals the
  `seg.raw` cutout's shape; raise otherwise. Value/dtype: raise on non-integer dtype, on a negative
  value in a signed dtype, or on any value `> 0xFFFFFFFF`; otherwise `astype(numpy.uint32)`
  losslessly. The cast is mandatory because `save_raw_data` writes `data.dtype` verbatim
  (`cut_chunk_common.py:38-50`) while the binary mmaps `nuc.raw` as `nuc_t`.
* `scripts/merge_chunks_me.py:59`, `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"` beside `"ongoing_semantic_labels"`.

### Phase D — documentation

`README.md`: the `NUC_PATH` key with dtype, range, and alignment contracts; `ABISS_NUC_DOMINANCE`
and `ABISS_NUC_MIN_TAGGED`; the three-state record and the Closure property; the `nuc:` counters and
the follow-up decision they inform; the `nuc_cuts` / `nuc_rejected_edges` outputs; the B4 abort as a
tripwire; **Invariant D reproduced verbatim including all three clauses, the `99/1` counterexample,
and the two withdrawn claims corrected**; and the usage caveat that the mask should tag
**perinuclear cytoplasm**, not raw nucleus interiors, because the nuclear envelope is a membrane the
affinity network boundaries.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, state enum, `nuc_record_t` (NONE ⇒ total 0), `nuc_wire_t` + static_asserts, `nuc_join`, Closure comment |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, dominance + `min_tagged`, four counters |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw` + unconditional size check, extractor in both branches, env parsing |
| `src/agg/mean_aggl.cpp` | `load_nuc`, `nuc_can_merge` (clause-annotated), veto, `nuc_join` propagation + predicate abort, counters, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | nucleus reducer; `nuc_join` + counter |
| `src/seg/match_chunks.cpp` | `process_nucs`; `nuc_join` + counter; never aborts |
| `scripts/set_env.py` | export `NUC_PATH` |
| `scripts/cut_chunk_agg.py` | `nuc.raw` with shape, dtype, range validation |
| `scripts/atomic_chunk_me.sh`, `composite_chunk_me.sh`, `overlap_chunk_me.sh` | nucleus artifacts; `cat nuc_cuts.data >> vetoed_edges_*` |
| `scripts/merge_chunks_me.py`, `merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | Invariant D verbatim, Closure, counters, caveats |
| `work/test/` | fixtures, `test_nuc_algebra.cpp`, hierarchy scripts |

Untouched on purpose: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy);
`src/ws/*`; `CMakeLists.txt` (header-only addition needs no target change — confirm, edit only if
the build proves otherwise).

## Verification Plan

**All paths below are relative to `work/abiss`, which is the working directory for every command
unless stated otherwise** (J5: plan_v4 said this and then wrote `cd work/abiss`, a nested path that
does not exist).

**Fixture binary layouts — every format the tests need, resolved from source.** Little-endian,
`align=False`, all `__attribute__((packed))`:

| File | numpy dtype | itemsize | Source |
|---|---|---|---|
| `ongoing_nuclei_labels*.data`, `ongoing_nuc.data`, `done_nuc.data` | `[('sid','<u8'),('state','u1'),('id','<u4'),('count','<u8'),('total','<u8')]` | 29 | A2 |
| `input_rg.data`, `residual_rg_<tag>.data`, `edges_<tag>.data`, `new_edges.data` | `[('s1','<u8'),('s2','<u8'),('aff','<f4'),('area','<u8')]` | 28 | `rg_entry<seg_t,aff_t>`, `Types.h:37-51` |
| `remap.data`, `extra_remaps.data` | `[('os','<u8'),('ns','<u8')]` | 16 | `remap_data_t<seg_t>`, `reduce_chunk.cpp:24` |
| `ongoing_segments.data`, `done_segments.data`, `ns.data`, `ongoing_seg_size.data` | `[('sid','<u8'),('size','<u8')]` | 16 | `size_data_t` |
| `boundary_<face>_<tag>.data`, `frozen.data` | `'<u8'` (flat array) | 8 | `BoundaryExtractor::output`, `BoundaryExtractor.hpp:34-44` |
| `matching_faces.data`, `o_boundary_<i>_<tag>.tmp` | `[('oid','<u8'),('boundary_size','<u8'),('nid','<u8'),('agg_size','<u8')]` | 32 | `matching_entry_t<seg_t>`, `Types.h:28-34` |
| `vetoed_edges.data`, `nuc_cuts.data`, `sem_cuts.data` | `[('v0','<u8'),('v1','<u8')]` | 16 | `mean_aggl.cpp:1050-1052` |
| `chunkmap.data` | `[('k','<u8'),('v','<u8')]` | 16 | `loadChunkMap`, `Utils.hpp:53-66` |

`param.txt` is three lines: `offset[0] offset[1] offset[2]`, `dim[0] dim[1] dim[2]`, `ac_offset`
(`atomic_chunk_ME.cpp:31-34`). The coder must confirm each layout against the source before building
a fixture and record the confirmation in `code_v0.md`.

**V1 — build.**

```bash
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8 && cd ..
```

Compiles clean including A2's `static_assert`s. Report any new warning.

**V2 — default-path invariance.** This repo reproduces a Seuron provenance record, so a silent
change to no-nucleus output is a hard failure.

1. Build **baseline** binaries from `run_start_ref` in a scratch worktree of the clone into a
   separate build directory; build **modified** binaries.
2. `work/test/make_fixture.py` (fixed seed), chunk shape `(64,64,64)`, tag `0_0_0_0`: `aff.raw`
   (float32 `(64,64,64,3)` Fortran), `seg.raw` (uint64 Fortran), empty `chunkmap.data`, `param.txt`.
   No `nuc.raw`.
3. Under both builds in separate directories, staging exactly as `atomic_chunk_me.sh:34-41`:
   `../build/acme param.txt 0_0_0_0`; `mv edges_0_0_0_0.data input_rg.data`;
   `for i in 0 1 2 3 4 5; do cat boundary_${i}_0_0_0_0.data >> frozen.data; done`;
   `touch ns.data ongoing_semantic_labels.data ongoing_seg_size.data`;
   `../build/agg 0.25 input_rg.data frozen.data ns.data`.
4. **The comparison set is exactly the files the baseline run produced**; each must be
   byte-identical (`cmp`).
5. Files produced only by the modified build must exist and be **empty**:
   `ongoing_nuclei_labels.data`, `ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`.
6. `set_env.py` on a param JSON without `NUC_PATH` exports no `NUC_` variable; `cut_chunk_agg.py`
   against a small local HDF5 volume writes no `nuc.raw`.

**V3 — the veto fires.** A uint32 `nuc.raw` tagging two supervoxel groups ids 1 and 2, joined by a
high-affinity path so the unconstrained run merges them; ≥2000 tagged voxels each. Without
`nuc.raw`: one segment. With: two segments, `nuc_cuts.data` non-empty naming the pair. Fires despite
high affinity.

**V4 — pass-through.** One group tagged (allowed); both same id (allowed); one tagged `0xFFFFFFFF`
and one `1` (vetoed — proves the full uint32 domain).

**V5 — algebra and Closure (J1's regression test).** A standalone unit binary,
`work/test/test_nuc_algebra.cpp`, `#include "../../src/seg/Types.h"`, built directly:

```bash
g++ -std=c++20 -O1 -I src work/test/test_nuc_algebra.cpp -o work/test/test_nuc_algebra && \
  work/test/test_nuc_algebra
```

Over the record set `{ NONE(0,0,0), PROPER(id1,c,t), PROPER(id2,c,t), CONFLICT(0,0,t>0) }` with
several `(c,t)` satisfying `c >= 0.6*t`, for all pairs and triples:

* `nuc_join` is commutative and associative on the full record, including `(id1, id2, id2)` in both
  groupings;
* idempotence on the state/id projection only, with an explicit assertion
  `nuc_join(a,a).count == 2*a.count`;
* **Closure**: every join result satisfies `state == PROPER => count >= 0.6*total` and
  `state != PROPER => count == 0`;
* the J1 case specifically: `NONE(0,0,0)` joined with `PROPER(id1,50,50)` gives
  `PROPER(id1,50,50)` — `total` stays 50, not 99.

**V6 — resolution.** `5000 id1 + 100 id2` → PROPER id1, `total=5100`, `minority_sv` 1;
`500 id1 + 400 id2` → CONFLICT, `total=900`; a supervoxel with 10 tagged voxels → NONE with
**`total == 0`**, `subfloor_sv` 1 and `subfloor_voxels` 10. Barrier: the CONFLICT supervoxel fails
to merge with PROPER id3 and with another CONFLICT, and succeeds with NONE.
`ABISS_NUC_DOMINANCE=0.5` and `=1.1` are rejected.

**V7 — input contracts.** uint16 accepted, written at uint32 width; `> 0xFFFFFFFF` raises; float32
raises; negative in a signed dtype raises; shape mismatch against `seg.raw` raises. A **truncated
`nuc.raw`** makes `acme` exit nonzero with the A5 message, against the Release build.

**V8 — the B4 tripwire (J4).** `test_nuc_algebra` asserts `nuc_can_merge` is false for
`PROPER id1 + PROPER id2`, `CONFLICT + PROPER`, and `CONFLICT + CONFLICT`, and true for the
permitted pairs — covering exactly Invariant D's three clauses. `code_v0.md` must then state
plainly: **the B4 abort is unreachable through `agg` because B3 rejects the same pairs first; its
call site is verified by inspection, not execution.** Do not claim a live abort was observed.

**V9 — hierarchy binaries.** Scripts under `work/test/`, every command commented with the driver
line it replicates.

* **T1 `../build/reduce_chunk 0_0_0_0`** — build `ongoing_nuclei_labels_0_0_0_0.data`,
  `remap.data`, `ongoing_segments.data`, `done_segments.data`, `residual_rg_0_0_0_0.data`, and
  `boundary_{0..5}_0_0_0_0.data` using the layout table above. Assert `sid`s remapped and payloads
  preserved in `reduced_ongoing_nuclei_labels_0_0_0_0.data`. Then a conflicting many-to-one case:
  two PROPER records ids 1 and 2 remapped onto one sid must produce exactly one record with
  `state == CONFLICT` and the collision counter reporting 1 — **not** a nonzero exit (I4).
* **T2 `../build/match_chunks 0_0_0_0`** — consumes `matching_faces.data`, `o_residual_rg.data`,
  `o_incomplete_edges_0_0_0_0.tmp`, `vetoed_edges.data`, `o_boundary_{0..5}_0_0_0_0.tmp`,
  `o_ongoing_supervoxel_counts.data`, `o_ongoing_seg_size.data`, `o_ongoing_semantic_labels.data`,
  and the new `o_ongoing_nuclei_labels.data` (enumerated from `match_chunks.cpp:33,144,145,190,250,
  342,363,384`). Same assertions, including the conflicting case, which is exactly the legitimate
  boundary-reconciliation scenario.
* **T3 the real cut chain.** Run `agg` on a nucleus-conflicting fixture so it produces
  `nuc_cuts.data`. Then attempt the real driver:
  `OVERLAP=2 UPLOAD_CMD="cp -r" DOWNLOAD_CMD="cp -r" FILE_PATH=<local> IO_SCRATCH_PATH=<local>
  BIN_PATH=<abs build> SCRIPT_PATH=<abs scripts> bash scripts/overlap_chunk_me.sh
  work/test/run/chunk_a.json`. The JSON is what `chunk_utils.read_inputs` reads: keys `bbox`
  (6 ints), `ac_offset`, `boundary_flags` (6), `mip_level`, `indices`, `neighbours`. If it cannot
  run — `init.sh`, redis, or cpu-slot dependencies — record the exact failing command and its
  stderr in `code_v0.md`, then run the fallback: extract the command sequence between
  `try $BIN_PATH/reduce_chunk` (line 44) and the final `mv` (line 66), execute it, and `diff` the
  extracted block against those driver lines to prove no drift. Report which path ran. Then
  `python3 scripts/merge_chunks_me.py work/test/run/composite.json ""` and
  `../build/match_chunks 0_0_0_0`, asserting the vetoed edge is absent from the region graph.
* **T4 two-chunk end-to-end, mandatory.** `work/test/run_hierarchy.sh`, two `(64,64,64)` chunks
  tagged with different nucleus ids under `work/test/run/{chunk_a,chunk_b}`, tags `0_0_0_0` and
  `0_1_0_0`. Per chunk, mirroring `atomic_chunk_me.sh:34-60`: `acme param.txt <tag>`;
  `mv edges_<tag>.data input_rg.data`; the six `cat boundary_*` lines;
  `touch ns.data ongoing_semantic_labels.data ongoing_nuclei_labels.data ongoing_seg_size.data`;
  `agg 0.25 input_rg.data frozen.data ns.data`; `split_remap chunk_offset.txt <tag>`;
  `assort <tag> ""`; `mv residual_rg.data residual_rg_<tag>.data`;
  `mv ongoing_segments.data ongoing_supervoxel_counts_<tag>.data`;
  `mv ongoing_nuc.data ongoing_nuclei_labels_<tag>.data`.

  Composite level under `work/test/run/composite`, mirroring `composite_chunk_me.sh:37-70`:
  `python3 scripts/merge_chunks_me.py work/test/run/composite.json ""`;
  `mv ongoing.data localmap.data`; `mv residual_rg.data input_rg.data`; `meme 0_0_0_1 ""`;
  `cat new_edges.data >> input_rg.data`; the six `cat boundary_*` lines;
  `agg 0.25 input_rg.data frozen.data ongoing_supervoxel_counts.data`. `composite.json` carries
  `mip_level: 1`, `children` naming the two atomic tags, and the union `bbox`.

  **Final assertion:** read `remap.data` (dtype `[('os','<u8'),('ns','<u8')]`) plus
  `done_segments.data` / `ongoing_segments.data` (`[('sid','<u8'),('size','<u8')]`), resolve each
  tagged group's supervoxels to their final representative, and assert the two representatives
  **differ**. **If T4 cannot be run, the run is `Status: blocked` — not a passing run with a
  caveat.**

**V10 — driver syntax.** `bash -n` every modified shell script.

Every claim in `code_v0.md` quotes real command output; a step not run is reported as not run.

## Risks and Questions

**R1 — the shipped guarantee is Invariant D.** The `99/1` counterexample is permitted by design and
appears verbatim in the README. User-approved contract choice, not an oversight.

**R2 — `min_tagged = 50` discards evidence and those discards accumulate.** Under Invariant D that
is a recall limitation, not a broken guarantee — which is what narrowing the contract bought. With
A1, discarded mass is now visible in `subfloor_voxels` rather than hidden inside `total`.

**R3 — `ABISS_NUC_DOMINANCE = 0.6` is unmeasured**, matching the semantic heuristic's default as
precedent, not evidence. The counters make the effect visible on the first real run.

**R4 — the B4 abort is a tripwire, not a data path.** Unreachable if B3 is correct; a future edit
that decouples them turns a silently wrong segmentation into a loud failure. Per I4 the hierarchy
sites do not abort, so there is no data-reachable DoS.

**R5 — V2 proves invariance on a synthetic fixture, not real EM data.** A Seuron provenance re-run
is out of scope and must be stated in `code_v0.md`.

**R6 — T3's real-driver path may still be unrunnable**; the plan requires attempting it, reporting
the exact blocker, and proving the fallback matches by `diff`.

**R7 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per tagged
supervoxel; negligible when nuclei cover a small voxel fraction. Worth a code comment.

**Q1 — is `min_tagged = 50` right?** It trades the single-bleed-voxel veto against discarded weak
evidence. Both directions are counted, so the first real run answers it.

**Q2 — should the hierarchy collision counters carry per-occurrence detail?** Counts alone cannot
distinguish legitimate boundary reconciliation from a hierarchy defect, as the reviewer noted.
Deferred; an env-gated id list is the natural follow-up.

## Changes Since Previous Plan Version

Responds to `plan_v4_review.md`. All five findings accepted. The reviewer confirmed I3 and I4 fixed
in plan_v4 and those designs are carried forward unchanged.

* **J1 (the floor breaks merge closure) — fixed by `NONE => total == 0`.** A NONE record no longer
  carries sub-floor mass, so `nuc_join`'s NONE branches add 0 and the dominance property survives
  every join. The reviewer's case now gives `NONE(0,0,0) + PROPER(id1,50,50) = PROPER(id1,50,50)`
  with `total` 50, not 99. This is strong enough to state as the **Closure** theorem with a proof,
  from which **Bound C** follows as a corollary rather than as a separate claim. Discarded sub-floor
  mass is reported by the new `nuc: subfloor_voxels` counter instead of being smuggled into `total`.
  V5 tests Closure over all pairs and triples and pins the exact J1 case. Answers reviewer question
  2: `NONE(total>0)` is eliminated rather than given a conservative transition, and Bound C is kept.
* **J2 (B3 stricter than the stated contract) — fixed by amending the contract.** Invariant D gains
  clause 3, `joins two CONFLICT clusters`. The refusal is intentional — compounding contamination
  with no benefit, and no extra over-segmentation cost since both sides are already isolated from
  proper ids — so the statement is corrected to match the code rather than the reverse. Each branch
  of `nuc_can_merge` is annotated with the clause it implements so they cannot drift again.
* **J3 (the honesty statement was not honest) — both false claims withdrawn.** The "~40%
  contamination" qualification is removed: the limitation applies at any level, and the README now
  carries the reviewer's `99 id1 + 1 id2` / `99 id1 + 1 id3` example. The claim that within-supervoxel
  contamination yields CONFLICT is corrected: `60 id1 + 40 id2` is PROPER, and any supervoxel below
  `min_tagged` is NONE however mixed. The contract now says plainly that **minority identities are
  not tracked at all** and that only minority *mass* is bounded.
* **J4 (V8 tests an unreachable condition) — fixed with a real seam.** `nuc_can_merge` and
  `nuc_join` live in `Types.h`, so `work/test/test_nuc_algebra.cpp` includes it and tests the
  predicate directly with a concrete `g++` build and invocation. `code_v0.md` must state that the
  B4 abort is unreachable through `agg` by construction and that its call site is verified by
  inspection. Plan_v4's contradictory demand — force it via an unspecified harness *and* have `agg`
  abort — is dropped.
* **J5 (hierarchy verification under-specified) — fixed point by point.** V1's `cd work/abiss` bug
  is gone; all paths are stated relative to `work/abiss` once. A **layout table** now resolves every
  fixture format from source, including `residual_rg_<tag>.data` (`rg_entry`, `Types.h:37-51`) and
  the boundary files (flat `<u8`, `BoundaryExtractor.hpp:34-44`). T2 enumerates all nine files
  `match_chunks` consumes with their source line numbers. T3 gives the positional invocation, the
  full env stub, the JSON keys, and the exact extraction boundaries (lines 44-66) for the fallback
  diff. T4 fixes tags, chunk shapes, directory topology, the composite JSON contents, and names the
  final assertion file `remap.data` with its dtype and the resolution procedure.
=========================== END artifacts/plan_v5.md ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then any
questions, then the final READY: line.
