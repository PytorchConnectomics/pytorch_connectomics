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
