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
