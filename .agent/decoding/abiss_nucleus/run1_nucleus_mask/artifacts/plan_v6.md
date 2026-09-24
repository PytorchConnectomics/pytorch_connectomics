# Plan v6

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, used as a
dominant-id cannot-link constraint during agglomeration, under the contract the user fixed on
2026-07-30.

Plan_v6 changes five things relative to plan_v5, all from `plan_v5_review`. Two are arithmetic
hygiene in the Closure proof (K1), two are statements that were stronger than the truth (K2, K3),
one is an internal contradiction about where a function lives (K4), and one completes the fixture
specification (K5). **No design semantics change.** The record, the lattice, the veto, the
hierarchy policy, and the contract are all as in plan_v5.

The properties this plan rests on, stated correctly this time:

> **Invariant D.** No merge performed by agglomeration (1) joins two clusters whose recorded
> dominant nucleus ids differ, (2) joins a CONFLICT cluster to a cluster carrying a recorded
> dominant id, or (3) joins two CONFLICT clusters.
>
> **Closure.** Every reachable record satisfies `state == PROPER => count*den >= num*total` (exact
> integer form of the dominance ratio) and `state != PROPER => count == 0`.
>
> **Bound C — scope corrected (K2).** In a PROPER cluster, the *recorded usable* minority evidence
> is `total - count <= (1 - dominance_ratio) * total`. **This does not bound the cluster's actual
> minority voxels.** Sub-floor supervoxels are absorbed as NONE and contribute real voxels that no
> record accounts for; that mass is reported only in aggregate by `nuc: subfloor_voxels` and is not
> bounded per cluster.
>
> **What Invariant D does not promise.** Minority identities are not tracked at all, at any
> contamination level. `A: 99 id1 + 1 id2 -> PROPER id1` and `B: 99 id1 + 1 id3 -> PROPER id1` may
> merge, joining identities 2 and 3. A mixed supervoxel does not necessarily become CONFLICT:
> `60 id1 + 40 id2` is PROPER id1 at the default ratio, and any supervoxel below `min_tagged` is
> NONE however mixed.

## Scope

**In scope** (`lib/abiss`, in the isolated clone `work/abiss`): record and wire types; `nuc_join`
and `nuc_can_merge`; `NucExtractor`; propagation through the distributed hierarchy including
OVERLAP=2 veto feedback; the veto in `mean_aggl.cpp`; `NUC_PATH` plumbing; documentation carrying
the contract verbatim; executable tests.

**Out of scope, decided by the user:** identity sets and `NUC_WS` — the follow-up if the counters
show mixed or sub-floor supervoxels are common on real data. Also out: `NUC_MIP` (replaced by the
C5 shape assertion), the pytorch_connectomics wrapper, mask generation, perinuclear-shell tagging,
and `scripts/reduce_chunk.py` / `scripts/match_chunks.py`, which are **dead legacy**
(`overlap_chunk_me.sh:44` and `composite_chunk_me.sh:42` invoke the C++ binaries).

## Proposed Changes

### Phase A — types, algebra, extraction

**A1. `src/seg/Types.h`** — beside `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                       // voxel dtype of nuc.raw; 0 == background

enum : uint8_t {
    NUC_STATE_NONE     = 0,   // no usable evidence; count == 0 AND total == 0
    NUC_STATE_PROPER   = 1,   // count*den >= num*total
    NUC_STATE_CONFLICT = 2,   // tagged, no dominant id; count == 0
};

struct __attribute__((packed)) nuc_record_t {
    uint8_t  state = NUC_STATE_NONE;
    uint32_t id    = 0;   // meaningful only when state == PROPER
    uint64_t count = 0;   // voxels backing `id`; 0 unless PROPER
    uint64_t total = 0;   // usable tagged voxels; 0 when state == NONE
};
```

Full uint32 id domain `[1, 0xFFFFFFFF]` usable; only `0` is background. `NONE => total == 0` is
what makes Closure hold; document the property above the struct.

**A2. Wire format** — unchanged from plan_v5:

```cpp
struct __attribute__((packed)) nuc_wire_t {
    seg_t sid; uint8_t state; uint32_t id; uint64_t count; uint64_t total;
};
static_assert(sizeof(nuc_wire_t) == 29);
static_assert(offsetof(nuc_wire_t, sid) == 0 && offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id) == 9 && offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);
```

**A3. The dominance ratio as an exact rational, and checked addition (K1).** Both live in
`Types.h`:

```cpp
struct nuc_ratio_t { uint64_t num = 3, den = 5; };   // default 0.6 == 3/5

// exact: no floating point in the comparison path
inline bool nuc_is_dominant(uint64_t count, uint64_t total, const nuc_ratio_t & r) {
    return static_cast<__uint128_t>(count) * r.den >= static_cast<__uint128_t>(r.num) * total;
}

inline uint64_t nuc_add(uint64_t a, uint64_t b) {      // checked
    if (a > UINT64_MAX - b) {
        std::cerr << "nuc: voxel count overflow: " << a << " + " << b << std::endl;
        std::abort();
    }
    return a + b;
}
```

Plan_v5 compared `count >= dominance_ratio * total` in `double`, which the reviewer showed silently
passes at `total == 2^53 + 1`, and summed `uint64` unchecked, which wraps at `~10^19` voxels. Both
counterexamples are **physically unreachable** — a whole-brain EM dataset is on the order of `10^15`
voxels — so this is proof hygiene, not a data path; but the exact form costs nothing and removes the
hole from the Closure proof rather than arguing around it. `__uint128_t` is a GCC/Clang extension
and this project already requires C++20 with those compilers (`CMakeLists.txt:10`).

`ABISS_NUC_DOMINANCE` is parsed to a rational: reject non-finite values (`NaN`, `inf`) and anything
outside `(0.5, 1.0]`, then set `num = llround(value * 1000)`, `den = 1000`. Reject `NaN` explicitly
rather than relying on comparison behavior, since every comparison with `NaN` is false and a bare
range check would silently accept it.

**A4. `nuc_join`** — in `Types.h`, using checked addition:

```cpp
inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t r;
    r.total = nuc_add(a.total, b.total);
    if (a.state == NUC_STATE_CONFLICT || b.state == NUC_STATE_CONFLICT) {
        r.state = NUC_STATE_CONFLICT;
    } else if (a.state == NUC_STATE_NONE) {
        r.state = b.state; r.id = b.id; r.count = b.count;
    } else if (b.state == NUC_STATE_NONE) {
        r.state = a.state; r.id = a.id; r.count = a.count;
    } else if (a.id == b.id) {
        r.state = NUC_STATE_PROPER; r.id = a.id; r.count = nuc_add(a.count, b.count);
    } else {
        r.state = NUC_STATE_CONFLICT;
    }
    return r;
}
```

**Closure proof, now sound.** Extraction establishes it: NONE gets `(0,0)`; PROPER by
`nuc_is_dominant`; CONFLICT has `count == 0`. `nuc_join` preserves it: a NONE operand contributes
`(0,0)` so the other side's exact inequality carries unchanged; two same-id PROPER operands add both
sides of `count*den >= num*total` (valid because addition is now checked, so no wraparound can
break the inequality); every other combination yields CONFLICT with `count == 0`. ∎

Algebra: the state/id projection is a flat-lattice join — associative, commutative, idempotent on
that projection. `count`/`total` are an additive commutative monoid — associative and commutative,
**not** idempotent. So `nuc_join` is associative and commutative on the full record, which suffices
because every combination site joins records over **disjoint voxel sets**.

**A5. `nuc_can_merge` — declared in `Types.h` (K4).**

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

Plan_v5 was internally inconsistent here: V5 compiled a unit binary that included only `Types.h` and
called this predicate, while B3 and the Files table placed it in `mean_aggl.cpp`. `Types.h` is the
correct owner — it is a pure inline predicate over the record type, with no agglomeration state —
and `mean_aggl.cpp` calls it. Each branch is annotated with the Invariant D clause it implements so
code and contract cannot drift.

**A6. `src/seg/NucExtractor.hpp` (new).** Modeled on `SemExtractor.hpp`; no class LUT; nullable
source so A7 does not branch.

* Constructor takes `const Chunk *` (may be `nullptr`), `nuc_ratio_t`, `min_tagged`.
* `collectVoxel(Coord c, Tseg segid)`: return if null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]` and, if `id != 0`, increment `m_counts[segid][id]`
  (`MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`).
* `collectBoundary`, `collectContactingSurface`: empty.

`output(chunkMap, filename)` remaps supervoxel ids through `chunkMap` and merges the id→count maps
of supervoxels sharing a target, as `SemExtractor::output` does (`SemExtractor.hpp:31-51`). Then per
target supervoxel:

```text
tagged := sum of all counts (via nuc_add)
if tagged == 0 or tagged < min_tagged:              (NONE,     -,      0,         0)
(max_id, max_count) := largest count, ties -> smaller id
if nuc_is_dominant(max_count, tagged, ratio):       (PROPER,   max_id, max_count, tagged)
else:                                               (CONFLICT, -,      0,         tagged)
```

`min_tagged` (`ABISS_NUC_MIN_TAGGED`, default **50**) stops a single bleed voxel in an otherwise
untagged supervoxel from becoming PROPER and casting a veto that hard-splits a real cell.
`dominance_ratio` (`ABISS_NUC_DOMINANCE`, default **0.6**, matching
`agglomeration_semantic_heuristic_t::dominant_signal_ratio`, `mean_aggl.cpp:132`) handles bleed
inside an already-tagged supervoxel.

Tie-break by smaller id: `MapContainer` may be `absl::flat_hash_map` (`CMakeLists.txt:42-49`), whose
iteration order is unstable. This argmax runs **once**, over complete evidence; no downstream stage
selects a winner.

Counters, `nuc:`-prefixed: `conflict_sv`, `minority_sv` (PROPER with `total > count`), `subfloor_sv`,
`subfloor_voxels`. The last two are the only record of sub-floor mass — see Bound C's corrected
scope — and are the measurement that decides whether identity sets or `NUC_WS` are worth a follow-up.

`output()` **always creates the file**, writing zero records when the source was null.

**A7. `src/seg/atomic_chunk_ME.cpp`.** The file duplicates the `traverseSegments<1>(...)` call for
the sem-present and sem-absent cases (lines 78-104); a nullable extractor keeps it at two branches.
Declare the `bio::mapped_file_source` for `nuc.raw` in the same scope as `seg_file`/`aff_file` so it
outlives the traversal; if `std::filesystem::exists("nuc.raw")`, open it and build
`ConstChunkRef<nuc_t, 3>` with the same extents and `fortran_storage_order()` as `sem.raw` (lines
79-84). **Unconditional** size check, never `assert`: on mismatch with
`sizeof(nuc_t)*dim[0]*dim[1]*dim[2]`, print expected and actual bytes and `std::abort()`. Add
`nuc_extractor` to both packs; call `nuc_extractor.output(map, "ongoing_nuclei_labels.data")`
unconditionally after the branch. Parse `ABISS_NUC_DOMINANCE` and `ABISS_NUC_MIN_TAGGED` here.

### Phase B — agglomeration

**B1. No new agglomeration parameters.** `agglomeration_param_t` and `heuristics_aff_threshold`
(line 148) untouched; all tuning is at extraction.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156) and
`load_nuc(...)` beside `load_sem` (line 245), called next to line 545 on
`ongoing_nuclei_labels.data`, reading `nuc_wire_t`. Mirror `load_sem`'s index mapping; empty or
missing yields an empty vector (lines 248-251). Duplicate sids combine with `nuc_join`; CONFLICT is
legitimate, because the file is concatenated across chunks and one boundary-spanning supervoxel
contributes one record per chunk over disjoint voxels.

| Site | Operation | Conflict policy |
|---|---|---|
| `NucExtractor::output` | aggregate, resolve (A6) | CONFLICT valid |
| `load_nuc` duplicates | `nuc_join` | CONFLICT valid |
| `reduce_chunk` collision | `nuc_join` | CONFLICT valid; **counted** |
| `match_chunks` collision | `nuc_join` | CONFLICT valid; **counted** |
| merge propagation (B4) | `nuc_join` | **abort** if `!nuc_can_merge(a,b)` |

**B3. The veto.** Call `nuc_can_merge` (A5) in `agglomerate_cc`'s main loop immediately after the
frozen-edge block ending at line 697 and before the semantic check at line 699, guarded by
`if (!nuc_ids.empty())`, with **no affinity gate**. On refusal push the edge to a new
`nuc_rg_vector` in `agglomeration_output_t` (line 167), set `e.edge->w = Limits::min()`, `continue`
— the shape of the semantic refusal at lines 700-706. A nucleus refusal is logged in preference to
a semantic one.

**Clause 3's cost, stated honestly (K3).** Refusing `CONFLICT + CONFLICT` **does** cost
over-segmentation: two adjacent CONFLICT clusters with a mergeable edge stay two segments instead of
one. Plan_v5 claimed there was "no additional over-segmentation" and "no benefit"; both claims are
withdrawn. The clause is kept because merging two clusters that each already mix nuclei compounds
contamination, and CONFLICT clusters are expected to be rare — `nuc: conflict_sv` measures whether
that expectation holds.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687**, which already
reads `!sem_counts.empty()`; replicating it would make enabling nuclei silently alter frozen-edge
handling.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), same `v0`/`v1`/`s`
swap discipline, guarded by `!nuc_ids.empty()`. Evaluate `nuc_can_merge(a, b)`; if **false**, print
both records and both `seg_indices` values and `std::abort()`. Then combine with `nuc_join`.

This abort is **unreachable through `agg` by construction** — B3 rejects the same pairs earlier. It
is a tripwire against a future edit that breaks that correspondence. V8 tests the predicate directly
and verifies the call site by inspection; it does not claim `agg` can be made to abort on data.

**B5. File lifecycle.** Creation unconditional — `agg` opens and closes `ongoing_nuc.data`,
`done_nuc.data`, `nuc_cuts.data` every run regardless of `nuc_ids.empty()`, and
`NucExtractor::output` always creates `ongoing_nuclei_labels.data`. Required, because the drivers
`mv` these under `set -euo pipefail` and `reduce_chunk.cpp`'s `read_array` aborts on a file it
cannot open. Content conditional on `!nuc_ids.empty()`. Mirror lines 848-857, 880-891, 960-961,
1050-1052.

**B6. Diagnostics.** All `nuc:`-prefixed: the four A6 counters, `load_nuc` duplicate combinations
producing CONFLICT, `reduce_chunk` / `match_chunks` collisions producing CONFLICT. Only B4's
violation aborts, plus `nuc_add`'s overflow guard.

### Phase C — hierarchy and drivers

Unchanged from plan_v5.

**C1. `src/seg/reduce_chunk.cpp`** — nucleus reducer reading `nuc_wire_t`, applying the `sid` remap
from `remap.data`, combining collisions with `nuc_join` and counting those yielding CONFLICT; one
record per sid. Call from line 228 with `ongoing_nuclei_labels_%1%.data` /
`reduced_ongoing_nuclei_labels_%1%.data`. Absent or empty input still produces the output.

**C2. `src/seg/match_chunks.cpp`** — mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data`, writing `ongoing_nuclei_labels.data`, with
C1's collision handling. Per I4, this site must **not** abort.

**C3. OVERLAP=2 veto feedback** — after the existing `cp` at `overlap_chunk_me.sh:50`, add
`cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. `merge_chunks_me.py:62` merges the stream
and `match_chunks.cpp:190-238` removes those edges; the record is the same `(seg_t, seg_t)` pair and
the consumer already sorts and dedups (`match_chunks.cpp:212-214`), so no C++ change is needed.

**C4. Shell drivers.**

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40 plus the same four moves at the
  line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), `done_nuc.data`
  (line 64), `nuc_cuts.data` (line 66), and the C3 `cat`.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"`; not `NUC_MIP`. The loop is `for e in env: if e in
  data:`, so a JSON without `NUC_PATH` exports nothing new; comment it.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49): `load_data` at
  `AFF_RESOLUTION` with `NUC_FILL_MISSING`, `cut_data` with the same `start_coord`/`end_coord` as
  `seg.raw`, validate, `save_raw_data("nuc.raw", ...)`. Assert the nucleus cutout's spatial shape
  equals `seg.raw`'s. Raise on non-integer dtype, on a negative value in a signed dtype, or on any
  value `> 0xFFFFFFFF`; otherwise `astype(numpy.uint32)` losslessly. The cast is mandatory because
  `save_raw_data` writes `data.dtype` verbatim (`cut_chunk_common.py:38-50`) while the binary mmaps
  `nuc.raw` as `nuc_t`.
* `scripts/merge_chunks_me.py:59`, `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"`.

### Phase D — documentation

`README.md`: the `NUC_PATH` key with dtype, range, alignment contracts; `ABISS_NUC_DOMINANCE` and
`ABISS_NUC_MIN_TAGGED`; the three-state record and Closure; the `nuc:` counters and the follow-up
decision they inform; the `nuc_cuts` / `nuc_rejected_edges` outputs; the B4 abort and the `nuc_add`
overflow guard as tripwires; **Invariant D verbatim with all three clauses, clause 3's real
over-segmentation cost, the `99/1` counterexample, and Bound C with its corrected scope — including
that it does not bound actual minority voxels**; and the caveat that the mask should tag
**perinuclear cytoplasm**, not raw nucleus interiors.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, state enum, `nuc_record_t`, `nuc_wire_t` + static_asserts, `nuc_ratio_t`, `nuc_is_dominant`, `nuc_add`, `nuc_join`, **`nuc_can_merge`**, Closure comment |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, exact dominance + `min_tagged`, four counters |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw` + unconditional size check, extractor in both branches, env parsing |
| `src/agg/mean_aggl.cpp` | `load_nuc`, veto calling `nuc_can_merge`, `nuc_join` propagation + predicate abort, counters, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | nucleus reducer; `nuc_join` + counter |
| `src/seg/match_chunks.cpp` | `process_nucs`; `nuc_join` + counter; never aborts |
| `scripts/set_env.py`, `cut_chunk_agg.py` | `NUC_PATH`; `nuc.raw` with shape/dtype/range validation |
| `scripts/atomic_chunk_me.sh`, `composite_chunk_me.sh`, `overlap_chunk_me.sh` | nucleus artifacts; `cat nuc_cuts.data >> vetoed_edges_*` |
| `scripts/merge_chunks_me.py`, `merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | Invariant D, Closure, Bound C scope, counters, caveats |
| `work/test/` | fixtures, `test_nuc_algebra.cpp`, hierarchy scripts |

Untouched: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy); `src/ws/*`;
`CMakeLists.txt` (header-only addition needs no target change — confirm, edit only if the build
proves otherwise).

## Verification Plan

**All paths are relative to `work/abiss`, the working directory for every command.**

**Fixture layouts — complete (K5).** Little-endian, `align=False`, all packed:

| File | numpy dtype | size | Source |
|---|---|---|---|
| `ongoing_nuclei_labels*.data`, `ongoing_nuc.data`, `done_nuc.data` | `[('sid','<u8'),('state','u1'),('id','<u4'),('count','<u8'),('total','<u8')]` | 29 | A2 |
| `input_rg.data`, `residual_rg_<tag>.data`, `edges_<tag>.data`, `new_edges.data`, `o_residual_rg.data`, **`o_incomplete_edges_<tag>.tmp`** | `[('s1','<u8'),('s2','<u8'),('aff','<f4'),('area','<u8')]` | 28 | `rg_entry<seg_t,aff_t>`, `Types.h:37-51`; `match_chunks.cpp:144-145` |
| **`remap.data`, `localmap.data`, `ongoing.data`, `ongoing_<tag>.data`, `extra_remaps.data`** | `[('os','<u8'),('ns','<u8')]` | 16 | `std::pair<T,T>`, `split_remap.cpp:62`; `reduce_chunk.cpp:24` |
| `ongoing_segments.data`, `done_segments.data`, `ns.data`, `ongoing_seg_size.data`, **`o_ongoing_supervoxel_counts.data`** | `[('sid','<u8'),('size','<u8')]` | 16 | `size_data_t`; `match_chunks.cpp:342` |
| `boundary_<face>_<tag>.data`, `frozen.data` | `'<u8'` flat | 8 | `BoundaryExtractor.hpp:34-44` |
| `matching_faces.data`, `o_boundary_<i>_<tag>.tmp` | `[('oid','<u8'),('boundary_size','<u8'),('nid','<u8'),('agg_size','<u8')]` | 32 | `matching_entry_t<seg_t>`, `Types.h:28-34` |
| `vetoed_edges.data`, `nuc_cuts.data`, `sem_cuts.data` | `[('v0','<u8'),('v1','<u8')]` | 16 | `mean_aggl.cpp:1050-1052` |
| `chunkmap.data` | `[('k','<u8'),('v','<u8')]` | 16 | `Utils.hpp:53-66` |

**`chunk_offset.txt`** is a text file holding the decimal `ac_offset` and nothing else, as written by
`cut_chunk_agg.py`. **`param.txt`** is three lines: `offset[0..2]`, `dim[0..2]`, `ac_offset`
(`atomic_chunk_ME.cpp:31-34`). **Any ancillary input a test does not exercise must be created empty
rather than omitted**, because `read_array` aborts on a file it cannot open. The coder must confirm
each layout against the cited source before building a fixture and record the confirmation in
`code_v0.md`.

**V1 — build.** `mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8 &&
cd ..`. Clean compile including A2's `static_assert`s. Report any new warning.

**V2 — default-path invariance.** Baseline binaries from `run_start_ref` in a scratch worktree into
a separate build directory; modified binaries alongside. `work/test/make_fixture.py` (fixed seed),
chunk `(64,64,64)`, tag `0_0_0_0`, no `nuc.raw`. Stage and run exactly as `atomic_chunk_me.sh:34-41`
under both builds in separate directories. **The comparison set is exactly the files the baseline
produced**; each byte-identical (`cmp`). Files produced only by the modified build must exist and be
**empty**. Plus: `set_env.py` on a JSON without `NUC_PATH` exports no `NUC_` variable, and
`cut_chunk_agg.py` against a small local HDF5 volume writes no `nuc.raw`.

**V3 — the veto fires.** Two groups tagged 1 and 2, joined by a high-affinity path, ≥2000 tagged
voxels each. Without `nuc.raw`: one segment. With: two segments and `nuc_cuts.data` naming the pair.

**V4 — pass-through.** One group tagged (allowed); both same id (allowed); one tagged `0xFFFFFFFF`
and one `1` (vetoed — proves the full uint32 domain).

**V5 — algebra, Closure, and the K1 edges.** Standalone binary
`work/test/test_nuc_algebra.cpp`, `#include "../../src/seg/Types.h"`:

```bash
g++ -std=c++20 -O1 -I src work/test/test_nuc_algebra.cpp -o work/test/test_nuc_algebra && \
  work/test/test_nuc_algebra
```

Over `{ NONE(0,0,0), PROPER(id1,c,t), PROPER(id2,c,t), CONFLICT(0,0,t>0) }` with several `(c,t)`
satisfying the exact inequality, for all pairs and triples: commutativity and associativity on the
full record including `(id1,id2,id2)` in both groupings; idempotence on the state/id projection only,
with `nuc_join(a,a).count == 2*a.count` asserted; **Closure** on every result. Plus the specific
cases:

* J1: `NONE(0,0,0) + PROPER(id1,50,50) == PROPER(id1,50,50)` — `total` stays 50.
* K1a: `nuc_is_dominant(2^53, 2^53 + 1, {1,1})` is **false** — the exact integer comparison the
  `double` form got wrong.
* K1b: `nuc_add(UINT64_MAX, 1)` aborts (checked via a death test or a separate expected-failure
  invocation).
* `ABISS_NUC_DOMINANCE` parsing rejects `NaN`, `inf`, `0.5`, `1.1`.

**V6 — resolution.** `5000 id1 + 100 id2` → PROPER id1, `total=5100`, `minority_sv` 1;
`500 id1 + 400 id2` → CONFLICT, `total=900`; 10 tagged voxels → NONE with **`total == 0`**,
`subfloor_sv` 1, `subfloor_voxels` 10. Barrier: CONFLICT fails against PROPER id3 and against
another CONFLICT, succeeds with NONE.

**V7 — input contracts.** uint16 accepted at uint32 width; `> 0xFFFFFFFF` raises; float32 raises;
negative in a signed dtype raises; shape mismatch raises. A **truncated `nuc.raw`** makes `acme`
exit nonzero with the A7 message, against the Release build.

**V8 — the B4 tripwire.** `test_nuc_algebra` asserts `nuc_can_merge` false for
`PROPER id1 + PROPER id2`, `CONFLICT + PROPER`, `CONFLICT + CONFLICT`, and true for the permitted
pairs — Invariant D's three clauses exactly. `code_v0.md` states plainly that the B4 abort is
unreachable through `agg` because B3 rejects the same pairs first, and that its call site is
verified by inspection, not execution.

**V9 — hierarchy binaries.** Scripts under `work/test/`, each command commented with the driver line
it replicates.

* **T1 `../build/reduce_chunk 0_0_0_0`** — build `ongoing_nuclei_labels_0_0_0_0.data`, `remap.data`,
  `ongoing_segments.data`, `done_segments.data`, `residual_rg_0_0_0_0.data`,
  `boundary_{0..5}_0_0_0_0.data` per the table; all other consumed files created empty. Assert sids
  remapped and payloads preserved. Then the conflicting case: records `(sid=100, PROPER id1)` and
  `(sid=200, PROPER id2)` with `remap.data` containing `(100,300)` and `(200,300)` must produce one
  record `sid=300, state=CONFLICT` and a collision count of 1 — not a nonzero exit.
* **T2 `../build/match_chunks 0_0_0_0`** — consumes `matching_faces.data`, `o_residual_rg.data`,
  `o_incomplete_edges_0_0_0_0.tmp`, `vetoed_edges.data`, `o_boundary_{0..5}_0_0_0_0.tmp`,
  `o_ongoing_supervoxel_counts.data`, `o_ongoing_seg_size.data`, `o_ongoing_semantic_labels.data`,
  `o_ongoing_nuclei_labels.data` (`match_chunks.cpp:33,144,145,190,250,342,363,384`); every file the
  test does not exercise is created **empty**. **The concrete canonicalization (K5):**
  `matching_faces.data` carries `matching_entry_t` records `(oid=100, boundary_size=10, nid=200,
  agg_size=10)` so face matching maps sid 100 onto 200, and `o_ongoing_nuclei_labels.data` carries
  `(100, PROPER id1)` and `(200, PROPER id2)`. Assert the output holds one record for sid 200 with
  `state == CONFLICT` and the collision counter at 1 — the legitimate boundary-reconciliation case,
  which must not abort.
* **T3 the real cut chain.** Run `agg` on a nucleus-conflicting fixture so it produces
  `nuc_cuts.data`. Attempt the real driver:
  `OVERLAP=2 UPLOAD_CMD="cp -r" DOWNLOAD_CMD="cp -r" FILE_PATH=<local> IO_SCRATCH_PATH=<local>
  BIN_PATH=<abs build> SCRIPT_PATH=<abs scripts> bash scripts/overlap_chunk_me.sh
  work/test/run/chunk_a.json`, where the JSON carries the keys `chunk_utils.read_inputs` reads:
  `bbox` (6 ints), `ac_offset`, `boundary_flags` (6), `mip_level`, `indices`, `neighbours`. If it
  cannot run, record the exact failing command and its stderr in `code_v0.md`, then execute the
  command sequence extracted from lines 44-66 and `diff` the extracted block against those lines to
  prove no drift. Report which path ran. Then `python3 scripts/merge_chunks_me.py
  work/test/run/composite.json ""` and `../build/match_chunks 0_0_0_0`, asserting the vetoed edge is
  absent from the region graph.
* **T4 two-chunk end-to-end, mandatory.** `work/test/run_hierarchy.sh`, two `(64,64,64)` chunks with
  different nucleus ids under `work/test/run/{chunk_a,chunk_b}`, tags `0_0_0_0` and `0_1_0_0`. Per
  chunk, mirroring `atomic_chunk_me.sh:34-60`: `acme param.txt <tag>`;
  `mv edges_<tag>.data input_rg.data`; six `cat boundary_*` lines;
  `touch ns.data ongoing_semantic_labels.data ongoing_nuclei_labels.data ongoing_seg_size.data`;
  `agg 0.25 input_rg.data frozen.data ns.data`; `cat remap.data >> localmap.data`;
  `split_remap chunk_offset.txt <tag>`; `assort <tag> ""`;
  `mv residual_rg.data residual_rg_<tag>.data`;
  `mv ongoing_segments.data ongoing_supervoxel_counts_<tag>.data`;
  `mv ongoing_nuc.data ongoing_nuclei_labels_<tag>.data`.

  Composite under `work/test/run/composite`, mirroring `composite_chunk_me.sh:37-70`:
  `python3 scripts/merge_chunks_me.py work/test/run/composite.json ""`;
  `mv ongoing.data localmap.data`; `mv residual_rg.data input_rg.data`; `meme 0_0_0_1 ""`;
  `cat new_edges.data >> input_rg.data`; six `cat boundary_*` lines;
  `agg 0.25 input_rg.data frozen.data ongoing_supervoxel_counts.data`. `composite.json` carries
  `mip_level: 1`, `children` naming the two atomic tags, and the union `bbox`.

  **Final assertion, with the id-resolution step spelled out (K5, reviewer question 2).** Child
  supervoxel ids are **not** assumed directly addressable in the composite `remap.data`. Build the
  transitive map: load `localmap.data` (`[('os','<u8'),('ns','<u8')]`) and the composite
  `remap.data` (same layout), compose them in that order, and follow each mapping to a fixed point.
  Resolve every supervoxel of each tagged group through that composed map and assert the two
  groups' representatives **differ**. If composing turns out to be unnecessary because
  `localmap.data` is the identity for these ids, assert that explicitly rather than skipping the
  step. **If T4 cannot be run, the run is `Status: blocked` — not a passing run with a caveat.**

**V10 — driver syntax.** `bash -n` every modified shell script.

Every claim in `code_v0.md` quotes real command output; a step not run is reported as not run.

## Risks and Questions

**R1 — the shipped guarantee is Invariant D**, with the `99/1` counterexample verbatim in the README.
User-approved contract, not an oversight.

**R2 — Bound C does not bound actual minority voxels (K2).** It bounds recorded usable evidence
only. Sub-floor supervoxels absorbed as NONE contribute unaccounted real voxels; that mass appears
only in aggregate as `nuc: subfloor_voxels`. Stated in the README.

**R3 — `min_tagged = 50` and `dominance = 0.6` are unmeasured.** The counters make both visible on
the first real run.

**R4 — three tripwires now abort:** B4's predicate violation and `nuc_add`'s overflow guard. Both are
unreachable on real data — the overflow needs `~10^19` voxels against `~10^15` in a whole brain — so
they convert a future silent corruption into a loud failure rather than adding a runtime risk.

**R5 — V2 proves invariance on a synthetic fixture, not real EM data.** A Seuron provenance re-run
is out of scope and must be stated in `code_v0.md`.

**R6 — T3's real-driver path may still be unrunnable**; the plan requires attempting it, reporting
the exact blocker, and proving the fallback matches by `diff`.

**R7 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per tagged
supervoxel; negligible when nuclei cover a small voxel fraction. Worth a code comment.

**Q1 — is `min_tagged = 50` right?** Both failure directions are counted, so the first real run
answers it.

**Q2 — should the hierarchy collision counters carry per-occurrence detail?** Counts alone cannot
distinguish legitimate boundary reconciliation from a hierarchy defect. Deferred; an env-gated id
list is the natural follow-up.

## Changes Since Previous Plan Version

Responds to `plan_v5_review.md`. All five findings accepted. The reviewer confirmed J2 fixed and
I3/I4 intact; those designs carry forward unchanged. **No design semantics changed in this
revision** — the record, lattice, veto, hierarchy policy, and contract are as in plan_v5.

* **K1 (Closure fails at numeric limits) — fixed with exact arithmetic.** A3 replaces the `double`
  comparison with `nuc_is_dominant`, an exact `__uint128_t` integer form `count*den >= num*total`
  over a rational ratio, and adds `nuc_add`, a checked addition that aborts on `uint64` overflow.
  The reviewer's `total = 2^53 + 1` case is now decided exactly, and the wrapping-addition case
  cannot occur. `ABISS_NUC_DOMINANCE` parsing explicitly rejects `NaN` and `inf`, which a bare range
  check would have silently accepted. V5 tests both counterexamples directly. Recorded honestly:
  these thresholds are physically unreachable — a whole-brain EM dataset is on the order of `10^15`
  voxels against `2^53 ≈ 9·10^15` and `2^64 ≈ 1.8·10^19` — so this closes a hole in the *proof*
  rather than a hole in the pipeline. Answers reviewer question 1: overflow aborts.
* **K2 (Bound C misdescribes actual mass) — scope corrected.** Bound C is now stated explicitly as
  bounding **recorded usable evidence**, with an explicit sentence that it does **not** bound the
  cluster's actual minority voxels, because sub-floor supervoxels absorbed as NONE contribute real
  voxels no record accounts for. The reviewer's `NONE(49 id2) + PROPER(id1,50,50)` case is exactly
  this, and `nuc: subfloor_voxels` is named as the only place that mass is visible. R2 and the
  README carry the correction.
* **K3 (clause-3 justification contains another false claim) — withdrawn.** "No additional
  over-segmentation" and "no benefit" are removed. B3 now states the real cost — two adjacent
  CONFLICT clusters with a mergeable edge stay two segments — and keeps the clause on the narrower
  grounds that merging two already-mixed clusters compounds contamination, with `nuc: conflict_sv`
  measuring how often it bites. This is the third false-precision claim the reviewer has caught in
  this document's own honesty section; each has been withdrawn rather than defended.
* **K4 (the unit-test seam contradicts the file layout) — fixed.** `nuc_can_merge` is now declared
  in `src/seg/Types.h` (A5) beside `nuc_join`, which is where a pure inline predicate over the
  record type belongs, and `mean_aggl.cpp` calls it. B3, the Files table, and V5/V8 all now agree,
  so the unit binary can see the production predicate.
* **K5 (fixtures still not reproducible) — completed.** The layout table adds
  `o_incomplete_edges_<tag>.tmp` (`rg_entry`, `match_chunks.cpp:145`),
  `o_ongoing_supervoxel_counts.data` (`size_data_t`, `match_chunks.cpp:342`), and
  `localmap.data` / `ongoing.data` / `ongoing_<tag>.data` (`std::pair<seg_t,seg_t>`,
  `split_remap.cpp:62`), plus `chunk_offset.txt`'s content and a blanket rule that unexercised
  ancillary inputs are created **empty** rather than omitted. T1 and T2 now give concrete records —
  T2 supplies the `matching_entry_t` that actually canonicalizes sid 100 onto 200 and the two
  nucleus records that collide there. T4 gains the missing `cat remap.data >> localmap.data` step
  and an explicit transitive resolution through `localmap.data` then the composite `remap.data`,
  answering reviewer question 2 without assuming child ids are directly addressable.
