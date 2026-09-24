# Plan v2

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, and use it
as an absolute cannot-link constraint during agglomeration.

Plan_v1 tried to make a two-state record (`id = 0` meaning "untagged", any other value meaning
"this nucleus") carry three distinct meanings. The reviewer showed that fails in three separate
ways at once — the invariant leaks (G1), the combination is non-associative (G2), and remap
coalescence is unhandled (G3) — all from the same root cause: **"no evidence" and "conflicting
evidence" were the same value.**

Plan_v2 separates them. The nucleus state becomes a three-element flat lattice:

```text
BOTTOM   id == NUC_NONE     (0)              no evidence
proper   id in [1, 0xFFFFFFFE]               unambiguous evidence for one nucleus
TOP      id == NUC_CONFLICT (0xFFFFFFFF)     conflicting evidence
```

with the standard lattice join as the one and only combination operation, and TOP acting as a
**merge barrier**. Lattice joins are associative, commutative, and idempotent by construction, so
G2 and G3 dissolve rather than being patched. TOP-as-barrier closes G1's counterexample: a
supervoxel that physically bridges two nuclei can no longer propagate that contamination outward.

Plan_v2 also retracts plan_v1's overclaim. The guarantee that is actually deliverable is:

> **Invariant N'.** For any two distinct nuclei *i* and *j*, no merge performed by agglomeration
> ever places material tagged *i* and material tagged *j* into the same cluster.
>
> Contamination confined *within a single watershed supervoxel* is not repairable at
> agglomeration level — the supervoxel is atomic, and the two cells were already fused before
> agglomeration began. Plan_v2 isolates such supervoxels (they become barriers that merge with
> nothing) and counts them, rather than claiming they do not exist.

That distinction is the honest version of "hard cannot-link", and it is what makes `NUC_WS` an
optional future optimization rather than a hidden correctness dependency.

## Scope

**In scope** (repository `lib/abiss`, worked in the isolated clone `work/abiss`):

* `nuc_t`, the three-state nucleus record, and an explicit fixed-layout wire struct.
* A single shared `nuc_join` used at every combination point.
* `NucExtractor` with extraction-time resolution and diagnostics.
* Payload propagation through the distributed hierarchy including OVERLAP=2 veto feedback.
* Absolute veto in `mean_aggl.cpp` with TOP as a barrier, refusals logged to `nuc_cuts.data`.
* `NUC_PATH` plumbing with strict dtype, range, and alignment contracts.
* Executable tests covering the default path, the veto, the resolution rule, malformed input,
  **and the hierarchy binaries with conflicting many-to-one remaps**.

**Out of scope, with reasons:**

* `NUC_WS`. With TOP-as-barrier, mixed supervoxels are *contained* rather than merely
  *tolerated*, so `NUC_WS` is a recall optimization (it would recover the constraints a barrier
  gives up) and not a correctness dependency. The A3 counters measure, on real data, how much
  recall is at stake before we spend a change on the watershed path.
* `NUC_MIP` — replaced by the C5 shape assertion, which checks the property the knob was meant to
  deliver.
* The pytorch_connectomics wrapper; generating the nucleus mask; perinuclear-shell tagging.
* `scripts/reduce_chunk.py`, `scripts/match_chunks.py` — dead legacy. `overlap_chunk_me.sh:44`
  invokes the C++ `$BIN_PATH/reduce_chunk`; `composite_chunk_me.sh:42` invokes the C++
  `$BIN_PATH/match_chunks`. Do not use them as templates.

## Proposed Changes

### Phase A — the lattice, the wire format, and extraction

**A1. `src/seg/Types.h`** — beside `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                                  // voxel dtype of nuc.raw

inline constexpr uint32_t NUC_NONE     = 0u;             // BOTTOM: no evidence
inline constexpr uint32_t NUC_CONFLICT = 0xFFFFFFFFu;    // TOP: conflicting evidence
// valid nucleus ids are [1, 0xFFFFFFFE]; C5 rejects a mask containing NUC_CONFLICT.

struct __attribute__((packed)) nuc_record_t {
    uint32_t id    = NUC_NONE;
    uint64_t count = 0;   // voxels backing `id`; 0 when id is NONE or CONFLICT
    uint64_t total = 0;   // voxels carrying any proper id; DIAGNOSTIC ONLY, never decides
};
```

`total` must never be read by a decision. Say so in a comment so a later change cannot quietly
promote it.

**A2. The wire format — explicit, because `std::pair` is not a layout contract (G6).**

Plan_v1 read and wrote `std::pair<seg_t, nuc_record_t>`. `std::pair` has implementation-defined
layout, and pairing an 8-byte-aligned `seg_t` with a packed 20-byte record yields 4 bytes of
trailing padding, so the file format would silently be 32 bytes per record rather than 28. Define
the wire record explicitly and use it at **every** producer and consumer:

```cpp
struct __attribute__((packed)) nuc_wire_t {
    seg_t    sid;      // offset 0
    uint32_t id;       // offset 8
    uint64_t count;    // offset 12
    uint64_t total;    // offset 20
};
static_assert(sizeof(nuc_wire_t) == 28);
static_assert(offsetof(nuc_wire_t, sid)   == 0);
static_assert(offsetof(nuc_wire_t, id)    == 8);
static_assert(offsetof(nuc_wire_t, count) == 12);
static_assert(offsetof(nuc_wire_t, total) == 20);
```

The matching numpy dtype for every V7 fixture, stated here so fixtures cannot drift from the
binary (little-endian, which is the only target this code runs on):

```python
NUC_WIRE = numpy.dtype([('sid','<u8'), ('id','<u4'), ('count','<u8'), ('total','<u8')], align=False)
assert NUC_WIRE.itemsize == 28
```

**A3. `nuc_join` — the single combination operation (G2, G3).** Put it next to `nuc_record_t` so
every call site shares one implementation:

```cpp
inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t r;
    r.total = a.total + b.total;
    if (a.id == NUC_CONFLICT || b.id == NUC_CONFLICT) { r.id = NUC_CONFLICT; r.count = 0; }
    else if (a.id == NUC_NONE)      { r.id = b.id;         r.count = b.count; }
    else if (b.id == NUC_NONE)      { r.id = a.id;         r.count = a.count; }
    else if (a.id == b.id)          { r.id = a.id;         r.count = a.count + b.count; }
    else                            { r.id = NUC_CONFLICT; r.count = 0; }
    return r;
}
```

This is the join of a flat lattice with BOTTOM `NUC_NONE` and TOP `NUC_CONFLICT`, so it is
associative, commutative, and idempotent by construction — not by argument. The reviewer's G2
counterexample now behaves: for `a = id1`, `b = id2`, `c = id2`,
`(a ⊕ b) ⊕ c = TOP ⊕ id2 = TOP` and `a ⊕ (b ⊕ c) = id1 ⊕ id2 = TOP`. Equal.

`total` sums unconditionally, which is associative on its own.

**A4. `src/seg/NucExtractor.hpp` (new).** Modeled on `SemExtractor.hpp`, no class LUT, nullable
source so the A5 call site does not branch:

* Constructor takes `const Chunk *` (may be `nullptr`) and the noise floor `min_voxels`.
* `collectVoxel(Coord c, Tseg segid)`: return if the pointer is null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]`, and if `id != NUC_NONE`, increment `m_counts[segid][id]`,
  where `m_counts` is `MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`. Supervoxels with no
  nucleus voxels never enter the outer map.
* `collectBoundary`, `collectContactingSurface`: empty, as in `SemExtractor`.

`output(chunkMap, filename)` first remaps supervoxel ids through `chunkMap` and merges the
id→count maps of supervoxels sharing a target, exactly as `SemExtractor::output` does
(`SemExtractor.hpp:31-51`). This is the only place raw per-id voxel evidence is ever aggregated.
Then, per target supervoxel:

```text
total     := sum of all counts                       (recorded, diagnostic only)
survivors := { id : count >= min_voxels }
|survivors| == 0  ->  (NUC_NONE,     0,             total)
|survivors| == 1  ->  (s,            count[s],      total)
|survivors| >= 2  ->  (NUC_CONFLICT, 0,             total)     # G1 fix: TOP, not BOTTOM
```

The `>= 2` case is the G1 fix. Plan_v1 emitted BOTTOM here, which let the supervoxel merge freely
and carry two nuclei into a third cluster. TOP makes it a barrier instead.

There is no argmax anywhere — that was plan_v0's associativity defect and it is not reintroduced.

Two counters printed to stdout, both needed to size the `NUC_WS` question on real data:

* supervoxels resolved to `NUC_CONFLICT` (above-floor mixing),
* supervoxels where `total > count` (sub-floor minority evidence dropped).

`output()` **always creates the file**, writing zero records when the source was null (see B5).

The noise floor comes from `ABISS_NUC_MIN_VOXELS` (default 100), parsed in `atomic_chunk_ME.cpp`
`main`. It is an extraction parameter; `agg` has no nucleus tunable at all.

**A5. `src/seg/atomic_chunk_ME.cpp`.** The file duplicates the whole `traverseSegments<1>(...)`
call for the sem-present and sem-absent cases (lines 78-104). Because `NucExtractor` is nullable,
keep exactly those two branches and pass `nuc_extractor` in both:

* Declare the `bio::mapped_file_source` for `nuc.raw` in the same scope as `seg_file`/`aff_file`
  so it outlives the traversal. If `std::filesystem::exists("nuc.raw")`, open it and construct
  `ConstChunkRef<nuc_t, 3>` with the same extents and `fortran_storage_order()` as `sem.raw`
  (lines 79-84); else leave the pointer null.
* **Unconditional** size check, never `assert` (G5): if the mapped size differs from
  `sizeof(nuc_t)*dim[0]*dim[1]*dim[2]`, print the expected and actual byte counts to `stderr` and
  `std::abort()`. `assert` is the wrong tool for a data-integrity check regardless of build type.
* Add `nuc_extractor` to both packs; call `nuc_extractor.output(map, "ongoing_nuclei_labels.data")`
  unconditionally after the branch.

### Phase B — agglomeration

**B1. No new agglomeration parameters.** Nothing is added to `agglomeration_param_t`, and
`heuristics_aff_threshold` (line 148) is not touched. The veto is parameterless.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156) and
`load_nuc(...)` beside `load_sem` (line 245), called next to line 545 with
`ongoing_nuclei_labels.data`, reading `nuc_wire_t` records. Mirror `load_sem`'s index-mapping
structure. An empty or missing file yields an empty vector exactly as `load_sem` does (lines
248-251), so every nucleus path is skipped when the feature is unused.

Duplicate sids combine with `nuc_join`. Log a counter of combinations that produced
`NUC_CONFLICT`; that counter is the observable for G3 (see B6).

**Where combination happens, and where it does not** — read from the code so the implementer does
not add it where it does not belong:

| Stage | Operation |
|---|---|
| `NucExtractor::output` | aggregate raw per-id counts, resolve to a record (A4) |
| `reduce_chunk.cpp` `reduce_sem` (line 183) | `sid` remap; plan_v2 adds `nuc_join` on collision (C1) |
| `match_chunks.cpp` `process_sems` (line 382) | `sid` remap; plan_v2 adds `nuc_join` on collision (C2) |
| `load_nuc` | `nuc_join` on duplicate sids |
| merge propagation (B4) | `nuc_join` |

Every one of these is the *same* `nuc_join`. That is the point.

**B3. The veto.**

```cpp
inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    if (a.id == NUC_CONFLICT || b.id == NUC_CONFLICT) return false;  // barrier (G1)
    if (a.id == NUC_NONE     || b.id == NUC_NONE)     return true;
    return a.id == b.id;                                             // absolute, no escape
}
```

Call it in `agglomerate_cc`'s main loop immediately after the frozen-edge block ending at line 697
and before the semantic check at line 699, guarded by `if (!nuc_ids.empty())`, with **no affinity
gate**. On refusal push the edge to a new `nuc_rg_vector` in `agglomeration_output_t` (line 167),
set `e.edge->w = Limits::min()`, and `continue` — the shape of the semantic refusal at lines
700-706.

A nucleus refusal is the stronger statement, so it is logged in preference to a semantic one.

TOP refusing to merge with BOTTOM is deliberate and is what makes the barrier a barrier. The cost
is that a conflicted supervoxel is orphaned as a singleton; that is the containment-optimal
outcome for a supervoxel that provably bridges two cells, and the A4 counter makes the rate
measurable rather than hidden.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687.** That condition
already reads `!sem_counts.empty()` and changes behavior when a semantic payload is present;
replicating it for nuclei would make enabling the feature silently alter frozen-edge handling.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), with the same
`v0`/`v1`/`s` swap discipline, guarded by `!nuc_ids.empty()`: combine with `nuc_join`.

Plan_v1 called for `abort()` if a merge ever combined two different proper ids. Plan_v2 does not
abort. `nuc_join` yields `NUC_CONFLICT`, which is a safe, containing state, and aborting a
multi-hour distributed run is a worse failure mode than containment. Instead, increment a counter
and print a clear warning naming both ids; B3 makes the case unreachable, so a nonzero count is a
bug signal, not a data condition.

**B5. File lifecycle — creation and content are separate concerns.**

* **Creation is unconditional.** `agg` opens and closes `ongoing_nuc.data`, `done_nuc.data`, and
  `nuc_cuts.data` on every run regardless of `nuc_ids.empty()`; `NucExtractor::output` always
  creates `ongoing_nuclei_labels.data`. Required, not cosmetic: the drivers `mv` these under
  `set -euo pipefail`, and `reduce_chunk.cpp`'s `read_array` aborts on a file it cannot open.
* **Content is conditional.** Records are written only when `!nuc_ids.empty()`.

Mirror the ongoing/done split at lines 848-857 and 880-891, and `of_sem_cuts` at lines 960-961 and
1050-1052.

**B6. Diagnostics summary.** Four counters, all printed with a stable `nuc:` prefix so tests can
grep them: conflicted supervoxels at extraction; `total > count` supervoxels at extraction;
conflicts produced by `load_nuc` duplicate combination; conflicts produced by merge propagation
(expected zero).

### Phase C — distributed hierarchy

**C1. `src/seg/reduce_chunk.cpp`.** `reduce_sem<T,S>` (line 183) is templated on the payload and
hardcodes the filename. Add a nucleus reducer that reads `nuc_wire_t` records, applies the `sid`
remap, and — the G3 fix — **applies `nuc_join` when two source records remap onto the same target
sid**, emitting one record per sid. Call it from line 228 with the
`ongoing_nuclei_labels_%1%.data` / `reduced_ongoing_nuclei_labels_%1%.data` pair.

Plan_v1 claimed a pure `sid` remap was sufficient because that is what `reduce_sem` does for the
semantic payload. That was wrong for nuclei: element-wise summation is safe for a histogram but a
nucleus id is not summable, so a many-to-one remap silently produces one logical cluster carrying
two ids. Collapsing collisions through `nuc_join` at the point they are created keeps the on-disk
invariant "one record per sid, combined by the lattice", which is also what makes T1's assertion
meaningful.

An absent or empty input must still produce the reduced output file.

**C2. `src/seg/match_chunks.cpp`.** Mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data`, writing `ongoing_nuclei_labels.data`, with
the same collision handling as C1.

**C3. OVERLAP=2 veto feedback.** Semantic cuts are re-applied across the overlap boundary:
`overlap_chunk_me.sh:50` copies `sem_cuts.data` to `vetoed_edges_<chunk>.data`,
`merge_chunks_me.py:62` merges them, `match_chunks.cpp:190-238` removes those edges from the region
graph. Without the equivalent, a nucleus veto made in one round is silently forgotten in the next.

Append nucleus cuts to the same stream: at `overlap_chunk_me.sh:50`, after the existing `cp`, add
`cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. The consumer already sorts and dedups
(`match_chunks.cpp:212-214`) and the record is the same `(seg_t, seg_t)` pair, so no C++ change is
needed. T3 executes this chain end to end rather than assuming it.

**C4. Shell drivers.** Mirror every `ongoing_semantic_labels` / `done_sem` / `sem_cuts` line:

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40, plus the same three moves at
  the line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), the
  `done_nuc.data` move (line 64), the `nuc_cuts.data` move (line 66), and the C3 `cat`.

B5's unconditional creation is what makes every one of these `mv`s safe with no nucleus input.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"` to the `env` list; **not** `NUC_MIP`. The existing
  loop is `for e in env: if e in data:`, so a JSON without `NUC_PATH` exports nothing new; state
  that in a comment.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49), add the `NUC_PATH` block:
  `load_data(global_param['NUC_PATH'], mip=global_param['AFF_RESOLUTION'],
  fill_missing=global_param.get('NUC_FILL_MISSING', False))`, then `cut_data` with the same
  `start_coord`/`end_coord` used for `seg.raw`, then validate, then `save_raw_data("nuc.raw", ...)`.

  **Alignment contract (replaces `NUC_MIP`):** read at `AFF_RESOLUTION`, the same mip as affinity
  and watershed, because ids must align voxel-for-voxel with `seg.raw`. Assert the nucleus
  cutout's spatial shape equals the `seg.raw` cutout's spatial shape; raise otherwise.

  **Value and dtype contract:**

  ```text
  non-integer dtype (float, bool, complex)   -> raise
  signed dtype containing a negative value   -> raise
  any value >= 0xFFFFFFFF                    -> raise   (0xFFFFFFFF is reserved for NUC_CONFLICT)
  otherwise                                  -> astype(numpy.uint32), lossless
  ```

  A uint16 mask is accepted by lossless widening; a uint64 mask carrying an out-of-range id is
  rejected. The cast is mandatory because `save_raw_data` writes `data.dtype` verbatim
  (`cut_chunk_common.py:38-50`) while the binary mmaps `nuc.raw` as `nuc_t`; the same
  element-width trap is already documented in that file for affinity (`affinity_dtype()` /
  `ABISS_AFF_DTYPE`).
* `scripts/merge_chunks_me.py:59`, `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"` beside `"ongoing_semantic_labels"`.

### Phase D — documentation

A `README.md` section covering: the `NUC_PATH` key with its uint32, reserved-value, and alignment
contracts; `ABISS_NUC_MIN_VOXELS` acting at extraction rather than agglomeration; the three-state
lattice and what a barrier costs; the four `nuc:` diagnostic counters; the `nuc_cuts` /
`nuc_rejected_edges` outputs; Invariant N' stated precisely, including what it does **not**
promise; and the caveat that the mask should tag **perinuclear cytoplasm**, not raw nucleus
interiors, because the nuclear envelope is a membrane the affinity network boundaries, so an id
parked on the interior can sit on a cluster that never joins its soma.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, `NUC_NONE`/`NUC_CONFLICT`, `nuc_record_t`, `nuc_wire_t` + static_asserts, `nuc_join` |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, extraction-time resolution to the lattice, counters |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw` with an unconditional size check, extractor in both branches, `ABISS_NUC_MIN_VOXELS` |
| `src/agg/mean_aggl.cpp` | `load_nuc`, `nuc_can_merge`, veto, `nuc_join` propagation, counters, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | nucleus reducer with `nuc_join` collision handling |
| `src/seg/match_chunks.cpp` | `process_nucs` with `nuc_join` collision handling |
| `scripts/set_env.py` | export `NUC_PATH` only |
| `scripts/cut_chunk_agg.py` | `nuc.raw` with shape, dtype, and reserved-value validation |
| `scripts/atomic_chunk_me.sh`, `composite_chunk_me.sh`, `overlap_chunk_me.sh` | touch/move nucleus artifacts; append `nuc_cuts.data` to `vetoed_edges` |
| `scripts/merge_chunks_me.py`, `merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | document the feature, Invariant N', and the perinuclear caveat |
| `work/test/` (new, outside the shipped tree) | fixtures and drivers for V1-V8 |

Untouched on purpose: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy);
`src/ws/*` (no `NUC_WS`); `CMakeLists.txt` (header-only addition needs no target change — confirm
during implementation and edit only if the build proves otherwise).

## Verification Plan

All commands run inside `work/abiss`. The live `lib/abiss` checkout is never touched and no build
writes to `lib/abiss/build/`. Every claim in `code_v0.md` must quote real command output; a step
that was not run is reported as not run.

**V1 — build.** `cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8` compiles clean, including the
A2 `static_assert`s. Report any new warning.

**V2 — default-path invariance.** This repo reproduces a Seuron provenance record, so a silent
change to no-nucleus output is a hard failure.

1. Build **baseline** binaries from `run_start_ref` in a scratch worktree of the clone, into a
   separate build directory; build **modified** binaries.
2. `work/test/make_fixture.py` (fixed seed) writes `aff.raw` (float32, `(x,y,z,3)`, Fortran
   order), `seg.raw` (uint64), `chunkmap.data`, and `param.txt` in the format read at
   `atomic_chunk_ME.cpp:31-34`. No `nuc.raw`.
3. Run `acme` then `agg` under both builds in separate directories.
4. **The comparison set is exactly the set of files the baseline run produced.** Every one must
   be byte-identical (`cmp`).
5. Files produced only by the modified build must exist and be **empty** (assert size 0):
   `ongoing_nuclei_labels.data`, `ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`.
6. Driver-level check: `set_env.py` on a param JSON without `NUC_PATH` exports no `NUC_`
   variable; `cut_chunk_agg.py` against a small local HDF5/zarr volume writes no `nuc.raw` and
   leaves the other raw outputs unchanged.

**V3 — the veto fires.** Extend the fixture with a uint32 `nuc.raw` tagging two supervoxel groups
ids 1 and 2, joined by a **high-affinity** path so the unconstrained run definitely merges them.
Each group carries at least 2000 tagged voxels; additionally run with `ABISS_NUC_MIN_VOXELS=1` so
the result does not depend on the default.

* Without `nuc.raw`: one segment. With `nuc.raw`: two segments, `nuc_cuts.data` non-empty and
  naming the expected pair.
* The veto fires despite high affinity — the behavioral difference from the semantic veto.

**V4 — pass-through.** Only one group tagged (merge allowed); both tagged with the same id (merge
allowed); both tagged below `ABISS_NUC_MIN_VOXELS` (both BOTTOM, merge allowed).

**V5 — the lattice, tested as a lattice.**

* **Resolution:** a supervoxel straddling ids 1 and 2, both above the floor → record
  `NUC_CONFLICT`, conflicted-supervoxel counter reports 1. One above, one below → record the
  surviving id, `total > count`, second counter reports 1.
* **Barrier:** the `NUC_CONFLICT` supervoxel must fail to merge with an id-3 cluster **and** with
  an untagged cluster. This is the direct regression test for the reviewer's G1 counterexample and
  must be written as such.
* **Algebra:** a unit test over all pairs and triples drawn from
  `{NONE, id1, id2, CONFLICT}` asserting `nuc_join` is commutative, idempotent, and associative —
  including the reviewer's `(id1, id2, id2)` case in both groupings.

**V6 — input contracts.** uint16 mask accepted and written at uint32 width; uint64 mask with a
value `>= 0xFFFFFFFF` raises; float32 raises; signed mask with a negative value raises; nucleus
cutout whose shape differs from `seg.raw`'s raises. Separately, a **truncated `nuc.raw`** makes
`acme` exit nonzero with the A5 message — this tests the unconditional check, not the Python
guard, and must be run against the Release build.

**V7 — hierarchy binaries, executed.**

* **T1 `reduce_chunk`** — hand-build `ongoing_nuclei_labels_<tag>.data` using the A2 numpy dtype,
  plus the `remap.data` / `ongoing_segments.data` / `done_segments.data` inputs the binary
  consumes. Run it. Assert `sid`s are remapped and payloads preserved. **Include a conflicting
  many-to-one case**: two records with proper ids 1 and 2 remapped onto the same target sid must
  produce exactly one output record for that sid with `id == NUC_CONFLICT`. This is the direct
  test for G3.
* **T2 `match_chunks`** — same construction against `o_ongoing_nuclei_labels.data`, including the
  same conflicting many-to-one case.
* **T3 the real cut chain, end to end.** Not a hand-built `vetoed_edges.data`. Run `agg` on a
  nucleus-conflicting fixture so it *produces* `nuc_cuts.data`; apply the actual C3 append from
  `overlap_chunk_me.sh`; apply `merge_chunks_me.py`'s merge step; run `match_chunks`; assert the
  vetoed edge is absent from the resulting region graph. This is the producer → append → merge →
  consumer chain G4 asked for.
* **T4 multi-chunk end-to-end, mandatory.** Run `acme` on two adjacent synthetic chunks with
  different nucleus ids, then `meme`, then `agg`, following the binary sequence and file moves the
  drivers perform, with cloud upload/download stubbed out locally. Assert the two tagged groups
  remain separate in the final remap, and that the constraint survives a second hierarchy level.

  Plan_v1 allowed T4 to be skipped with a note. Plan_v2 does not. The drivers are file shuffling
  plus binary invocations; the cloud steps are `$UPLOAD_CMD`/`$DOWNLOAD_CMD` and can be replaced
  with local copies. If T4 nonetheless cannot be run, that is a `Status: blocked` for human
  decision, **not** a passing run with a caveat.

**V8 — driver syntax.** `bash -n` every modified shell script.

## Risks and Questions

**R1 — Invariant N' is weaker than "hard cannot-link", on purpose.** Contamination inside a single
watershed supervoxel predates agglomeration and cannot be undone there. Plan_v2 contains it and
counts it. Anyone reading the feature as an absolute guarantee will be wrong; the README must say
so, and V5's barrier test is what keeps the containment honest.

**R2 — the barrier costs recall.** A conflicted supervoxel merges with nothing and is orphaned.
Safe, but on a noisy mask it could orphan real tissue. The A4 counter is the measurement; if the
rate is high on real data, `NUC_WS` becomes worth doing, and that is exactly the decision the
counter exists to inform.

**R3 — `NUC_CONFLICT = 0xFFFFFFFF` is a reserved value carved out of the id space.** A mask
legitimately using it is rejected at C5 rather than silently reinterpreted. Valid ids are
`[1, 0xFFFFFFFE]`.

**R4 — `ABISS_NUC_MIN_VOXELS = 100` is unmeasured.** As an extraction-time floor, too large costs
BOTTOM records (lost constraints) and too small costs TOP records (orphaned supervoxels). Neither
corrupts the lattice. It is an environment variable precisely because the right value is data
dependent.

**R5 — V2 proves invariance on a synthetic fixture, not on real EM data.** A full Seuron
provenance re-run is out of scope and must be stated as such in `code_v0.md`.

**R6 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per
tagged supervoxel; negligible when nuclei cover a small voxel fraction. Worth a code comment.

**R7 — this is the terminal plan version under `p2` and it has not been reviewed.** The two
completed review rounds each found real defects, including one that invalidated the central
invariant. The changes in plan_v2 are substantial and unreviewed, which is a live risk to carry
into implementation.

**Q1 — is `NUC_CONFLICT` too aggressive as a full barrier?** A softer variant would let TOP merge
with BOTTOM while refusing all proper ids, trading containment for recall. Plan_v2 chooses full
containment because the failure it prevents (two whole cells fused) is far more expensive in ERL
terms than an orphaned supervoxel. This is the one design choice most worth a second opinion.

**Q2 — should `reduce_chunk`/`match_chunks` collision handling ever be reachable?** C1/C2 apply
`nuc_join` defensively; under B3 the remaps that feed them should never carry conflicting ids. The
counter in B6 will say whether that reasoning holds on real data. Plan_v2 deliberately makes it
safe rather than asserting it is impossible.

## Changes Since Previous Plan Version

Responds to `plan_v1_review.md`. All six findings accepted.

* **G1 (abstention does not establish the invariant) — fixed by the TOP state.** A4's `>= 2
  survivors` case now emits `NUC_CONFLICT` rather than `NUC_NONE`, and B3 refuses every merge
  involving TOP. The reviewer's counterexample is closed: the mixed supervoxel can no longer merge
  into an id-3 cluster, so the 1+2+3 segment is unreachable. V5's barrier test is written directly
  against that counterexample. Plan_v1's overclaimed "Invariant N" is retracted and replaced with
  Invariant N', which states precisely what is and is not guaranteed.
* **G2 (non-associative combination) — fixed by making the operation a lattice join.** Separating
  "no evidence" (BOTTOM) from "conflicting evidence" (TOP) removes the conflation the reviewer
  identified as the root cause. `nuc_join` is associative, commutative, and idempotent by
  construction. The reviewer's `(id1, id2, id2)` counterexample now evaluates to TOP in both
  groupings, and V5 tests it explicitly over all pairs and triples.
* **G3 (many-to-one remaps coalesce ids) — fixed at the point of coalescence.** Plan_v1's claim
  that a pure `sid` remap sufficed is retracted: it is true for a summable histogram and false for
  an id. C1 and C2 now apply `nuc_join` when two records remap onto one sid, so the on-disk
  invariant is "one record per sid, combined by the lattice", and a conflicting collision becomes
  TOP deterministically. T1/T2 test exactly that case. Plan_v2 does not claim such collisions are
  unreachable; it makes them safe and counts them (B6/Q2).
* **G4 (hierarchy only partially exercised) — fixed by rewriting V7.** T1/T2 now include
  conflicting many-to-one mappings; T3 runs the real producer → append → merge → consumer chain
  starting from an `agg`-produced `nuc_cuts.data` instead of a hand-built `vetoed_edges.data`; T4
  is mandatory and multi-level, with an explicit instruction that a blocked T4 is a blocked run
  rather than a caveated pass.
* **G5 (`assert` removed under `NDEBUG`) — remedy adopted in full.** A5 uses an unconditional
  check with a diagnostic and `abort()`, and V6 adds a truncated-`nuc.raw` test against the
  Release build. Recorded in `plan_v1_review.md`: the finding's premise does not hold for this
  repository, because `CMakeLists.txt:11` overrides `CMAKE_CXX_FLAGS_RELEASE` without `-DNDEBUG`,
  so `assert` survives here. The remedy is adopted anyway — depending on a non-standard flag
  override for a data-integrity check is fragile and the unconditional check is free.
* **G6 (wire layout underspecified) — fixed by A2.** An explicit `nuc_wire_t` with `static_assert`
  on its size and every field offset replaces `std::pair<seg_t, nuc_record_t>`, whose layout is
  implementation-defined and which would in fact have carried 4 bytes of trailing padding. The
  exact numpy dtype for fixtures is specified alongside it so tests cannot drift from the binary.

Answers to the reviewer's two questions:

1. **A permanent conflict/barrier state**, not `NUC_WS` and not an exact multi-id payload. A
   barrier delivers containment at fixed cost with no representation change; an exact multi-id
   payload would make the record variable-length and break the fixed-width plumbing that keeps
   this diff mechanical; `NUC_WS` remains available later as a recall optimization, with the A4
   counters sizing it first.
2. **No, and plan_v2 does not attempt that proof.** Under B3 a conflicting remap should be
   unreachable, but rather than rest on that, C1/C2 apply `nuc_join` at every many-to-one mapping
   so a collision resolves to TOP deterministically, and B6 counts occurrences so the assumption
   is measured rather than assumed.
