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
