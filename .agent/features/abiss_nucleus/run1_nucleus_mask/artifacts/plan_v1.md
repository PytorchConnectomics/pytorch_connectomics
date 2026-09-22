# Plan v1

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, and use it
as an **absolute** cannot-link constraint during agglomeration: two clusters carrying different
nucleus ids are never merged, at any affinity, with no threshold escape.

The central change from plan_v0 is where ambiguity is resolved. Plan_v0 put a
`min_voxel_threshold` at *merge* time, which legalized clusters holding two different ids and
thereby destroyed both the exactness of the fixed-width record and the associativity of its
combination (findings F1, F2, F3). Plan_v1 moves all ambiguity resolution to **extraction
time**, where the complete per-supervoxel id→count evidence is still available, and makes the
merge-time veto parameterless and absolute.

That single relocation establishes the invariant the whole design needs:

> **Invariant N.** A recorded nucleus id is unambiguous. No cluster ever holds two different
> nonzero ids.

Invariant N holds by induction: it is established at extraction (a supervoxel with two
surviving ids abstains, `id = 0`) and preserved by every merge (a merge of two different
nonzero ids is refused). Everything downstream follows from it — the record is exact, the
combination rule is "equal or zero" and therefore trivially associative and commutative, and
the abort in the merge path is genuinely unreachable rather than contradictory.

## Scope

**In scope** (repository `lib/abiss`, worked on in the isolated clone `work/abiss`):

* New voxel type, per-cluster record type, and `NucExtractor`.
* Extraction-time ambiguity resolution with a noise floor.
* Nucleus payload propagation through the distributed chunk hierarchy, including the OVERLAP=2
  `vetoed_edges` feedback path.
* Parameterless absolute cannot-link veto in `mean_aggl.cpp`, refusals logged to `nuc_cuts.data`.
* `NUC_PATH` plumbing from param JSON to `nuc.raw`, with a strict dtype contract.
* Executable tests covering the default path, the veto, the record, **and the hierarchy
  binaries** (`reduce_chunk`, `match_chunks`, veto-stream ingestion).

**Out of scope, with reasons:**

* `NUC_WS` (pre-watershed affinity cutting for nuclei). Plan_v0 left this open as Q2; plan_v1
  closes it. The extraction-time rule handles mixed supervoxels **exactly** by abstaining, so
  `NUC_WS` is no longer needed for correctness — only for recovering the constraints that
  abstention gives up. The diagnostic counter in A3 measures how many that is, on real data,
  before we spend a change on the watershed path.
* `NUC_MIP`. Plan_v0 exported it and never used it (F5). Plan_v1 does not introduce it at all;
  see C5 for the alignment contract that replaces it.
* The pytorch_connectomics wrapper; generating the nucleus mask; perinuclear-shell tagging.
* `scripts/reduce_chunk.py` and `scripts/match_chunks.py`. Planning established these are **dead
  legacy**: `overlap_chunk_me.sh:44` invokes the C++ `$BIN_PATH/reduce_chunk` and
  `composite_chunk_me.sh:42` invokes the C++ `$BIN_PATH/match_chunks`. Their `nlabels = 5` drift
  is a symptom of that deadness. Leave them alone; do not use them as templates.

## Proposed Changes

### Phase A — types, extraction, and ambiguity resolution

**A1. `src/seg/Types.h`** — next to `using semantic_t = uint8_t;` (line 25):

```cpp
using nuc_t = uint32_t;                 // voxel dtype of nuc.raw

struct __attribute__((packed)) nuc_record_t {
    uint32_t id    = 0;   // unambiguous nucleus instance id; 0 == untagged or ambiguous
    uint64_t count = 0;   // voxels backing `id`
    uint64_t total = 0;   // voxels carrying any nonzero id; DIAGNOSTIC ONLY, never decides
};                        // packed, 20 bytes
```

The id field is `uint32_t`, matching the task contract and success criterion 4 (F8). Plan_v0
widened it to 64 bits to match `sem_array_t`'s 24-byte shape; that justification was wrong.
`sem_data_t<T,S>` in `reduce_chunk.cpp:12` and the `std::pair<T,S>` reads in `match_chunks.cpp`
are already **templated on the payload type**, so they work for any `S` regardless of size. No
byte-shape coincidence is required or claimed.

`total` is recorded for diagnostics and must never be read by a merge decision. State that in a
comment so a future change does not quietly promote it.

**A2. New file `src/seg/NucExtractor.hpp`** — modeled on `SemExtractor.hpp`, with no class LUT
(ids survive verbatim) and a **nullable source** so the call site in A4 does not branch:

* Constructor takes `const Chunk *` which may be `nullptr`, plus the noise floor `min_voxels`.
* `collectVoxel(Coord c, Tseg segid)`: return immediately if the pointer is null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]` and, if `id != 0`, increment `m_counts[segid][id]`, where
  `m_counts` is `MapContainer<Tseg, MapContainer<uint32_t, uint64_t>>`. Supervoxels with no
  nucleus voxels never enter the outer map.
* `collectBoundary`, `collectContactingSurface`: empty, as in `SemExtractor`.

**A3. The extraction-time resolution rule — this is the core of plan_v1.**

`NucExtractor::output(chunkMap, filename)` first remaps supervoxel ids through `chunkMap` and
merges the id→count maps of supervoxels that share a target, exactly as
`SemExtractor::output` does (`SemExtractor.hpp:31-51`). This is the **only** point in the whole
pipeline where evidence for different ids is ever aggregated, and it is the point where the
complete map is available. Then, per target supervoxel:

```text
total   := sum of all counts (recorded, diagnostic only)
survivors := { id : count >= min_voxels }
|survivors| == 0  ->  record (id=0, count=0, total)          # untagged
|survivors| == 1  ->  record (id=s, count=count[s], total)   # unambiguous
|survivors| >= 2  ->  record (id=0, count=0, total)          # AMBIGUOUS: abstain
```

Abstention on `>= 2` survivors is the answer to reviewer question 2 and the fix for F2. A
supervoxel genuinely spanning two nuclei above the noise floor is a mask or watershed error.
Abstaining is the only choice that never produces a *wrong* veto and never leaves evidence
stranded, because no downstream stage retains per-id evidence anyway. Sub-floor ids are dropped
as mask bleed across a membrane; that is a deliberate precision/recall choice, and it is made
once, where it can be made correctly, rather than repeatedly at merge time.

Counters to print to stdout, both needed to judge whether `NUC_WS` is worth a follow-up:

* number of supervoxels that abstained due to `>= 2` survivors,
* number of supervoxels where `total > count` (minority evidence was dropped).

There is **no argmax anywhere in this design.** Plan_v0's argmax was the source of the
associativity defect (F3); "exactly one survivor or abstain" has no ordering dependence at all,
so the reviewer's `(1,6),(2,10),(1,6)` counterexample cannot arise. Note this also removes the
need for the tie-break rule plan_v0 specified against `absl::flat_hash_map` iteration order.

The noise floor comes from `ABISS_NUC_MIN_VOXELS` (default 100), parsed in `atomic_chunk_ME.cpp`
`main`. It is an **extraction** parameter now, not an agglomeration parameter; `agg` has no
nucleus tunable at all.

`output()` must **always create the file**, writing zero records when the source was null. See
F6/B5 for why.

**A4. `src/seg/atomic_chunk_ME.cpp`** — the file duplicates the whole `traverseSegments<1>(...)`
call for the sem-present and sem-absent cases (lines 78-104). Because `NucExtractor` is
nullable, keep exactly those **two** branches and pass `nuc_extractor` in both:

* Declare the `bio::mapped_file_source` for `nuc.raw` in the same scope as `seg_file`/`aff_file`
  so it outlives the traversal. If `std::filesystem::exists("nuc.raw")`, open it and construct
  `ConstChunkRef<nuc_t, 3>` with the same extents and `fortran_storage_order()` as `sem.raw`
  (lines 79-84); else the pointer stays null.
* Assert the mapped size equals `sizeof(nuc_t)*dim[0]*dim[1]*dim[2]`. A wrong-width `nuc.raw` is
  otherwise read as garbage.
* Add `nuc_extractor` to both packs; call
  `nuc_extractor.output(map, "ongoing_nuclei_labels.data")` unconditionally after the branch.

### Phase B — agglomeration

**B1. No new parameters.** Plan_v0 added `agglomeration_nucleus_heuristic_t`; plan_v1 deletes
that idea. The veto is parameterless, so nothing is added to `agglomeration_param_t` and
`heuristics_aff_threshold` (line 148) is not touched. This removes the largest default-path
regression risk plan_v0 carried (its own R1).

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t` (line 156)
and `load_nuc(...)` beside `load_sem` (line 245), called next to line 545 with
`ongoing_nuclei_labels.data`. Mirror `load_sem`'s index-mapping structure. An empty or missing
file yields an empty vector exactly as `load_sem` does (lines 248-251), so every nucleus path is
skipped when the feature is unused.

**Duplicate-sid combination rule** (the F4 answer at this layer): duplicates arise from
remapping, not from vetoed merges. Combine as:

```text
total := a.total + b.total
if a.id == 0            -> (b.id, b.count, total)
else if b.id == 0       -> (a.id, a.count, total)
else if a.id == b.id    -> (a.id, a.count + b.count, total)
else                    -> (0, 0, total)      # conflicting: abstain, and log
```

This is associative and commutative: the only non-trivial case is "conflict → abstain", and
abstention is absorbing. Abstain rather than abort here, because a conflict at load time is
reachable in principle (two independently-tagged supervoxels remapped onto one id by the
watershed chunk map) and abstention is the safe direction, consistent with A3.

**Where combination does and does not happen** — established by reading the code, and stated
here so the implementer does not add combination where none belongs:

| Stage | Operation on the payload |
|---|---|
| `NucExtractor::output` | full aggregation + resolution (A3) — the only place |
| `reduce_chunk.cpp` `reduce_sem` (line 183) | **pure `sid` remap**, records written verbatim |
| `match_chunks.cpp` `process_sems` (line 382) | **pure `sid` remap**, records written verbatim |
| `load_nuc` in `mean_aggl.cpp` | duplicate combination, rule above |
| merge propagation (B4) | equal-or-zero, Invariant N |

**B3. The veto.**

```cpp
inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    if (a.id == 0 || b.id == 0) return true;   // one side untagged or ambiguous
    return a.id == b.id;                       // absolute: no threshold escape
}
```

Call it in `agglomerate_cc`'s main loop immediately after the frozen-edge block that ends at
line 697 and before the semantic check at line 699, guarded by `if (!nuc_ids.empty())` and with
**no affinity gate**. On refusal push the edge to a new `nuc_rg_vector` in
`agglomeration_output_t` (line 167), set `e.edge->w = Limits::min()`, and `continue` — the same
shape as the semantic refusal at lines 700-706.

Ordering matters: a nucleus refusal is the stronger statement, so it should be the one logged
when both would fire.

**Do not add the nucleus payload to the frozen-edge condition at lines 686-687.** That condition
already reads `!sem_counts.empty()` and changes behavior when a semantic payload is present;
replicating that for nuclei would make enabling the feature silently alter frozen-edge handling
as a side effect.

**B4. Propagation on merge.** Beside the `sem_counts` transform (lines 754-758), using the same
`v0`/`v1`/`s` swap discipline, guarded by `!nuc_ids.empty()`:

* `total` fields add.
* If exactly one side has `id != 0`, the result takes that id and its count.
* If both are nonzero they are equal by Invariant N, so counts add.
* If both are nonzero and different, `abort()` with a diagnostic naming both ids. Unlike
  plan_v0 this is now genuinely unreachable — B3 has no threshold escape — so the abort is a
  real invariant check rather than a contradiction (F1).

**B5. File lifecycle — explicit, because plan_v0 was self-contradictory here (F6).**

Two separate concerns, specified separately:

* **Creation is unconditional.** `agg` opens and closes `ongoing_nuc.data`, `done_nuc.data`, and
  `nuc_cuts.data` on every run, regardless of `nuc_ids.empty()`. Likewise
  `NucExtractor::output` always creates `ongoing_nuclei_labels.data`. This is required, not
  cosmetic: the shell drivers `mv` these files under `set -euo pipefail`, and
  `reduce_chunk.cpp`'s `read_array` aborts on a file it cannot open.
* **Content is conditional.** Records are written only when `!nuc_ids.empty()`; otherwise the
  files are created empty.

Mirror `SemExtractor`'s pattern for the ongoing/done split (lines 848-857, 880-891) and
`of_sem_cuts` (lines 960-961, 1050-1052).

### Phase C — distributed hierarchy

**C1. `src/seg/reduce_chunk.cpp`.** `reduce_sem<T,S>` (line 183) is templated on the payload and
hardcodes only the filename. Generalize the filename to a parameter (or add a thin wrapper
reusing `sem_data_t<T,S>`) and call it a second time from line 228 with `S = nuc_record_t` and
the `ongoing_nuclei_labels_%1%.data` / `reduced_ongoing_nuclei_labels_%1%.data` pair. This is a
pure `sid` remap — confirmed by reading lines 183-192 — so **no combination operation is
introduced here**, which is the precise answer to F4 at this layer. An absent or empty input
must still produce the reduced output file.

**C2. `src/seg/match_chunks.cpp`.** Mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data` and writing `ongoing_nuclei_labels.data`.
Also a pure `sid` remap — confirmed by reading lines 382-400.

**C3. OVERLAP=2 veto feedback.** In OVERLAP mode semantic cuts are re-applied across the overlap
boundary: `overlap_chunk_me.sh:50` copies `sem_cuts.data` to `vetoed_edges_<chunk>.data`,
`merge_chunks_me.py:62` merges them, and `match_chunks.cpp:190-238` removes those edges from the
region graph. Without the equivalent, a nucleus veto made in one round is silently forgotten in
the next and the cells merge anyway.

Append nucleus cuts to the **same** `vetoed_edges` stream: at `overlap_chunk_me.sh:50`, after
the existing `cp`, add `cat nuc_cuts.data >> vetoed_edges_"$output_chunk".data`. The consumer
already sorts and dedups (`match_chunks.cpp:212-214`) and the record is the same `(seg_t, seg_t)`
pair, so no C++ change is needed. Unlike plan_v0, this path is now **executed** by test T3.

**C4. Shell drivers.** Mirror every `ongoing_semantic_labels` / `done_sem` / `sem_cuts` line:

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group);
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group);
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53 group);
  `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40
  (`mv ongoing_nuclei_labels.data o_ongoing_nuclei_labels.data`) plus the same three moves at
  the line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — `mv reduced_ongoing_nuclei_labels_...` (line 60), the
  `done_nuc.data` move (line 64), the `nuc_cuts.data` move (line 66), and the C3 `cat`.

Because B5 makes creation unconditional, every one of these `mv`s is safe with no nucleus input.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"` to the `env` list. **Do not add `NUC_MIP`** (F5).
  The existing export loop is `for e in env: if e in data:`, so a JSON without `NUC_PATH`
  exports nothing new and the default path is unchanged; state this in the code comment.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49), add the `NUC_PATH`
  block: `load_data(global_param['NUC_PATH'], mip=global_param['AFF_RESOLUTION'],
  fill_missing=global_param.get('NUC_FILL_MISSING', False))`, then `cut_data` with the same
  `start_coord`/`end_coord` used for `seg.raw`, then validate, then
  `save_raw_data("nuc.raw", ...)`.

  **Alignment contract, replacing `NUC_MIP` (F5):** the nucleus volume is read at
  `AFF_RESOLUTION`, the same mip as the affinity and watershed volumes, because the ids must
  align voxel-for-voxel with `seg.raw`. Assert that the nucleus cutout's spatial shape equals
  the `seg.raw` cutout's spatial shape and raise a clear error otherwise. That assertion is
  strictly stronger than a `NUC_MIP` knob, since it checks the property the knob was supposed
  to deliver.

  **Dtype contract, made internally consistent (F7):**

  ```text
  non-integer dtype (float, bool, complex)      -> raise
  signed dtype containing any negative value    -> raise
  any value > 2**32 - 1                         -> raise
  otherwise                                     -> astype(numpy.uint32), lossless
  ```

  So a uint16 mask is **accepted** (lossless widening) and a uint64 mask carrying an
  out-of-range id is **rejected**. Plan_v0 said "cast" in C5 and "reject uint16" in V6; this is
  the single contract, and V6 now tests exactly it. The cast is mandatory because
  `save_raw_data` writes `data.dtype` verbatim (`cut_chunk_common.py:38-50`) while the binary
  mmaps `nuc.raw` as `nuc_t`; the same element-width trap is already documented in that file for
  affinity (`affinity_dtype()` / `ABISS_AFF_DTYPE`).
* `scripts/merge_chunks_me.py:59` and `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"` beside `"ongoing_semantic_labels"`.

### Phase D — documentation

A `README.md` section covering: the `NUC_PATH` key and the uint32/alignment contract;
`ABISS_NUC_MIN_VOXELS` and that it acts at extraction, not agglomeration; the abstention rule
and its two diagnostic counters; the `nuc_cuts` / `nuc_rejected_edges` outputs; and the caveat
that the mask should tag **perinuclear cytoplasm**, not raw nucleus interiors, because the
nuclear envelope is a membrane the affinity network boundaries, so an id parked on the interior
can sit on a cluster that never joins its soma.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t` (uint32), `nuc_record_t` (uint32 id, packed) |
| `src/seg/NucExtractor.hpp` | **new**; nullable source, extraction-time resolution, diagnostics |
| `src/seg/atomic_chunk_ME.cpp` | mmap + size-assert `nuc.raw`, extractor in both branches, `ABISS_NUC_MIN_VOXELS` |
| `src/agg/mean_aggl.cpp` | `load_nuc`, parameterless `nuc_can_merge`, veto, propagation + abort, unconditional file creation |
| `src/seg/reduce_chunk.cpp` | second instantiation of the generic `sid` remap |
| `src/seg/match_chunks.cpp` | `process_nucs` |
| `scripts/set_env.py` | export `NUC_PATH` only |
| `scripts/cut_chunk_agg.py` | `nuc.raw` with shape assertion and strict dtype contract |
| `scripts/atomic_chunk_me.sh` | touch/move nucleus artifacts |
| `scripts/composite_chunk_me.sh` | touch/move/rename nucleus artifacts |
| `scripts/overlap_chunk_me.sh` | move nucleus artifacts; append `nuc_cuts.data` to `vetoed_edges` |
| `scripts/merge_chunks_me.py`, `scripts/merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | document the feature and the perinuclear caveat |
| `work/test/` (new, outside the shipped tree) | fixtures + test drivers for V2-V7 |

Untouched on purpose: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy),
`src/ws/*` (no `NUC_WS`), `CMakeLists.txt` (a header-only addition needs no target change —
confirm during implementation and edit only if the build proves otherwise).

## Verification Plan

All commands run inside `work/abiss`. The live `lib/abiss` checkout is never touched, and no
build ever writes to `lib/abiss/build/`.

**V1 — build.** `cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8` must compile clean. Report any
new warning introduced by the change.

**V2 — default-path invariance (F10 fixed: the comparison set is now defined).** This repo
reproduces a Seuron provenance record, so a silent change to no-nucleus output is a hard failure.

1. Build **baseline** binaries from `run_start_ref` in a scratch worktree of the clone, into a
   separate build directory. Build **modified** binaries.
2. Generate a synthetic atomic-chunk fixture with numpy (`work/test/make_fixture.py`, fixed
   seed): `aff.raw` (float32, `(x,y,z,3)`, Fortran order), `seg.raw` (uint64), `chunkmap.data`,
   and `param.txt` in the format read at `atomic_chunk_ME.cpp:31-34`. No `nuc.raw`.
3. Run `acme` then `agg` under both builds in separate directories.
4. **The comparison set is exactly the set of files the BASELINE run produced.** Every file in
   that set must be byte-identical (`cmp`) between the two runs. This is the pass criterion.
5. Files produced only by the modified build must exist and be **empty** — assert size 0 for
   `ongoing_nuclei_labels.data`, `ongoing_nuc.data`, `done_nuc.data`, `nuc_cuts.data`. This is
   the answer to reviewer question 3: new empty sidecars are permitted, and their emptiness is
   asserted rather than assumed.
6. **Driver-level no-`NUC_PATH` check:** run `set_env.py` on a param JSON with no `NUC_PATH` and
   confirm no `NUC_` variable is exported; run `cut_chunk_agg.py` against a small local
   HDF5/zarr volume (via `open_volume`) and confirm no `nuc.raw` is written and the other raw
   outputs are unchanged.

**V3 — the veto fires (F11 fixed).** Extend the fixture with a uint32 `nuc.raw` tagging two
supervoxel groups with ids 1 and 2, connected through a **high-affinity** path so the
unconstrained run definitely merges them. Each tagged group must contain **at least 2000
voxels**, comfortably above the 100 default floor; additionally run the case with
`ABISS_NUC_MIN_VOXELS=1` so the result does not depend on the default at all.

* Without `nuc.raw`: the two groups end in one segment.
* With `nuc.raw`: two segments, and `nuc_cuts.data` is non-empty and names the expected pair.
* The veto must fire despite the high affinity — this is the behavioral difference from the
  semantic veto and the point of the feature.

**V4 — pass-through cases.** Tag only one group (merge allowed); tag both with the same id
(merge allowed); tag both with ids whose voxel counts are below `ABISS_NUC_MIN_VOXELS` (both
abstain, merge allowed).

**V5 — the resolution rule.** Construct a supervoxel straddling ids 1 and 2:

* both above the floor → record `id = 0`, and the "ambiguous supervoxel" counter reports 1;
* one above, one below → record `id` = the surviving id, `count` = its voxels,
  `total` = the sum, and the `total > count` counter reports 1.

**V6 — dtype and alignment contract (F7 fixed).** uint16 mask → accepted, `nuc.raw` written at
uint32 width; uint64 mask with a value `> 2**32 - 1` → raises; float32 mask → raises; signed
mask with a negative value → raises; nucleus cutout whose shape differs from the `seg.raw`
cutout → raises.

**V7 — hierarchy binaries, executed (F9 fixed).** Plan_v0 declined to execute these; plan_v1
does not.

* **T1 `reduce_chunk`** — hand-build `ongoing_nuclei_labels_<tag>.data` with known records plus
  the `remap.data` / `ongoing_segments.data` / `done_segments.data` inputs it consumes, run the
  binary, and assert `reduced_ongoing_nuclei_labels_<tag>.data` has the `sid` fields remapped
  and every payload byte preserved verbatim.
* **T2 `match_chunks`** — hand-build `o_ongoing_nuclei_labels.data` and the remap inputs, run the
  binary, assert `ongoing_nuclei_labels.data` is remapped and payload-preserved.
* **T3 veto-stream ingestion** — build a `vetoed_edges.data` containing a nucleus cut and an
  `input_rg.data` containing that same edge, run `match_chunks`, and assert the edge is absent
  from the resulting region graph. This is the C3 path, which plan_v0 left verified by
  inspection only and flagged as its own weakest link.
* **T4 two-chunk end-to-end** — run `acme` on two adjacent synthetic chunks, then `meme`, then
  `agg`, and assert the two nucleus-tagged groups remain separate in the final remap. If the
  composite path cannot be driven locally without cloud storage or redis, **record the exact
  blocking command and its error output in `code_v0.md` and state plainly that multi-chunk
  coverage was not achieved.** Do not substitute inspection for execution and do not describe
  T4 as passing.

**V8 — driver syntax.** `bash -n` every modified shell script.

Every verification claim in `code_v0.md` must quote real command output. A step that was not run
is reported as not run.

## Risks and Questions

**R1 — Invariant N is the whole design; if any code path can violate it, the record silently
becomes lossy again.** The paths that could: a threshold reintroduced at merge time, a
combination rule that picks a winner instead of abstaining, or a new aggregation point
downstream of `NucExtractor::output`. The B2 table exists to make that last one visible. The
abort in B4 is the runtime tripwire.

**R2 — abstention silently gives up constraints.** A supervoxel spanning two nuclei above the
floor contributes no veto. This is safe (it never produces a wrong split) but it is a real loss
of recall, and its magnitude is unknown until measured on data. The two counters in A3 exist
precisely to measure it, and they are the evidence that decides whether `NUC_WS` is worth a
follow-up.

**R3 — `ABISS_NUC_MIN_VOXELS` default of 100 is a guess.** It is now an extraction-time noise
floor rather than a merge gate, so an over-large value costs abstentions rather than corrupting
the invariant — a much less dangerous failure direction than plan_v0's. Still unmeasured.

**R4 — T4 may not be runnable locally.** The composite drivers assume cloud paths and redis. The
plan requires the coder to attempt it and report the blocker honestly rather than assert
coverage. If T4 is blocked, hierarchy confidence rests on T1-T3, which do execute the actual
binaries and cover the payload's whole journey except the final assembly.

**R5 — default-path invariance is proven on a synthetic fixture, not on real EM data.** V2 is a
strong structural check but is not a Seuron reproduction. A full provenance re-run is out of
scope for this run and should be stated as such.

**R6 — memory.** `MapContainer<Tseg, MapContainer<uint32,uint64>>` allocates an inner map per
tagged supervoxel. Negligible when nuclei cover a small voxel fraction; worth a code comment for
the pathological case.

**Q1 — the noise floor's default.** 100 voxels is chosen as "large enough that mask bleed across
a membrane cannot claim a supervoxel, small enough that a genuine perinuclear tag always
qualifies." It is a guess, which is why it is an environment variable.

**Q2 — should abstention be logged per-supervoxel rather than counted?** Counters are proposed to
keep the diff small. If the reviewer wants to debug specific abstentions on real data, an
optional id list behind an env flag would be the follow-up.

## Changes Since Previous Plan Version

Every finding in `plan_v0_review.md` is addressed below. All eleven were accepted; none were
softened or discarded.

* **F1 (B3/B4 conflict) — fixed by removing the merge-time threshold entirely.** B3 is now
  parameterless and absolute, so the "different nonzero ids" state B4 aborts on is genuinely
  unreachable. The contradiction is gone rather than papered over.
* **F2 (record not exact) — fixed by A3's resolution rule.** Ambiguity is resolved once, at
  extraction, where the full evidence exists. A supervoxel with two surviving ids abstains
  (`id = 0`) instead of silently discarding the minority, and threshold-permitted mixed clusters
  no longer exist because the threshold no longer permits merges. The record is exact for what
  it now claims: an id is recorded only when unambiguous above the noise floor.
* **F3 (non-associative reduction) — fixed by eliminating the argmax.** "Exactly one survivor or
  abstain" has no ordering dependence, so the `(1,6),(2,10),(1,6)` counterexample cannot arise.
  The `absl::flat_hash_map` tie-break rule plan_v0 needed is no longer necessary.
* **F4 (reduction reuse unjustified) — fixed by reading the code and stating the answer.**
  `reduce_sem` (`reduce_chunk.cpp:183-192`) and `process_sems` (`match_chunks.cpp:382-400`) are
  **pure `sid` remaps** that write payload bytes verbatim; they perform no combination, so no
  combination operation is required there. The reuse is valid because both are templated on the
  payload type, not because of the 24-byte coincidence plan_v0 wrongly cited. The B2 table now
  enumerates every stage and its operation, and the one real combination point (`load_nuc`) has
  an explicit associative rule.
* **F5 (`NUC_MIP` exported but unused) — fixed by not introducing it.** Replaced with an explicit
  alignment contract: read at `AFF_RESOLUTION` and assert the cutout shape matches `seg.raw`.
* **F6 (contradictory file handling) — fixed by separating creation from content in B5.**
  Creation is unconditional (required by `set -euo pipefail` `mv`s and by `read_array`'s abort on
  an unopenable file); only content is gated on `!nuc_ids.empty()`. `set_env.py`'s existing
  `if e in data` loop preserves the default path, now stated explicitly.
* **F7 (inconsistent dtype contract) — fixed with one contract in C5:** reject non-integer,
  reject negative, reject out-of-range, otherwise cast losslessly. uint16 is accepted; V6 now
  tests exactly this instead of contradicting it.
* **F8 (uint64 id vs the uint32 contract) — fixed.** `nuc_record_t::id` is `uint32_t`. The
  byte-shape justification was wrong and is retracted: the reduce/match code is templated on the
  payload type and needs no size coincidence.
* **F9 (hierarchy not executed) — fixed by V7.** T1/T2 execute `reduce_chunk` and `match_chunks`
  on hand-built payload fixtures; T3 executes the OVERLAP=2 veto-stream ingestion that plan_v0
  left as inspection-only; T4 attempts two-chunk end-to-end with a mandatory honest report if it
  is blocked.
* **F10 (V2 comparison undefined) — fixed.** The comparison set is exactly the baseline run's
  output files, all of which must be byte-identical; new files must exist and be asserted empty;
  a driver-level `set_env.py` + `cut_chunk_agg.py` no-`NUC_PATH` check is added.
* **F11 (V3 may not exercise the veto) — fixed.** Tagged groups are at least 2000 voxels, and the
  case is additionally run with `ABISS_NUC_MIN_VOXELS=1` so it does not depend on the default.

Answers to the reviewer's three questions:

1. **Different ids always veto.** No weak-merge escape exists, because the escape was what broke
   the invariant. Weak evidence is filtered at extraction instead, where filtering it is correct.
2. **`NUC_WS` is not required.** A3 handles mixed supervoxels exactly by abstaining. `NUC_WS`
   would only recover the constraints abstention gives up; the A3 counters measure how many that
   is before we spend a change on the watershed path.
3. **Yes, new empty sidecars are permitted**, and V2 step 5 asserts they are empty rather than
   assuming it.
