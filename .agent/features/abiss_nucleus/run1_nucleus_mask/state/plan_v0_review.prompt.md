You are reviewing an implementation plan for a CCC run. You are the CODER for this run: you
will be the one implementing this plan, so review it for executability.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge the plan on:
1. Is it executable as written? Are the named files, line references, and mechanisms concrete
   enough to implement without re-deriving the design?
2. Is anything missing that would make the implementation silently wrong (especially the
   distributed chunk-hierarchy plumbing and the default-path invariance requirement)?
3. Is the verification plan real and sufficient, or does it assert rather than prove?
4. Is the scope right - anything included that should be cut, or excluded that must be in?

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

=========================== BEGIN artifacts/plan_v0.md ===========================
# Plan v0

## Summary

Add a `NUC_PATH` input to ABISS carrying a **uint32 nucleus instance segmentation**, and use
it as a hard cannot-link constraint during agglomeration: two clusters that carry different
nucleus ids are never merged, at any affinity.

The feature is built as a **parallel payload channel** to the existing semantic-class
channel, not as a reuse of it. Every stage of the existing sem plumbing has a one-to-one
nucleus counterpart, which keeps the diff mechanical and keeps the distributed correctness
guarantee intact: the veto is a pure function of associatively-accumulated per-cluster
state, exactly like the existing size veto, so the chunk-independence property from the
distributed-agglomeration paper still holds.

The on-disk nucleus record is deliberately chosen to be **byte-shaped identically to
`sem_array_t`** (three 8-byte fields), so the existing generic C++ reduce/match code paths
can be instantiated for it with no new serialization logic.

## Scope

**In scope** (repository `lib/abiss`, worked on in the isolated clone `work/abiss`):

* New voxel type and per-cluster record type.
* New `NucExtractor` collecting per-supervoxel nucleus identity.
* Nucleus payload propagation through the whole distributed chunk hierarchy: atomic chunk →
  composite chunk → overlap chunk, including the OVERLAP=2 `vetoed_edges` feedback path.
* Affinity-ungated cannot-link veto in `mean_aggl.cpp`, with refusals logged to a dedicated
  cut file.
* Config/env/driver plumbing so `NUC_PATH` flows from the param JSON to `nuc.raw`.
* A synthetic end-to-end micro-test proving (a) default-path bit-invariance and (b) that the
  veto actually fires.

**Out of scope, deliberately:**

* `SEMANTIC_WS`-style pre-watershed affinity cutting for nuclei (`NUC_WS`). Rationale in
  Risks. The watershed stage is a different binary with its own bit-identity risk, and the
  record stays *safe* without it — a supervoxel straddling two nuclei loses the minority id,
  which costs a veto opportunity but never produces a wrong veto. Instead this plan adds a
  cheap diagnostic counter so we can measure whether `NUC_WS` is worth a follow-up.
* The pytorch_connectomics wrapper (`connectomics/decoding/decoders/abiss.py`).
* Generating the nucleus mask, and the perinuclear-shell tagging preprocessing.
* Fixing the stale `nlabels = 5` drift in `scripts/match_chunks.py` / `scripts/reduce_chunk.py`.
  Investigation during planning established that these Python files are **not invoked by any
  shell driver** — `overlap_chunk_me.sh:44` calls the C++ `$BIN_PATH/reduce_chunk` and
  `composite_chunk_me.sh:42` calls the C++ `$BIN_PATH/match_chunks`. The Python versions are
  dead legacy. They are left untouched and must not be used as templates.

## Proposed Changes

### Phase A — types and the extractor

**A1. `src/seg/Types.h`** — next to `using semantic_t = uint8_t;` (line 25) add:

```cpp
using nuc_t = uint32_t;                 // voxel dtype of nuc.raw

struct __attribute__((packed)) nuc_record_t {
    uint64_t id    = 0;   // dominant nucleus instance id; 0 == untagged
    uint64_t count = 0;   // voxels of the cluster carrying `id`
    uint64_t total = 0;   // voxels of the cluster carrying ANY nonzero nucleus id
};
```

`nuc_record_t` is 24 bytes, the same as `sem_array_t = std::array<size_t,3>`. This is
intentional: it lets A4/A5 reuse the existing generic serialization templates.

`id` is widened to 64-bit inside the record even though the voxel type is 32-bit, so the
record has no internal padding and no alignment traps. The uint32 voxel width is the
user-facing contract; the record width is an implementation detail.

**A2. New file `src/seg/NucExtractor.hpp`** — modeled on `SemExtractor.hpp` but with **no
class LUT** (ids must survive verbatim) and with an **optional/nullable source**, so the
call site in A3 does not have to branch:

* Constructor takes a `const Chunk *` which may be `nullptr`.
* `collectVoxel(Coord c, Tseg segid)`: return immediately if the pointer is null; else read
  `id = (*m_nuc)[c[0]][c[1]][c[2]]`, and if `id != 0`, increment
  `m_counts[segid][id]` where `m_counts` is `MapContainer<Tseg, MapContainer<uint64_t, uint64_t>>`.
  Supervoxels with no nucleus voxels never enter the map, so this stays small.
* `collectBoundary` and `collectContactingSurface`: empty, as in `SemExtractor`.
* `output(chunkMap, filename)`: return immediately if the source was null. Otherwise remap
  each supervoxel id through `chunkMap` exactly as `SemExtractor::output` does
  (`SemExtractor.hpp:31-51`), merge the per-supervoxel id→count maps for supervoxels that map
  to the same target, then reduce each merged map to a `nuc_record_t`
  (`id` = argmax count, `count` = that max, `total` = sum of all counts) and write
  `(seg_t, nuc_record_t)` pairs.
* Tie-break the argmax **deterministically by smallest id** on equal counts. Do not rely on
  hash-map iteration order; `MapContainer` may be `absl::flat_hash_map`, whose order is not
  stable, and a nondeterministic dominant id would silently break the reproducibility this
  repo depends on.
* While reducing, count supervoxels where `total > count` (a supervoxel straddling more than
  one nucleus) and print that count to stdout. This is the diagnostic that tells us whether a
  follow-up `NUC_WS` is warranted.

**A3. `src/seg/atomic_chunk_ME.cpp`** — the file currently duplicates the entire
`traverseSegments<1>(...)` call for the sem-present and sem-absent cases (lines 78-104).
Adding a second optional payload naively would produce four branches. Because `NucExtractor`
is nullable (A2), keep exactly the existing **two** branches and pass `nuc_extractor`
unconditionally in both:

* Before the branch: if `std::filesystem::exists("nuc.raw")`, mmap it as
  `ConstChunkRef<nuc_t, 3>` with the same extents/`fortran_storage_order()` as `sem.raw`
  (mirroring lines 79-84), and construct `NucExtractor` over it; else construct it over
  `nullptr`. The mapped file object must outlive the traversal — declare it in the same scope
  as `seg_file`/`aff_file`, not inside an `if` block.
* Add `nuc_extractor` to both `traverseSegments<1>` packs.
* Call `nuc_extractor.output(map, "ongoing_nuclei_labels.data")` unconditionally after the
  branch (it self-disables when there was no input).

### Phase B — agglomeration

**B1. `src/agg/mean_aggl.cpp`, parameters.** Next to `agglomeration_semantic_heuristic_t`
(line 128) add:

```cpp
struct agglomeration_nucleus_heuristic_t
{
    size_t min_voxel_threshold = 1000;   // ignore an id backed by fewer voxels than this
};
```

and a `nuc_params` member in `agglomeration_param_t` (line 142). Do **not** give it an
`aff_threshold` and do **not** fold it into `heuristics_aff_threshold` (line 148); the
nucleus veto is unconditional and must not perturb the existing frozen-edge logic at lines
686-687, which reads `heuristics_aff_threshold`. Perturbing that value would change
default-path behavior even with no nucleus input.

Make `min_voxel_threshold` overridable from the environment (e.g. `ABISS_NUC_MIN_VOXELS`),
parsed in `main` (lines 1145-1149). Everything in `agglomeration_param_t` is currently a
compile-time constant and `main` only parses `argv[1]`; without this, tuning the threshold
means a rebuild. Keep the default as the value above when the variable is unset.

**B2. Loading.** Add `std::vector<nuc_record_t> nuc_ids;` to `agglomeration_data_t`
(line 156) and a `load_nuc(...)` beside `load_sem` (line 245), called next to line 545 with
filename `ongoing_nuclei_labels.data`. Mirror `load_sem`'s structure, but the duplicate-sid
combination rule differs: `load_sem` sums the arrays (lines 271-273); for nuclei, combine as
*sum the totals, and keep the id with the larger count* (ties → smaller id). Duplicates
should not occur for a well-formed run, but the rule must be defined and deterministic.

An empty or missing file must yield an empty vector, exactly as `load_sem` does
(lines 248-251), so that every nucleus code path is skipped when the feature is unused.

**B3. The veto.** Add:

```cpp
bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b,
                   const agglomeration_nucleus_heuristic_t & p)
{
    if (a.id == 0 || b.id == 0) return true;                      // one side untagged
    if (a.count < p.min_voxel_threshold) return true;             // evidence too weak
    if (b.count < p.min_voxel_threshold) return true;
    return a.id == b.id;
}
```

Insert the call in `agglomerate_cc`'s main loop **immediately after the frozen-edge block
that ends at line 697 and before the semantic check at line 699**, guarded by
`if (!nuc_ids.empty())` and with **no affinity gate**. On refusal, push the edge to a new
`nuc_rg_vector` in `agglomeration_output_t` (line 167), set `e.edge->w = Limits::min()`, and
`continue` — the same shape as the semantic refusal at lines 700-706.

Placing it before the semantic check matters: a nucleus refusal is the stronger statement and
should be the one that gets logged when both would fire.

**B4. Propagation on merge.** Next to the `sem_counts` transform at lines 754-758, add the
nucleus combine, using the same `v0`/`v1`/`s` swap discipline:

* `total` fields add.
* If exactly one side has `id != 0`, the result takes that id and its count.
* If both are nonzero they are equal — the veto guarantees it — so add the counts.
* Assert or `abort()` if both are nonzero and different. That state is unreachable; if it
  ever happens the veto has a hole, and failing loudly beats silently producing a merged
  cell. Guard the aborting branch behind the same `!nuc_ids.empty()` check.

**B5. Serialization out.** In `write_supervoxel_info` (lines 843-911), mirror the
`of_sem_ongoing` / `of_sem_done` pair (lines 848-857, 880-891) with `ongoing_nuc.data` and
`done_nuc.data`, guarded by `!nuc_ids.empty()`. In the output block (lines 940-1127), mirror
`of_sem_cuts` / `sem_cuts.data` (lines 960-961, 1050-1052) with `nuc_cuts.data`, writing the
`(seg_indices[e.v0], seg_indices[e.v1])` pairs from `nuc_rg_vector`.

Both files must be created unconditionally (possibly empty), because the shell drivers `mv`
them and `set -euo pipefail` turns a missing file into a hard failure.

### Phase C — distributed hierarchy

**C1. `src/seg/reduce_chunk.cpp`.** `reduce_sem<T,S>` (line 183) is already generic over the
payload type `S` and hardcodes only the filename. Generalize the filename to a parameter (or
add a thin `reduce_nuc` wrapper reusing `sem_data_t<T,S>`), then call it a second time from
line 228 with `S = nuc_record_t` and the `ongoing_nuclei_labels_%1%.data` /
`reduced_ongoing_nuclei_labels_%1%.data` pair. This is where the byte-identical record shape
pays off — no new serialization code.

If the input file is absent or empty, this must be a no-op that still produces the reduced
output file, since `overlap_chunk_me.sh` will `mv` it.

**C2. `src/seg/match_chunks.cpp`.** Mirror `process_sems` (line 382, called at line 440) with
`process_nucs`, reading `o_ongoing_nuclei_labels.data` and writing
`ongoing_nuclei_labels.data`.

**C3. OVERLAP=2 veto feedback — do not skip this.** In OVERLAP mode the semantic cuts are
re-applied across the overlap boundary: `overlap_chunk_me.sh:50` copies `sem_cuts.data` to
`vetoed_edges_<chunk>.data`, `merge_chunks_me.py:62` merges them, and
`match_chunks.cpp:190-238` removes those edges from the region graph. Without the equivalent,
a nucleus veto made in one round is silently forgotten in the next and the cells merge
anyway.

The cheapest correct wiring is to append nucleus cuts to the **same** `vetoed_edges` stream:
at `overlap_chunk_me.sh:50`, after the existing `cp`, add a `cat nuc_cuts.data >>
vetoed_edges_"$output_chunk".data`. The consumer already dedups and sorts
(`match_chunks.cpp:212-214`), and the record format is the same `(seg_t, seg_t)` pair. No C++
change is needed for this path.

**C4. Shell drivers.** Mirror every `ongoing_semantic_labels` / `done_sem` / `sem_cuts` line
with its nucleus counterpart:

* `scripts/atomic_chunk_me.sh` — `touch ongoing_nuclei_labels.data` (line 36 group),
  `mv done_nuc.data ${output_path}/info/nuclei_labels_"$output_chunk".data` (line 50 group),
  `mv nuc_cuts.data ${output_path}/info/nuc_rejected_edges_"$output_chunk".log` (line 53
  group), `mv ongoing_nuc.data ongoing_nuclei_labels_"$output_chunk".data` (line 60 group).
* `scripts/composite_chunk_me.sh` — the OVERLAP=2 rename at line 40
  (`mv ongoing_nuclei_labels.data o_ongoing_nuclei_labels.data`) plus the same three moves at
  the line 88/97 groups.
* `scripts/overlap_chunk_me.sh` — the `mv reduced_ongoing_nuclei_labels_...` at line 60, the
  `done_nuc.data` move at line 64, the `nuc_cuts.data` move at line 66, and the C3 `cat`.

**C5. Python drivers.**

* `scripts/set_env.py:30` — add `"NUC_PATH"` and `"NUC_MIP"` to the exported `env` list.
* `scripts/cut_chunk_agg.py` — after the `SEM_PATH` block (lines 46-49), add the `NUC_PATH`
  block: `load_data(global_param['NUC_PATH'], mip=global_param['AFF_RESOLUTION'],
  fill_missing=global_param.get('NUC_FILL_MISSING', False))`, then `cut_data` with the same
  `start_coord`/`end_coord` as `seg.raw`, then `save_raw_data("nuc.raw", ...)`.

  **The cutout must be explicitly cast to `numpy.uint32` before saving.** `save_raw_data`
  writes `data.dtype` verbatim (`cut_chunk_common.py:38-50`) and the binary mmaps `nuc.raw`
  as `nuc_t`; a mask stored as uint16 or uint64 would produce a wrong-width file that is
  silently misread. This exact failure mode is already documented in that file for affinity
  (`affinity_dtype()` / `ABISS_AFF_DTYPE`). Raise a clear error if any value exceeds the
  uint32 range rather than truncating.
* `scripts/merge_chunks_me.py:59` and `scripts/merge_chunks_overlap.py:65` — add
  `"ongoing_nuclei_labels"` next to `"ongoing_semantic_labels"`.

### Phase D — documentation

Add a short section to `README.md` covering: the `NUC_PATH` key and its uint32 requirement,
`ABISS_NUC_MIN_VOXELS`, the `nuc_cuts` / `nuc_rejected_edges` outputs, and — importantly —
the caveat that the mask should tag **perinuclear cytoplasm**, not raw nucleus interiors,
because the nuclear envelope is a membrane that the affinity network boundaries, so an id
parked on the interior can end up on a cluster that never joins its soma.

## Files and Areas

| File | Change |
|---|---|
| `src/seg/Types.h` | `nuc_t`, `nuc_record_t` |
| `src/seg/NucExtractor.hpp` | **new**; nullable-source per-supervoxel id accumulator |
| `src/seg/atomic_chunk_ME.cpp` | mmap `nuc.raw`, add extractor to both existing branches, emit `ongoing_nuclei_labels.data` |
| `src/agg/mean_aggl.cpp` | params + env override, `load_nuc`, `nuc_can_merge`, veto, merge propagation, `ongoing_nuc.data` / `done_nuc.data` / `nuc_cuts.data` |
| `src/seg/reduce_chunk.cpp` | instantiate the generic reducer for the nucleus payload |
| `src/seg/match_chunks.cpp` | `process_nucs` |
| `scripts/set_env.py` | export `NUC_PATH`, `NUC_MIP` |
| `scripts/cut_chunk_agg.py` | write `nuc.raw` with an explicit uint32 cast |
| `scripts/atomic_chunk_me.sh` | touch/move nucleus artifacts |
| `scripts/composite_chunk_me.sh` | touch/move/rename nucleus artifacts |
| `scripts/overlap_chunk_me.sh` | move nucleus artifacts; append `nuc_cuts.data` to `vetoed_edges` |
| `scripts/merge_chunks_me.py` | merge `ongoing_nuclei_labels` |
| `scripts/merge_chunks_overlap.py` | merge `ongoing_nuclei_labels` |
| `README.md` | document the feature and the perinuclear caveat |
| `work/test/` (new, not part of the shipped tree) | synthetic fixture + test driver |

Untouched on purpose: `scripts/reduce_chunk.py`, `scripts/match_chunks.py` (dead legacy),
`src/ws/*` (no `NUC_WS` in this run), `CMakeLists.txt` (header-only addition needs no target
change — confirm during implementation and only edit if the build proves otherwise).

## Verification Plan

All commands run inside `work/abiss`, never in the live `lib/abiss` checkout.

**V1 — build.**

```bash
cd work/abiss && mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8
```

Must compile clean. Report any new warnings introduced by the change.

**V2 — default-path bit-invariance (the load-bearing check).** This repo reproduces a Seuron
provenance record, so a silent change to no-nucleus output is a hard failure.

1. Build the **baseline** binaries from `run_start_ref` in a scratch worktree of the clone,
   into a separate build directory.
2. Build the **modified** binaries.
3. Generate a synthetic atomic-chunk fixture with numpy (`work/test/make_fixture.py`):
   `aff.raw` (float32, shape `(x,y,z,3)`, Fortran order), `seg.raw` (uint64 watershed labels),
   an empty `chunkmap.data`, and a `param.txt` in the format read at
   `atomic_chunk_ME.cpp:31-34`. Use a fixed seed. No `nuc.raw`.
4. Run `acme` then `agg` under both builds in separate directories.
5. `cmp` every produced `.data` output byte-for-byte. Any difference fails V2.

**V3 — the veto fires.** Extend the fixture with a `nuc.raw` (uint32) that tags two
well-separated supervoxel groups with ids 1 and 2, connected through a high-affinity path so
that the unconstrained run definitely merges them.

* Run with no `nuc.raw`: the two groups end in one segment.
* Run with `nuc.raw`: they end in two segments, and `nuc_cuts.data` is non-empty and names
  the expected pair.
* The veto must fire even though the connecting edge is high-affinity — this is the
  behavioral difference from the semantic veto and the point of the feature.

**V4 — threshold and pass-through.**

* Set `ABISS_NUC_MIN_VOXELS` above the tagged voxel count; the merge is allowed again.
* Tag only one of the two groups; the merge is allowed (one side untagged).
* Both groups tagged with the *same* id; the merge is allowed.

**V5 — record correctness.** With a supervoxel deliberately straddling ids 1 and 2, confirm
`done_nuc.data` reports `id` = the majority id, `count` = its voxels, `total` = the sum, and
that the "multi-nucleus supervoxel" diagnostic counter reports it.

**V6 — dtype guard.** Feed a uint16 nucleus array and confirm `cut_chunk_agg.py` raises a
clear error rather than writing a short file.

**V7 — driver smoke.** `bash -n` every modified shell script, and confirm no `set -euo
pipefail` driver references a nucleus file that is not guaranteed to exist.

**Not verified in this run, and to be stated as such in `code_v0.md`:** a real multi-chunk
distributed run against actual EM data. The hierarchy plumbing (C1-C4) is verified by
construction and by mirroring, not by execution. Do not claim otherwise.

## Risks and Questions

**R1 — default-path regression is the dominant risk.** The `heuristics_aff_threshold`
computation at line 148 is the specific trap: folding a nucleus `aff_threshold` into that max
would change the frozen-edge behavior at lines 686-687 for *every* run, including runs with
no nucleus input. B1 avoids this by giving the nucleus heuristic no affinity threshold at
all. V2 is the check that catches any other instance of this class of mistake.

**R2 — `!nuc_ids.empty()` must gate every new code path.** The existing code uses
`!sem_counts.empty()` as the feature flag, including inside the frozen-edge condition at
lines 686-687. Note that condition already changes behavior when a semantic payload is
present; the nucleus payload must *not* be added to it, or enabling nuclei would silently
alter frozen-edge handling as a side effect.

**R3 — OVERLAP=2 feedback (C3) is easy to miss and fails silently.** If nucleus cuts are not
fed into `vetoed_edges`, the feature appears to work per-chunk and then quietly loses its
constraints at every hierarchy level. There is no test in this run that exercises the overlap
path, so C3 is verified by code inspection only. This is the weakest link in the plan and
should get explicit reviewer attention.

**R4 — determinism.** `MapContainer` may be `absl::flat_hash_map` (`CMakeLists.txt` sets
`-DUSE_ABSL_HASHMAP` when abseil is found), whose iteration order is not stable across runs.
Any argmax over it needs the explicit id tie-break specified in A2/B2, or the dominant id
becomes nondeterministic and this pipeline's reproducibility guarantee breaks.

**R5 — memory.** `MapContainer<Tseg, MapContainer<uint64,uint64>>` allocates an inner map per
tagged supervoxel. On a volume where nuclei cover a small fraction of voxels this is
negligible; on a pathological input where the mask covers everything it is a per-supervoxel
allocation. Acceptable for the intended use; worth a note in the code.

**R6 — `nuc_t` is uint32 but `seg_t` is uint64.** A nucleus mask produced by the same tooling
that produces segmentations may exceed the uint32 range. C5 must raise rather than truncate.

**Q1 — is `NUC_WS` wanted in this run?** Deliberately excluded above. The diagnostic counter
in A2 is meant to answer this empirically on real data before we spend a change on the
watershed path. If the reviewer disagrees, this is the one scope item worth revisiting.

**Q2 — `min_voxel_threshold` default of 1000.** Chosen as "large enough that mask bleed
across a membrane cannot hard-split a cell, small enough that a genuinely tagged perinuclear
shell always qualifies." It is a guess, not a measured value, which is exactly why B1 makes
it an environment override rather than a compile-time constant.

## Changes Since Previous Plan Version

Initial plan.
=========================== END artifacts/plan_v0.md ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
