You are reviewing revision v1 of an implementation plan for a CCC run. You are the CODER: you
will implement this plan, so review it for executability.

You previously reviewed plan_v0 and returned READY: no with eleven [major] findings. Plan_v1 is
the planner's response. Your primary job is to judge whether those findings are actually fixed
(not merely acknowledged), and whether the fixes introduced new problems.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge plan_v1 on:
1. Does each of F1-F11 get a real fix? Name any that are unresolved or only partially resolved.
2. Does the new "Invariant N" design actually hold? Try to break it. In particular: is there any
   reachable path that puts two different nonzero nucleus ids into one cluster record?
3. Is it executable as written, without re-deriving the design?
4. Is the verification plan now sufficient, especially for the distributed hierarchy?
5. Anything new the revision broke.

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

=========================== BEGIN artifacts/plan_v0.md (superseded, for reference) ===========================
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

=========================== BEGIN artifacts/plan_v0_review.md (your prior review) ===========================
# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v0_review.review.raw.md`.

The reviewer returned `READY: no` with eleven findings, all tagged `[major]`, and no minor
findings. The verdict summary: "The plan is not executable safely yet. Its nucleus-state
invariant breaks under the proposed threshold, distributed combination is underspecified, and
verification does not exercise the hierarchy where silent constraint loss is most likely."

Three of the findings (F1, F2, F3) are the same root defect seen from different angles:
`min_voxel_threshold` was applied at *merge* time, which legalizes clusters holding two
different nucleus ids, which in turn destroys the exactness of the fixed-width record and the
associativity of its combination. The remaining findings are genuine internal contradictions
in the plan (file creation, dtype contract, `NUC_MIP`, id width) and real gaps in the
verification plan.

All findings are accepted. None are softened or discarded.

## Findings

**F1 [major] — B3 and B4 directly conflict.** `nuc_can_merge` permits different nonzero ids
whenever either count is below `min_voxel_threshold`, but B4 then aborts whenever a merge
combines different nonzero ids. The plan simultaneously permits and treats-as-unreachable the
same state.

**F2 [major] — the fixed record is not exact under the proposed behavior.** Without `NUC_WS`,
A2 discards minority ids in a mixed supervoxel, and threshold-permitted merges can also create
mixed-id clusters. Once reduced to `(dominant id, count, total)`, the discarded identities can
never participate in a later veto, so the "hard cannot-link" claim is not actually delivered.

**F3 [major] — the winner-selection reduction is not associative.** Concrete counterexample
from the reviewer: records `(id=1,count=6)`, `(id=2,count=10)`, `(id=1,count=6)` should make id
1 dominant after aggregation (12 vs 10), but pairwise "keep the larger count" retains id 2.
Chunk processing order can therefore change the veto state, which breaks the
chunking-independence property the design claims to preserve.

**F4 [major] — C1/C2 cannot reuse the semantic reduction merely because both records occupy 24
bytes.** Semantic payloads support element-wise summation; nucleus ids do not. A valid
combination operation must be defined and implemented at every remapping/reduction point,
including `reduce_chunk` and `match_chunks`. The reviewer notes the current record may be
incapable of such an operation without a stronger invariant.

**F5 [major] — `NUC_MIP` is exported but never used.** C5 passes `AFF_RESOLUTION` as the `mip`
argument instead. The plan must define the exact source-mip and alignment contract.

**F6 [major] — default-path file handling is contradictory.** B5 guards the nucleus output
streams with `!nuc_ids.empty()`, but also requires the files to exist unconditionally, while C4
unconditionally `mv`s `ongoing_nuc.data`, `done_nuc.data`, and `nuc_cuts.data` under
`set -euo pipefail`. File creation and conditional content must be specified separately.
Adding optional keys to `set_env.py` must also explicitly preserve operation when `NUC_PATH`
is absent.

**F7 [major] — the dtype contract is inconsistent.** C5 says valid inputs are cast to `uint32`;
V6 expects a `uint16` input to be rejected. Validation also covers only values above the
maximum, leaving negative, fractional, and non-finite values unspecified.

**F8 [major] — A1 serializes the id as `uint64_t`, contrary to the task's `uint32` contract**
and to success criterion 4. Byte-size equality with `sem_array_t` is not itself a serialization
requirement. One explicit, stable on-disk layout is needed.

**F9 [major] — the verification plan explicitly declines to execute the distributed
hierarchy.** Syntax checks and "mirroring by construction" do not prove that nucleus records
survive `reduce_chunk` / `match_chunks`, nor that OVERLAP=2 veto feedback actually prevents a
later merge. Executable tests of reduction, matching, veto-stream ingestion, and final
segmentation are required.

**F10 [major] — V2 does not fully prove default-path invariance.** The modified run
intentionally creates additional `.data` files, so "cmp every produced `.data` output" has no
defined one-to-one comparison set. V2 also bypasses config, `set_env.py`, cutout generation,
and the shell drivers.

**F11 [major] — V3 does not ensure both tagged groups clear the default 1000-voxel
threshold.** As written the primary veto test may not exercise the veto at all.

## Questions

The reviewer asked three questions, all of which are load-bearing for plan_v1:

1. Should different ids always veto, or may weak conflicting ids merge? If weak merges are
   allowed, what state preserves all evidence for later hierarchy levels?
2. Is `NUC_WS` required to establish the one-id-per-supervoxel invariant, or should the payload
   representation handle mixed supervoxels exactly?
3. Does default-path invariance permit new empty nucleus sidecar files, provided all
   pre-existing outputs remain byte-identical?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v0_review.md ===========================

=========================== BEGIN artifacts/plan_v1.md (UNDER REVIEW) ===========================
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
=========================== END artifacts/plan_v1.md ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then any
questions, then the final READY: line.
