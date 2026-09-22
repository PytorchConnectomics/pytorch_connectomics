You are reviewing revision v2 of an implementation plan for a CCC run. You are the CODER: you
will implement this plan, so review it for executability.

You reviewed plan_v0 (11 [major] findings) and plan_v1 (6 [major] findings, G1-G6). Plan_v2 is
the planner's response to G1-G6. Your primary job: judge whether G1-G6 are actually fixed, and
whether the redesign introduced new problems.

The redesign replaces the two-state nucleus record with a three-state flat lattice
(BOTTOM = NUC_NONE, proper id, TOP = NUC_CONFLICT), a single shared `nuc_join`, TOP acting as a
merge barrier, and an explicit `nuc_wire_t` wire struct.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Attack it specifically on:
1. Is each of G1-G6 really fixed? Name any that are not.
2. Try to break the lattice. Is `nuc_join` actually associative/commutative/idempotent as claimed,
   including the `total` field? Is there any reachable path that puts material from two distinct
   nuclei into one cluster via an agglomeration merge?
3. Is TOP-as-barrier correct, or does refusing TOP-BOTTOM merges create a new problem?
4. Is the plan executable as written - concrete enough to implement without re-deriving design?
5. Is the verification plan (V1-V8) sufficient, especially T1-T4?
6. Anything the redesign broke that plan_v1 had right.

Be specific: a finding without a concrete failing case or a concrete missing specification is
not useful at this stage.

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

=========================== BEGIN artifacts/plan_v1.md (superseded, for reference) ===========================
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

=========================== BEGIN artifacts/plan_v1_review.md (your prior review) ===========================
# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, reasoning effort `xhigh`, sandbox read-only). Raw transcript:
`state/plan_v1_review.review.raw.md`.

`READY: no`, with six findings, all tagged `[major]`. The reviewer's verdict: "Plan v1 is still
not executable safely. F1, F5-F7, F10, and F11 are fixed. F8 fixes the id width but leaves the
enclosing wire layout underspecified. F2-F4 remain unresolved, and F9 is only partially
addressed. Most importantly, Invariant N does not hold."

Six of the eleven plan_v0 findings are confirmed fixed: F1, F5, F6, F7, F10, F11.

The central result is that **Invariant N does not hold**, and the reviewer supplied a concrete
reachable counterexample rather than a general doubt. The root cause it identifies is precise
and correct: plan_v1's record conflates "no evidence" with "conflicting evidence", using a
single `id = 0` for both. That conflation is simultaneously the hole in the invariant (G1), the
associativity break (G2), and the reason remap coalescence cannot be handled (G3). All findings
are accepted; none are softened or discarded.

One finding (G5) rests on a premise that is factually wrong for this repository. It is recorded
in full below with the correction, and its remedy is adopted anyway because the remedy is
strictly better and free.

## Findings

**G1 [major] — F2 remains unresolved: abstention does not establish Invariant N.** A supervoxel
containing above-floor ids 1 and 2 emits `id=0`, but the underlying cluster still physically
contains both. It can then merge with a cluster carrying id 3, because `nuc_can_merge` treats
zero as unconstrained, and B4 rewrites the record to id 3. The resulting segment physically
contains material from nuclei 1, 2, and 3 while recording only 3. That is a reachable
counterexample to the hard cannot-link guarantee. Abstention avoids choosing a wrong dominant
id, but it neither makes the record exact nor removes the conflicting evidence. The reviewer
names the acceptable remedies: `NUC_WS`, an exact set representation, or **a distinct conflict
state with appropriately restrictive merge behavior**.

**G2 [major] — F3 remains unresolved: B2's combination rule is not associative.** Plan_v1 calls
abstention absorbing but implements zero as an *identity*, `0 + x = x`. The reviewer's
counterexample, for `a = id1`, `b = id2`, `c = id2`:

```text
(a + b) + c = 0 + c = id2
a + (b + c) = a + id2 = 0
```

Remap grouping or input order therefore changes the recorded id and the resulting veto behavior.
The representation conflates "no evidence" with "conflicting evidence", so the required
associative operation cannot be implemented as written.

**G3 [major] — F4 remains unresolved: the "pure sid remaps" can be semantic many-to-one cluster
merges.** Plan_v1 itself admits duplicate sids with conflicting ids can arise from remapping.
`reduce_chunk` / `match_chunks` would then emit `(sid=K, id=1)` and `(sid=K, id=2)` — one logical
cluster already carrying two nonzero ids, with `B3` never seeing a merge edge. A later
`load_nuc` abstention cannot undo that coalescence. The induction must cover every many-to-one
remap: either prove such conflicts unreachable, or veto them before the mapping is accepted.

**G4 [major] — F9 only partially fixed.** T3 starts from a hand-built final `vetoed_edges.data`,
which bypasses the actual C3 path that appends `nuc_cuts.data`, merges the per-chunk streams, and
produces that input. T1/T2 test only byte-preserving remaps and never exercise a conflicting
many-to-one mapping. T4, the sole proposed end-to-end hierarchy test, is permitted to fail and is
not specified through the complete workflow. A sufficient test must cover the real
producer/append/merge/consumer chain plus a multi-level case with different nucleus ids.

**G5 [major] — A4's size check uses `assert`, which the reviewer notes is removed under
`NDEBUG` in Release builds**, so a malformed `nuc.raw` could still be mmaped and silently
misread. An unconditional runtime size check and a direct malformed-file test are needed.

*Coordinator correction, recorded rather than used to soften the finding:* the premise does not
hold for this repository. `CMakeLists.txt:11` overrides the Release flags outright —
`set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -fopenmp")` — dropping CMake's default
`-DNDEBUG`, so `assert` remains live in this project's Release build. The finding's *remedy* is
nonetheless adopted in full: relying on a non-standard flag override for a data-integrity check
is fragile, an unconditional check costs nothing, and the surrounding code already depends on
`assert` for the same purpose. The direct malformed-file test the reviewer asks for is also
added.

**G6 [major] — the stable on-disk layout remains underspecified.** Although `nuc_record_t` is
declared packed at 20 bytes, the hierarchy reads and writes raw `std::pair<seg_t, nuc_record_t>`
records. That pair has implementation-defined layout and likely trailing padding, and the packed
record also contains unaligned `uint64_t` members. The plan states neither the exact pair
size/offsets nor how the hand-built V7 fixtures encode it. An explicit wire record with static
size and offset assertions, used consistently at every producer and consumer, is required.

## Questions

1. Should an above-floor mixed supervoxel require `NUC_WS`, become a permanent conflict/barrier
   state, or force an exact multi-id payload?
2. Can the planner prove that every hierarchy remap is incapable of mapping different recorded
   ids onto one sid? If not, where is that remap vetoed before the ids are coalesced?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v1_review.md ===========================

=========================== BEGIN artifacts/plan_v2.md (UNDER REVIEW) ===========================
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
=========================== END artifacts/plan_v2.md ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then any
questions, then the final READY: line.
