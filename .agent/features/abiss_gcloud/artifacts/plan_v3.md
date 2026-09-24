# Plan v3

## Summary

Reading two more ABISS scripts answered three of the four majors outright and
exposed a contradiction inside `plan_v2` that no review caught.

**The contradiction:** `plan_v2` §4 put scratch on local SSD, and §3 proposed
distributing decode across machines. Those are incompatible. ABISS's per-task
completion flags live in `SCRATCH_PATH` via `CloudFiles`
(`check_task_flag.py` reads `done/{TASK_KEY}.txt`, `update_task_flag.py` writes
it), so local scratch means the shared state that makes distribution safe is not
shared. `plan_v3` resolves it with the mechanism ABISS already supports: Redis
for task flags (`REDIS_SERVER` / `REDIS_DB`, both already honoured), bulk scratch
on local SSD.

**The dispatcher is smaller than v2 thought.** The contract is now pinned exactly:
`PARALLEL_CMD="parallel --halt 2 -j ${ncpus}"`, fed task tokens on stdin, running
`run_wrapper.sh . <op> {}` per line. And `run_wrapper.sh` **already implements
per-task idempotency and resume** — it calls `check_task_flag.py` to skip
completed work and `update_task_flag.py` to record it. So a distributed
replacement needs distribution and `--halt 2` semantics; claim/lease is an
*optimisation against duplicate work*, not a correctness requirement, because
tasks are idempotent and completion is already recorded.

This also corrects `plan_v2` §2, which said resume granularity is the layer. It
is **per chunk**.

**The criterion contradiction dissolves once S3's purpose is stated precisely.**
S3 is an *equivalence* test: it asks whether chunked decoding equals whole-volume
decoding, not which criterion is better. So both sides run at `mean` regardless
of what S1 prefers. S1 (completed, job `3031548`) found `max` 0.5279 vs `mean`
0.5827 over six cubes — a real 0.055 gap — but that is a **quality** question
for card MSIDEPLOY-SCALE-001, and it does not block or alter the equivalence
test.

## Scope

Unchanged: S3 and S4. Out of scope unchanged: which criterion is scientifically
preferred, S2, acquisition, retraining, a 100 µm run, and publication.

## Proposed Changes

### 1. The criterion: S3 runs at `mean` on both sides, and says why

*Finding 1.* S3's question is "does chunking change the answer", so the criterion
must be **held fixed**, not chosen. Both decoders run `mean`:

- chunked: `agglomerate_mean_edge` with `AGG_THRESHOLD` in probability space;
- whole-volume: `run_abiss_volume.py --ws-merge-function mean` at the same
  threshold, producing a **new `mean` reference** for `ExPID108_32x_Cortex_L1_01`.
  This is one CPU decode of a 0.26 Gvoxel volume — minutes — and it is what keeps
  §5's comparison coherent.

The existing published `max` layer is **not** the reference for S3 and is not
regenerated.

**Who decides what:** this plan decides that S3 uses `mean` on both sides. It does
**not** decide whether the 100 µm product should be a `mean` segmentation despite
S1's 0.055 VOI gap, nor whether `rlme` (`ac`/`agg`, untested) should be measured
first. Those are card-level scientific decisions, recorded as the run's open
question and owned by Donglai. S3 and S4 are executable without them.

### 2. Affinity is written in probability space

*Finding 2.* Decision: **inference writes the affinity already uncompressed** —
`scale_sigmoid` inverted at write, `p = sigmoid(logit(v)/0.2)` — as the single
`AFF_PATH` artifact both decoders consume.

Why this and not a conversion pass: at 100 µm a second copy is 772 GB and a full
extra read/write. And it is safe for the `max` reference because S1 **measured**
`max`'s monotone invariance rather than assuming it — `max_compressed` and
`max_uncompressed` agreed to four decimals at all five thresholds — so the
whole-volume numbers are unchanged by the representation.

Specified: `float32` for the probability layer (probabilities span ~10⁻³–1 after
inversion and float16 loses resolution at the low end that `mean` weights),
channel order `ch0=Z, ch1=Y, ch2=X` with `AFF_CHANNELS [2,1,0]`, storage chunk =
`seg_chunk_size_xyz`, `voxel_offset`/`volume_size` from `BBOX`.

Validation, in this order: (a) the written layer's min/max lie in `(0,1)`;
(b) a whole-volume `max` decode of the probability layer at the mapped threshold
reproduces the compressed-artifact result to within VOI 0.001 — the S1 invariance
check, re-run as a data-integrity assertion on the real artifact.

### 3. The dispatcher: pinned contract, minimal replacement

*Finding 4.* Pinned from `lib/abiss/scripts/init.sh` line 107 and `run_layer.sh`:

```
PARALLEL_CMD="parallel --halt 2 -j ${ncpus}"
cat "<layer>".txt | $PARALLEL_CMD $SCRIPT_PATH/run_wrapper.sh . "<op>" {}
```

Contract the replacement must honour, and nothing more:

- read one task token per line on **stdin**;
- for each, run the given command with `{}` replaced by the token;
- **`--halt 2`**: on the first task failure, stop dispatching and exit non-zero;
- exit zero only if every dispatched task exited zero.

It is supplied by **setting `PARALLEL_CMD` in the environment**, not by editing
vendored ABISS.

Already provided by ABISS, therefore *not* built here: per-task skip
(`check_task_flag.py`), completion recording (`update_task_flag.py`), and
idempotency (a chunk recomputed overwrites its own output).

Provided by this change: distribution across workers, and the `--halt 2`
propagation across machines. **Claims/leases are optional** — a duplicated task
wastes work but cannot corrupt, because completion is flag-guarded and outputs
are idempotent. A claim marker is a later optimisation, explicitly deferred.

### 4. Shared task state — Redis, not GCS scratch

*Corrects `plan_v2` §4.* Distribution requires the task flags to be visible to
every worker. Two supported backends: `CloudFiles` on `SCRATCH_PATH`, or Redis
via `REDIS_SERVER`/`REDIS_DB`.

Decision: **Redis for flags, local SSD for bulk scratch.** Flags are tiny and
high-frequency — the worst possible object-storage workload and the best Redis
one — while bulk scratch is large and local-only by nature. One small Redis
instance per run, in the compute region.

Consequence: single-VM decode works with **no** Redis (the `CloudFiles`
`SCRATCH_PATH` fallback on local disk is sufficient when there is one machine),
so the equivalence test in §5 does not depend on this.

### 5. Equivalence test

Unchanged from `plan_v2` §6 and still the S3 acceptance gate: VOI computed on
**masked volumes** rather than label subsets; boundary set `B` = whole-volume
labels whose bbox intersects a `CHUNK_SIZE` plane, `I` = the rest; acceptance
`VOI_total(chunked, whole) ≤ 0.01` **and** `VOI_total(B) − VOI_total(I) ≤ 0.01`;
chunk-size invariance at two alignment-valid sizes; one-chunk case is a plumbing
test. Both sides run `mean` per §1.

### 6. Inference distribution

*Finding 3.* Reuses §3's dispatcher and ABISS's flag mechanism rather than
inventing a second scheme:

- **manifest**: block indices enumerated from volume shape and block size, written
  as a task-token file with the same one-token-per-line shape the dispatcher
  already reads;
- **per-block flags**: the same `check_task_flag`/`update_task_flag` keys, prefix
  `infer_`, so a re-run skips completed blocks;
- **in-flight detection**: the dispatcher's `--halt 2` propagation surfaces a
  failing block immediately rather than at the end;
- **barrier before decode**: a standalone check that every manifest token has a
  `done/` flag *and* that the affinity layer is readable and non-empty at each
  block's bounding box. Decode refuses to start otherwise.

### 7. Sizing and cross-region, with the arithmetic shown

*Minor finding 5.* Sizes are decimal GB; the rate is per GiB. Conversion shown:

| artifact | GB | GiB (÷1.024³ ×10⁹/2³⁰ → ×0.9313) | at $0.02/GiB |
|---|---|---|---|
| source at model grid, uint8 | 129 | 120.1 | **$2.40** |
| affinity, 3-ch **float32** (§2) | 1544 | 1437.9 | $28.76 |
| segmentation, uint32 | 516 | 480.6 | $9.61 |

Note the affinity doubled against `plan_v2` because §2 chose float32; this
strengthens §4's policy of never moving it across regions and is called out
rather than buried. Chunk table, defaults and the RSS model are unchanged from
`plan_v2` §5, with per-chunk RSS still labelled a model.

## Files and Areas

| Path | Change |
|---|---|
| `lib/abiss/scripts/init.sh` | **not edited**; `PARALLEL_CMD` overridden via environment |
| `tutorials/neuron_liconn_moe/gcloud/dispatch_tasks.py` | new: §3 contract |
| `tutorials/neuron_liconn_moe/gcloud/chunked_abiss.yaml` | new: config, Redis flags, local scratch |
| `tutorials/neuron_liconn_moe/gcloud/run_volume.sh` | chunked decode stage; criterion + alignment assertions |
| `tutorials/neuron_liconn_moe/gcloud/launch.sh` | worker fan-out; zone→shape→region→on-demand |
| `tutorials/neuron_liconn_moe/gcloud/equivalence_test.py` | new: §5 |
| `tutorials/neuron_liconn_moe/gcloud/size_report.py` | new: §7 |
| inference output path | probability-space precomputed writer + §2 validation |

## Verification Plan

1. `bash -n` on shell changes; chunked config resolves on the dry path.
2. Criterion assertion fails at config resolution for a criterion absent from the
   chunked stage list (`max`).
3. Alignment assertion rejects a mis-aligned `CHUNK_SIZE`/`seg_chunk_size_xyz`.
4. **§2 integrity check:** whole-volume `max` on the probability layer reproduces
   the compressed-artifact result within VOI 0.001.
5. Dispatcher contract: with `-j 1` and one worker it is byte-equivalent to GNU
   parallel on the same task file; a failing task produces non-zero exit and stops
   dispatch (`--halt 2`).
6. Plumbing: one-chunk chunked decode reproduces whole-volume `mean`,
   `VOI_total = 0`.
7. **S3 acceptance:** multi-chunk equivalence at two alignment-valid chunk sizes,
   `VOI_total ≤ 0.01` and `VOI_total(B) − VOI_total(I) ≤ 0.01`.
8. Distribution: N workers vs 1 worker `VOI_total ≤ 0.01`; a killed worker's task
   is re-run and completes; deleting one `done/` flag causes exactly that task to
   re-run.
9. Inference barrier fails a deliberately missing block.
10. Fallback: unavailable shape → alternate reported; only-cross-region → source
    staged, affinity confirmed created in the compute region, cost printed.
11. `size_report.py` 100 µm totals match §7 to two significant figures.

## Risks and Questions

- **The `mean` quality gap is real and unresolved at the card level.** S1
  measured `max` 0.5279 vs `mean` 0.5827. S3/S4 are executable regardless, but a
  100 µm product decoded with `mean` inherits that gap, and `rlme` is untested.
  Owned by Donglai, not by this change.
- **All S1 optima sit at the bottom edge of the swept range**, so S1 establishes
  the *ordering* of criteria at matched thresholds, not their best values. Any
  `AGG_THRESHOLD` taken from S1 should come from an extended sweep.
- **float32 affinity doubles storage** (§7). Justified by `mean` weighting the
  low-probability tail, but it is a real cost and float16 was not measured
  against it.
- **Per-chunk RSS remains a model**, extrapolated from whole-volume measurements.
- **Redis is a new runtime dependency** for distributed decode only; single-VM
  decode does not need it, so it cannot block S3.
- **Publication at 100 µm remains unsolved** and out of scope.

## Changes Since Previous Plan Version

Addresses all 4 major and 1 minor findings in `plan_v2_review.md`, and fixes one
contradiction no review caught.

- **Finding 1 — resolved by stating S3's purpose precisely** (§1). S3 is an
  equivalence test, so the criterion is held fixed at `mean` on both sides and a
  new whole-volume `mean` reference is produced. The quality decision is named,
  assigned to Donglai, and shown not to block S3.
- **Finding 2 — decided** (§2): the affinity is written in probability space at
  inference, consumed by both decoders, with S1's measured invariance re-run as a
  data-integrity assertion on the real artifact. float32 chosen and its cost
  surfaced.
- **Finding 3 — specified** (§6) by reusing §3's dispatcher and ABISS's existing
  flag mechanism: manifest, per-block flags, halt propagation, and an explicit
  pre-decode barrier.
- **Finding 4 — contract pinned, and the work shrank** (§3). `PARALLEL_CMD` read
  from `init.sh` line 107; stdin/`{}`/`--halt 2`/exit-code semantics stated.
  Per-task idempotency and resume already exist in `run_wrapper.sh`, so
  claims/leases are demoted to a deferred optimisation rather than a requirement.
- **Finding 5 (minor) — arithmetic shown** (§7), with the GB→GiB conversion
  explicit.
- **Self-correction, unprompted:** `plan_v2` §4 (local scratch) contradicted
  `plan_v2` §3 (distributed decode), because ABISS's task flags live in
  `SCRATCH_PATH`. Resolved in §4 by using Redis for flags and keeping bulk scratch
  local. `plan_v2` §2's claim that resume is layer-granular is also corrected: it
  is per chunk.
