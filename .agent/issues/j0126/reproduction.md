# j0126 one-command run on the MPCDF cajal cluster (GPFS + Slurm)

Commit `1546f47`, clean worktree, python 3.11, `pip install -e .`, zarr 3.1.6, torch 2.14.
Two runs of the whole volume (10664 x 10913 x 5700) from scratch -- no released checkpoint
anywhere in the path -- plus a 1008^3 crop to shake the chain out.

**Where to find the code.** Branch `j0126-minimal` is what I would send you: 804 insertions
over `1546f47`, of which **268 lines are actual code fixes**. The rest is a 168-line
`tutorials/neuron_j0126/run_everything.sh` (the one command), a 222-line byte-exact revert
of your own `e604c46` deletion of `reproduction/evaluate.py` (verify with
`git diff e604c46^ HEAD -- tutorials/neuron_j0126/reproduction/evaluate.py`, which is empty),
and 39 lines of site-specific `params.yaml` that should not be merged. Branch
`j0126-tutorial-fixes` additionally carries three defensive guards. Patch:
`bootstrap/j0126-minimal.patch`.

**Honest caveat before the numbers: the headline run was not produced by the code above.**
Run 2's decode ran through an out-of-tree driver (`abiss_fleet.sh`) that shards ABISS's
atomic layer across nodes and drives `run_layer.sh` directly, because the single shared-memory
form could not finish in any walltime this site allows. It is byte-identical to the stock
path on a crop, and at full volume it produced a complete decode with no failures -- but the
three guards in the diff were never exercised by the run that produced these numbers.

## Results

All full-volume, from-scratch affinity, scored against the 50 test skeletons.

| segmentation | NERL mt0 | NERL mt5 | VOI split | VOI merge | frag/skel | multi-owner nodes |
|---|---|---|---|---|---|---|
| FFN reference (your published row) | **0.5258** | **0.5380** | 1.729 | 0.127 | 123.9 | 0.021 |
| raw watershed (floor) | 0.0011 | 0.0011 | 10.231 | 0.108 | 2790.5 | 0.097 |
| ABISS decode, exclusion mask OFF | 0.0858 | 0.1530 | 2.206 | 0.811 | 52.6 | 0.557 |
| ABISS decode, exclusion mask ON | 0.1161 | **0.2613** | 1.659 | 0.248 | 47.4 | 0.612 |
| *target row: ABISS + exclusion mask* | *0.301* | *0.470* | | | | |

Read the first row first: scoring your published node LUT reproduces your reference row to
four decimals, VOI 1.856 included, so the metric, the skeletons and the LUT convention here
are all correct. Everything below rests on that.

Rows three and four are **the same affinity decoded twice**; the difference is the bug in the
next section, not a parameter.

I am 1.8x short of 0.470 and do not know why yet. What remains is splits, not merges -- and
oddly we are *less* fragmented than FFN (47.4 vs 123.9 per skeleton) with a *better* VOI
split (1.659 vs 1.729), yet 228 labels hold 61.2% of all skeleton nodes. One thing I cannot
explain and would rather flag than hide: the multi-owner node fraction **rose** (0.557 ->
0.612) while VOI merge fell 3.3x. Error correction is running on the masked segmentation;
after that my suspects are `AGG_THRESHOLD: 0.20` (your own comment says the thresholds were
chosen to under-agglomerate on principle, not fitted) and the checkpoint the driver picks by
mtime, which on my run was the final overfit step (val 1.32) rather than the best epoch
(val 0.74).

## The exclusion mask never reaches ABISS

This is the finding of the whole exercise. Three defects line up, and the result is a
silently degraded segmentation with no error anywhere.

**1. The masked read path is the one the tutorial does not take.**
`volume_backends.open_volume` wraps its h5, zarr and h5-chunkstore branches in
`_with_keep_mask` (:801, :807, :812). The precomputed fallback (:824) returns a bare
`CloudVolume`, which has no `attach_keep_mask`. So `AFF_KEEP_MASK` is found, loaded, and
dropped without a word. The tutorial copies the affinity h5 into a precomputed layer and
points `AFF_PATH` at the copy -- precisely the branch that discards the mask. Verified
directly: `_keep_mask_for(AFF_PATH)` returns a zarr array while `open_volume(AFF_PATH)`
returns `CloudVolumePrecomputed` with no `_mask` attribute.

**2. You cannot simply repoint `AFF_PATH`, because of a second defect.** The driver writes
`AFF_CHANNELS` as a channel *count* (`3`); `volume_backends._channels_for` reads it as an
*index list*, so `3` asks for channel index 3 of a 3-channel volume. The precomputed backend
ignores the field, so this is invisible there; the h5 backend -- the only one that masks --
raises `AFF_CHANNELS [3] out of range`. (Worth knowing: ABISS reads this key both ways;
`cut_chunk_common.cut_data` treats it as a count in a fallback branch that 1-, 3- and
4-channel affinity never reach. A contiguous `0..n-1` prefix is the only value that
satisfies both.)

**3. The mask itself was built to the wrong recipe.** `build_j0126_keep_mask.py` thresholded
channels 1/3/5 at a uniform **128**. FFN publishes the exact per-channel recipe in its bucket
README: blood vessel >= 252, myelin >= 252, out-of-bounds >= 25. Uniform 128 excluded
**17.53%** of the volume; the published recipe gives **14.61%** against the 15.64% you quote.

**What it costs.** 17.5% of j0126 is vessel, myelin and out-of-bounds, and without the mask
segments grow through it and chain neurons together -- exactly what `attach_keep_mask`'s own
docstring warns about. Measured, same affinity: VOI merge 0.811 -> 0.248, NERL mt5 0.153 ->
0.261.

**Fixes.** Point `AFF_PATH` at the affinity h5 (which also makes the ~28 h / 3.7 TB
precomputed copy unnecessary -- `load_data`'s docstring says that is what the h5 backend is
for), emit `AFF_CHANNELS` as an index list, and use the published thresholds. I also added a
guard that refuses to decode when a configured mask would not be applied, so this cannot
recur silently.

## A pattern, not four separate bugs

Four defects share one shape: **an artifact that announces work before the work happens, then
gets trusted as proof it happened.** Each produces a wrong answer rather than an error.

- **`info` is written before the first block of the affinity copy.** Any interrupted copy --
  walltime, preemption -- looks complete, and the next run decodes a volume whose tail is
  zeros. Not hypothetical: that copy is 41,580 blocks at ~1,500/h, i.e. ~28 h, longer than
  the step's own walltime. It reached 61% and stopped. (Superseded on `j0126-minimal`:
  pointing `AFF_PATH` at the h5 removes the copy entirely.)
- **`seg/info` is written before any decoding**, so an OOM-killed decode reports `[done]` and
  every later run skips it -- the pipeline then reports an ERL for a segmentation that was
  never produced.
- **Chunk descriptors are never regenerated.** `chunk_volume.py` writes
  `<workdir>/<mip>_<x>_<y>_<z>.json` only when absent, so a descriptor outlives the
  `CHUNK_SIZE` that made it. I changed `CHUNK_SIZE` twice and both changes were silently
  ignored; three jobs died identically, ~11 minutes apart, with nothing saying the configured
  value was not being used.
- **`download_precompute.py` writes resume sidecars next to the store.** Delete a partial
  store, rerun, and every shard resumes from a record of work whose output is gone: the array
  reports DONE over a half-empty volume. This cost me a 264 GB "complete" download with 3 of
  6 sampled blocks empty.

Two completion checks in `run_j0126.py` are the same family: step 3 called a layer finished
on **one** chunk file (ABISS writes 33,264), and step 2 read an index whose chunk paths are
relative to the run directory that produced them, so a *finished* 2.6 TB affinity reported
`0/726` and a resume would have relaunched two days of GPU inference.

## The decode's chunk size is unusable at full volume -- and cannot be changed

Two separate defects, and the second hid the first for three failed jobs.

**`CHUNK_SIZE` defaults to the full XY extent**, `[bbox_x, bbox_y, 80]`, and `3_abiss.yaml`
ships that default. On j0126 that is one chunk of 10664 x 10913 x 80 = 9.3 Gvoxel. ABISS
reads affinity as float32, so the chunk is 104 GiB, and `run_layer.sh` runs layer 0 at
`parallel -j ${ncpus}` -- 64 concurrent on this node. The decode was OOM-killed at 250 GB,
and raising the job to 900 GB did not save it.

**`chunk_volume.py` never regenerates a chunk descriptor that already exists.** So after the
first run wrote `run/0_0_0_0.json` with the full-XY bbox, every later run silently kept it:

```
run/0_0_0_0.json   bbox [0, 0, 0, 10664, 10913, 80]    <- written by run 1, never updated
param              CHUNK_SIZE [2048, 2048, 80]         <- what the config said
```

I changed `CHUNK_SIZE` twice and both changes were no-ops. Three jobs died the same way,
~11 minutes apart, and nothing in the output said the configured value was being ignored.
Clearing the workdir is the fix, and it is not discoverable from any error message. This is
the third instance in this pipeline of one pattern -- **state that outlives the input it was
derived from** (see the copy above, and `seg/info`) -- and it is the single most expensive
class of bug I hit.

For the record, the working numbers: `[2048, 2048, 80]` measures 7.8 GB peak per chunk
(`measure_ws_chunk.py` replays the real read path; 1024^2 gives 2.0 GB, so it is ~2x the
array and linear in voxels), i.e. ~500 GB across 64 concurrent tasks, and leaves `top_mip`
at 7 so the merge depth is unchanged.

## The decode is I/O bound, and the resource table does not say so

With all 64 slots occupied, the watershed used **5.3 of 64 cores -- 8%**. Per chunk: 4.1
minutes of CPU, ~49 minutes holding a slot. It spends ~92% of its time waiting on the
filesystem.

Two causes. The precomputed affinity is stored in 512x512x**64** chunks while ABISS processes
512-aligned XY by **80** in z, so every chunk straddles two storage chunks in z and its
neighbour re-reads them: a 2 GB payload costs ~3.1 GB of reads, ~8 TB total against a 3.7 TB
layer. And the concurrency is self-defeating -- a single stream of this same read benchmarked
at ~170 MB/s, while 64 concurrent streams together got ~65 MB/s. `parallel -j ${ncpus}` takes
every core whether or not the access pattern benefits; 8-16 would very likely finish sooner.

**Timing, measured.** Three documents give three different deployments for one figure:
`params.yaml:108` "3.75 h on 64 cores", `README:122` "the recorded 40-node run took 3.75 h"
(immediately after "one shared-memory job ... never a job array"), and RESOURCE.md's
layer-aware fleet of 16-CPU nodes, about 640 cores. I followed params.yaml, ran the
single-job form the driver implements on one 64-core node, and the watershed stage alone
took ~38 h. If the real provenance is the 40-node fleet, then 3.75 h x 10 is exactly my
number and only the comment is wrong -- but that comment is the one a reader follows. The
watershed stage alone:

| | |
|---|---|
| atomic pass, 2592 chunks | ~33 h at ~78 chunks/h |
| composite merge, mip 1 | 37-92 tasks/h of ~324 |
| composite mip 2+ | ~100 tasks at `-j 8`, then `-j 4` |

and watershed is one of four stages (`watershed`, `remap_watershed`, `agglomerate_mean_edge`,
`remap_agglomeration`), the two remap stages each writing a full-resolution 5.3 TB uint64
layer. The shipped `time: "24:00:00"` cannot finish any of this; I am running with 5 days.

**The one thing that works properly here:** the decode resumes at chunk granularity.
`run_wrapper.sh` skips a chunk whose `done/<task>.txt` flag exists, and `update_task_flag.py`
writes a flag *only* for DONE, so an interrupted chunk is redone rather than skipped. When my
24 h walltime expired at 72%, the resubmission skipped exactly 1871 chunks and carried on.
That design is right, and it is what made the rest of this survivable.

## Four bugs that block the single command on a fresh machine

1. **`build_j0126_keep_mask.py --stage tissue --init` dies on zarr 3.** Line 202 passes the
   zarr-2 `compressor=Blosc(...)` kwarg; zarr 3.1.6 raises `ValueError: compressor cannot be
   used for arrays with zarr_format 3`. This aborts the driver before it submits anything.
   Trying `zarr.create_array(...)` first and falling back gets past it.

2. **`--wrap=python ...` assumes the compute node inherits the submitter's PATH.** Step 2
   fails in 3 s with `ModuleNotFoundError: No module named 'omegaconf'`. Steps 0b-0d happen
   to survive it on other nodes, so the run looks healthy until the first GPU step. Using
   `sys.executable` in the driver fixes every step at once. (This site's login shell also
   presets `SBATCH_EXPORT=SRUN_EXPORT_ENV=ALL`, which is why the usual `--export=ALL` default
   does not save it.)

3. **Step 2's output lands next to the checkpoint, where step 3 never looks.** For `--mode
   test --checkpoint X`, `runtime/checkpoint_dispatch.py` *overwrites* `inference.save_path`
   with a directory derived from the checkpoint's own location. Step 3 then dies with `No
   *.h5.index.json under <output_root>/affinity` -- after 22 minutes of GPU. Putting the
   checkpoint at `<infer save_path>/downloaded/checkpoints/` (the layout a training run
   produces, and the one `check_checkpoint` already globs for) and pointing `--discover` at
   `output_root` fixes both arms. Related: a single-chunk ROI stitches straight to one `.h5`
   with no `.index.json`, so `--vds --discover` has nothing to build a view over; that file
   already is the whole volume, and ABISS reads a plain h5 anyway.

4. **`KEEP_CHUNKS = (126, 252, 252)` is 87,032 files for the keep mask**, plus 10,890 for the
   tissue mask. This filesystem enforces an **inode** quota, not a byte quota, and had ~40,000
   files of headroom. Two probes settle it: 39,884 *empty* files (5 MB total) hit `Errno 122
   Disk quota exceeded`, while 254 GB in 8 files did not. A mostly-uniform binary mask
   compresses to nothing, so small chunks buy little locality. But the replacement must
   still **tile `CELL = 1008` exactly**: the keep stage writes CELL-sized cubes round-robin
   across concurrent array tasks, and a zarr write covering part of a chunk is a
   read-modify-write, so two shards touching one chunk lose a write -- and since
   `fill_value` is 1 ("keep"), the loss silently *unmasks* tissue. I learned this the hard
   way by shipping `(80, 512, 512)`, which does not tile 1008 (12.6, 1.97, 1.97).
   `(126, 504, 504)` does (8, 2, 2) and is 22,264 files; an assert now enforces the
   invariant rather than a comment asserting it. `download_precompute.py` has the same shape
   of inode problem -- its default `64 x 256 x 256` is 162,540 files for the EM volume -- but
   it is also the *fastest* shape: measured per 48x96x96 sliding window, `64 x 256 x 256`
   costs 13.4 ms, `128 x 512 x 512` 49.7 ms and the inode-cheap `64 x 2048 x 2048` 384 ms.
   So the chunk default is right and the inode budget is what breaks it; that belongs in
   RESOURCE.md next to the terabytes.

## Three more that stop step 3 or step 4, each an hour or more in

5. **`NUC_PATH: ""` is not the same as absent.** `abiss_chunk.py:_write_param` writes empty
   `NUC_*` keys, and ABISS tests `if "NUC_PATH" in global_param` -- key presence, not
   truthiness -- so it tries to open `""` and dies with `UnsupportedProtocolError` five
   minutes in. Dropping empty keys at write time fixes it.

6. **`4_error_correction.yaml`'s `size_glob` is too broad.** It matches ABISS scratch files
   that are not size tables, and the run dies with `Size of available data is not a multiple
   of the data-type size`. `.../abiss/scratch/**/seg_size_*.data` is the intended set.

7. **Zero-byte size tables crash the reader.** `decoding/error_correction/sizes.py` memmaps
   every match; ABISS legitimately writes empty ones for empty chunks, and `np.memmap` raises
   `cannot mmap an empty file`. Skipping zero-length paths (and returning an empty array when
   all are empty) is enough.

## Error correction runs serially as shipped, and cannot finish

`run_j0126.py` invokes step 4 as `--stage all --num-tasks 1`. `workflow.py` defines that as
the serial path -- it rejects any other task count -- and on the full volume it is ~131 h for
10,626 chunks at the measured ~70 s each, against a 24 h walltime. Meanwhile
`4_error_correction.yaml` already sets `task_count: 80`, RESOURCE.md already budgets "80
array tasks, 8 CPU workers/task", and `workflow.py` already declares
`ARRAY_STAGES = {skeletonize, contacts, postprocess}`. Nothing was missing except a driver
willing to use them. The driver now emits the twelve stages in order, those three as 80-way
arrays and the nine reductions on the configured block.

## The decode's atomic layer wants to be a fleet

The README says ABISS is "one shared-memory job ... never a job array: independent copies
race on the same hierarchy layers". That is right about the composite levels and wrong about
layer 0, which is one independent task per chunk and already has per-chunk done-flags. What
actually makes a naive array unsafe is that `update_task_flag.py` records only DONE, never
IN_PROGRESS, so two workers can pick the same chunk and write the same files. Sharding the
chunk list removes that by construction: give each array task a disjoint slice of `0.txt`,
then write layer 0's batch flag and hand the composite levels back to stock `run_batch.sh`.

Validated byte-identical against the stock path on a 1008^3 crop -- all 128 output chunks,
zero differing bytes -- and then at full volume. Run 1's single-node atomic watershed took
~33 h; run 2's sharded one took ~18 h while only ever getting 6 of its 40 slots. At full
width it is a couple of hours. Three portability defects surfaced only once it ran
distributed, and none can be hit by the single-job form: `init.sh` creates `config.sh` with
`>` on first use, so concurrent cold starts read it half-written; `AIRFLOW_TMP_DIR` defaults
to a `/tmp/airflow` that nothing creates (the pytc driver quietly compensates, so ABISS is
unrunnable by any other launcher); and the CPU locks are named by CPU id in one shared
directory, so two nodes both seeing cpus 0-15 contend.

## Undeclared dependencies

`tinybrain`, `chunk_iterator` (imported as `chunkiterator`, not on PyPI, referenced only in
`lib/abiss/docker/Dockerfile:53`) and GNU `parallel`. None is in `setup.py` or the README, and
none fails legibly: the last one surfaces as `run_layer.sh: cannot cat 0.txt`, because the
wrapper's `try`/`die` swallows `parallel: command not found`. Installing the C++ toolchain
into the same conda env as the python side corrupted numpy once, so ABISS's build deps want
their own env.

## Smaller things, in descending order of how much time they cost me

- **`em_bbox` never reaches the mask, ABISS or error-correction geometry** (`run_j0126.py`
  passes `--bbox` only to the EM download). So the crop smoke test is not "a few minutes":
  the crop downloads in 40 s and then the tissue mask builds over the whole volume, ~2 h on 16
  shards. Later steps then fail with `BBOX out of range` and `found 16/10626 chunk files`.
  Resolving the crop into the generated configs is what finally made a 40-minute end-to-end
  test possible, and that test is the thing I would most want shipped with the tutorial.
- **The EM chunk default is right; the inode quota is what breaks it.** Measured on one
  store, same window, same data (`bench_em_chunks.py`):

  | chunks | per 48x96x96 window | files for the volume |
  |---|---|---|
  | `64 x 256 x 256` (shipped default) | **13.4 ms** | 162,540 |
  | `64 x 2048 x 2048` (my inode workaround) | 384.2 ms | 3,240 |
  | `128 x 512 x 512` (compromise) | 49.7 ms | 20,790 |

  An earlier draft of this report claimed a "6x speedup from rechunking". That was measured
  against my own workaround rather than the shipped default, and it is the wrong comparison:
  `download_precompute.py`'s default is the fastest of the three, and I had moved away from
  it only to fit ~40,000 inodes. The defect is not the chunk shape. It is that nothing warns
  you the inode-cheap choice costs 29x on every sliding window of a 220 GPU-hour step.
  `128 x 512 x 512` is the compromise when inodes are tight. Raising `sw_batch_size` 12->32
  was worth about a fifth of the recovery; the rest was chunk shape.
- **`inference.time: "8:00:00"` is too short for the full volume**: 14 of 40 shards hit
  TIMEOUT. 24 h is comfortable.
- **The driver picks the checkpoint by `ls -t | head -1`,** i.e. the newest file, which is the
  *final* step, not the best. My run overfit visibly (train 1.016 -> 0.487 while val 0.775 ->
  1.32), so the inference ran on `step=00200000.ckpt` while `epoch=001-val_loss_total=0.7406`
  sat next to it. Whatever the intended policy, it should be explicit rather than a
  side effect of mtime ordering.
- **`3_abiss.yaml` sets `resolution_xyz: [10, 10, 10]`** while the README says 9x9x20 nm
  throughout. Evaluation fails with `ScaleUnavailableError: Scale <9,9,20> not found`.
- **`download_precompute.py` writes resume sidecars *next to* the store** as
  `<out>.progress.<shard>`. Delete a partial store, rerun, and every shard resumes from a
  record of work whose output is gone: the array reports DONE over a half-empty volume. This
  cost me a 264 GB "complete" download with 3 of 6 sampled blocks empty. Putting the sidecar
  inside the store makes the record die with the data. (Same family as finding 11 above:
  progress state that outlives the thing it describes.)
- **`--stage keep` silently over-masks when the EM is a crop**: off-volume reads pad with 0,
  which `_border_pad_slice` treats as padding, so against the full grid a 1008^3 EM masks
  essentially the whole outer ring. No error.
- **`inference.num_shards: 80` submits 79 jobs with nothing to do** for a one-chunk crop.
- **There was no evaluation step at all**, so the pipeline could not produce the number it is
  measured by. `tutorials/neuron_j0126/reproduction/evaluate.py`, deleted in the restructure,
  still works: it reproduces the published FFN reference row exactly -- NERL mt0 0.5258 vs
  0.526, mt5 0.5380 vs 0.538, VOI 1.856 vs 1.856 -- against `ffn_node_lut_test_50.h5`. I
  restored it as step 5. That it reproduces the reference row exactly is also the evidence
  that the metric, the skeletons and the LUT convention here are all correct.

## One more, trivial but total

`.gitignore`'s blanket `*.sh` means `tutorials/neuron_j0126/run_everything.sh` -- the
tutorial's own entry point -- could not have been committed by anyone. Negating it for
tutorial entry points is a one-line change.

## On params.yaml

The per-step resource blocks are the right shape and `--check` is genuinely useful -- it is
how I found most of the above. Two suggestions. `slurm_partition: gpu` is a plausible-looking
default that every site must change, so a value that fails loudly (`CHANGE_ME`) would surface
at submit time rather than at first run. And the resource table in the README should say
inodes as well as terabytes: this run needs ~110,000 files, which is the quota that actually
bites first on a shared filesystem.

## Reproducing this

`tutorials/neuron_j0126/run_everything.sh` is committed in the repository, clones nothing
when run from inside a checkout, and patches nothing -- the fixes are in the code:

    bash tutorials/neuron_j0126/run_everything.sh --root <dir> --partition <name>

Downloaded on its own it clones instead, so from nothing:

    curl -fsSL https://raw.githubusercontent.com/PytorchConnectomics/pytorch_connectomics/\
    master/tutorials/neuron_j0126/run_everything.sh | bash -s -- --root /scratch/j0126 \
    --partition gpu

with `J0126_REPO_URL` / `J0126_REPO_REF` to override the clone -- necessary until these
commits are merged, since that raw URL serves upstream `master`, which does not carry them.
It builds the environment, builds ABISS, fetches the EM volume, the labelled cubes, the test
skeletons and the FFN node LUT, then trains, infers, decodes, corrects and scores.

`--bbox "2900 3908 5000 6008 5000 6008"` exercises every step on a 1008^3 crop in ~40 min
instead of days. **That crop path is the single thing I would most want shipped with the
tutorial** -- it found the `config.sh` race, the `/tmp/airflow` default, the CPU-lock
collision, and a regression my own mask fix introduced, each in minutes rather than days.

Scope of what I actually smoke-tested, so you can discount accordingly: from a fresh clone it
built the environment, cloned, installed, built ABISS, wrote `params.yaml` and submitted the
whole pipeline including the twelve-stage error-correction chain. It ran with `--no-train`,
so step 1 is not covered by it.

## Error correction cannot run without the nucleus volume

This is the newest finding and it changes the shape of the problem. `morphology.py:912`
calls `load_nucleus_firewall(nucleus_targets_path)` unconditionally inside `classify`, and
`load_nucleus_firewall` raises when the manifest has no nucleus histograms or qualified
segment labels:

    ValueError: .../nucleus_competition/manifest.json: missing nucleus histograms or
    qualified segment labels

There is no flag to disable it -- `nucleus_targets_path` is a required argument threaded
through the stage. So step 4 does not degrade gracefully without nuclei; it stops. The
twelve-stage chain dies at `skeletons`, the second reduction, after `skeletonize` has
processed all 10,626 chunks.

That means the rows above *ABISS + exclusion mask* are not merely unreachable for scoring
purposes: **the shipped pipeline cannot complete on the data it ships with.** Any site
running this tutorial gets through the decode and then hits this.

## Two things only you can settle

**The nucleus volume.** It is not published: not in the FFN bucket (seven layers, one
top-level prefix, README enumerates six data types, and the only occurrence of "nucleus",
"soma" or "cell body" anywhere is `2: cell body`, a probability channel), and not under
`pytc` on HuggingFace (`zebrafinch-j0126` has skeletons, LUT, training cubes and the FFN
agglomeration; `pytc/vEM` has the affinity checkpoint and nothing else). The paper describes
it as hand-edited, which would explain the absence. Without it `3_abiss.yaml`'s nucleus
competition stays off *and* error correction cannot run at all. If it could go up next to
the checkpoint, both problems close at once.

**Were 0.470 and 0.539 computed in native voxels or in nm?** My evaluator uses native
voxels, and so does the reference LUT scoring, so the two are internally consistent -- but
on a 9x9x20 nm grid the choice reweights skeletons by orientation, and it changes the
comparison if yours differed.

## What produced these numbers, and what the patch is

Worth separating, because they are not the same thing.

The patch (`j0126`, one commit, 556 insertions over upstream `master`) is the minimal set
that makes the tutorial run end to end and compute its own number. It does not include the
out-of-tree machinery this particular run needed: a driver that shards ABISS's atomic layer
across nodes, because the single shared-memory form the README prescribes could not finish
in any walltime this site allows. That sharding is byte-identical to the stock path on a
crop (128 output chunks, zero differing bytes) and produced a clean full-volume decode, but
it is not in the diff, and the numbers above came from it.

So: the fixes are reviewable on their own, and the decode timings are honest measurements
from a deployment the shipped driver does not implement. If you want the sharded driver too
it is a separate ~150 lines and I can send it.

