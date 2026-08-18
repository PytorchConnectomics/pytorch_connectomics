# Zebrafinch whole-volume job manual

This is the operational checklist for whole-volume ABISS decoding. Read it before
preparing, launching, resuming, or modifying a production run. Experimental findings
belong in `lesson*.md`; repeatable job procedures belong here.

## Non-negotiable rule

**Run whole-volume decoding through the sharded Slurm launcher.** Do not run
`scripts/run_seuron_provenance.py --execute`, `run_batch.sh`, or a long GNU Parallel
command directly from Codex, an interactive shell, `nohup`, or tmux. Those are suitable
only for bounded smoke tests.

**Do not launch heavy jobs on a login machine.** Login nodes are only for preparation,
submission, lightweight metadata checks, and short diagnostics. Decoding, inference,
large-volume reads or scans, evaluation over full datasets, multiprocessing, and
high-memory or long-running work must be submitted to Slurm. If there is any doubt about
whether a command is heavy, treat it as heavy and use Slurm.

`backend: abiss_local` in a replay YAML selects the ABISS execution backend. It does not
mean that a whole-volume job should be hosted by the login/current node. The production
path still uses Slurm:

```text
wholevol_prepare.py
  -> submit_wholevol_sharded.sh
     -> sbatch_abiss_shard.sh arrays
     -> sbatch_nucleus_competition.sh, when NUC_PATH is enabled
```

Why: an interactive or Codex-managed process dies with its hosting session. Slurm owns
the job independently, records its exit state, schedules many nodes, and preserves the
dependency chain.

## Production pipeline

For a nucleus-aware run, the submitted dependency chain is:

```text
watershed L0..L5
  -> watershed remap
  -> competitive nucleus growth
  -> constrained mean-edge agglomeration L0..L5
  -> final agglomeration remap
```

The nucleus stage is submitted only when the prepared parameter JSON has `NUC_PATH` and
competition is enabled. Agglomeration then consumes its manifest and enforces the hard
nucleus cannot-link.

Chunks within one layer are independent and are distributed across a Slurm array. Layers
remain sequential because a composite chunk depends on children from the preceding layer.
The launcher has no array throttle: it uses as much concurrency as Slurm grants.

## Native arm0_96 affinity prerequisite

`arm0_96` means the checkpoint inferred with sliding window `[48,96,96]`, and its chunk
store must end in `_win48x96x96.h5.chunks`. The same checkpoint's default-suffix store is
the `[144,144,144]` inference and must be called `arm0_win144`. Never infer identity from
the checkpoint or experiment nickname alone; record the exact affinity path.

The canonical full-volume native-window launcher is:

```bash
sbatch dev/zebrafinch/sbatch_arm0_native96_affinity.sh
```

It runs one GPU Slurm array element per ROI chunk, requests 96 GiB host memory, and safely
skips existing chunks. Before allowing ABISS preparation, require all of the following:

- the generated index names the `_win48x96x96` store and contains exactly 726 chunks;
- every indexed HDF5 exists with CZYX shape `(3, ...)` and dtype `float16`;
- a live shard log names the `_win48x96x96` output path, not the default suffix;
- the native nucleus match-guard reports zero fused source pairs and separates sources
  611 and 651.

`sbatch_finalize_arm0_native96.sh` enforces these checks, prepares a fresh replay, submits
the sharded ABISS chain, and attaches `wholevol_nerl.py --merge-threshold 50`. Submit it
with `afterok` dependencies on the affinity array and any still-live nucleus gate jobs. If a
completed one-task gate has already left Slurm's live controller, depending only on the
affinity array is acceptable because the finalizer recomputes the ABISS runtime hash and
rejects stale or biologically failed gate reports itself. Do not bypass those checks by
preparing the decoder early.

## 1. Choose an isolated run namespace

Run from the repository root and use a new output namespace for every independent decode:

```bash
REPO=/projects/weilab/weidf/lib/pytorch_connectomics
PYTC_PYTHON=/projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python
CFG="$REPO/dev/zebrafinch/wholevol_arm096_nuc_competitive.yaml"
WV="$REPO/dev/zebrafinch/wholevol_arm096_nuc_competitive"
RUN_NAME=seg_arm096_nuc_competitive

cd "$REPO"
export HDF5_USE_FILE_LOCKING=FALSE
```

`WV` is the launcher directory. The actual replay root is `$WV/$RUN_NAME`. Do not point a
new experiment at a scored or otherwise verified run.

Use cached affinity inputs normally. Do not symlink another run's watershed or scratch
tree unless reuse is explicitly intended and the manifests/build/configuration are proven
identical. Resuming the same run via its DONE flags is the safe form of reuse.

## 2. Always run the preparer before `sbatch`

For a new run:

```bash
"$PYTC_PYTHON" dev/zebrafinch/wholevol_prepare.py \
  --config "$CFG" \
  --name "$RUN_NAME" \
  --out-root "$WV" \
  --runtime-json "$WV/runtime.json" \
  --mode fresh
```

For the same interrupted run, use `--mode resume`. Resume checks the existing manifest
against the requested provenance, parameter payload, ABISS build, and bounding box.

Do not casually use `--mode overwrite`: it deletes the existing replay root. Prefer a new
name. The preparer performs ABISS and affinity preflights, prepares output layers and
runtime secrets, writes the canonical parameter JSON, and generates all layer task lists.
It must finish successfully before submission.

The preparer must import `scripts/run_seuron_provenance.py` from this checkout. It must
never point at a personal or stale worktree.

## 3. Inspect prepared state

Run this check before submitting:

```bash
RUNTIME_JSON="$WV/runtime.json" "$PYTC_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

runtime_path = Path(os.environ["RUNTIME_JSON"])
runtime = json.loads(runtime_path.read_text())
for key in ("workdir", "abiss_home", "secrets", "param", "top_mip", "layers", "seg_path"):
    assert runtime.get(key) not in (None, ""), f"missing runtime key: {key}"

workdir = Path(runtime["workdir"])
secrets = Path(runtime["secrets"])
param_path = Path(runtime["param"])
assert (secrets / "config.sh").stat().st_size > 0
assert param_path.stat().st_size > 0

layers = {int(k): int(v) for k, v in runtime["layers"].items()}
assert set(layers) == set(range(int(runtime["top_mip"]) + 1))
assert all(count > 0 for count in layers.values())
for layer, expected in layers.items():
    actual = sum(bool(line.strip()) for line in (workdir / f"{layer}.txt").read_text().splitlines())
    assert actual == expected, (layer, actual, expected)

param = json.loads(param_path.read_text())
for key in ("AFF_PATH", "WS_PATH", "SEG_PATH", "AGG_THRESHOLD"):
    assert param.get(key) not in (None, ""), f"missing param: {key}"
if param.get("NUC_PATH"):
    assert param.get("NUC_COMPETITION_ENABLED", True)
    assert param.get("NUC_COMPETITION_MANIFEST")

print("prepared OK")
print("layers:", layers)
print("affinity:", param["AFF_PATH"])
print("keep mask:", param.get("AFF_KEEP_MASK"))
print("nuclei:", param.get("NUC_PATH"))
print("threshold:", param["AGG_THRESHOLD"])
print("output:", runtime["seg_path"])
PY
```

Also check available project storage with `df -h`; completed ABISS scratch can occupy about
670 GB. Do not use a recursive `du` as a routine check on these trees because hundreds of
thousands of files make it very slow.

Before submission, confirm that:

- the affinity, keep mask, nucleus mask, threshold, and output paths are the intended ones;
- the layer counts are plausible (the current full volume is
  `10626 -> 1452 -> 216 -> 27 -> 8 -> 1`);
- no old job chain is still writing to the same `WV`;
- nobody will rebuild or replace the selected ABISS binaries during the run.

## 4. Submit the sharded Slurm chain

```bash
mkdir -p "$REPO/slurm_outputs"
set -o pipefail
WV="$WV" bash dev/zebrafinch/submit_wholevol_sharded.sh \
  | tee "$WV/slurm_submission.txt"
```

The submission transcript is the authoritative list of stage and array job IDs.
`$WV/final_jobid.txt` contains the final remap job. Keep both files with the run.

Do not substitute a local multiprocessing command. `sbatch_abiss_shard.sh` already uses
GNU Parallel inside each allocated node and `submit_wholevol_sharded.sh` spreads shards
across nodes. For layer 0, the current configuration can reach `80 x 19 = 1520` chunk
processes when the scheduler has capacity; actual concurrency depends on priority and free
nodes.

## 5. Monitor health, not just existence

Read the first and final IDs from the submission record:

```bash
FIRST_JOB=$(awk '$1 == "ws_L0" {for (i=1; i<=NF; i++) if ($i == "->") {print $(i+1); exit}}' \
  "$WV/slurm_submission.txt")
FINAL_JOB=$(cat "$WV/final_jobid.txt")

squeue -j "$FIRST_JOB,$FINAL_JOB" -o '%.18i %.18j %.10T %.24R %.6C'
sacct -j "$FIRST_JOB" -X -o JobIDRaw,JobName,State,ExitCode,Elapsed,AllocCPUS,MaxRSS
```

For an active watershed layer, inspect exact logs and progress:

```bash
rg -n -i 'traceback|exception|error|failed|killed|oom|no space|abort' \
  "$REPO"/slurm_outputs/abshard_ws_L0_${FIRST_JOB}_*.out

find "$WV/$RUN_NAME/scratch/$RUN_NAME/ws/remap" \
  -maxdepth 1 -type f -name 'done_*.data' | wc -l
```

Healthy startup means array elements are `RUNNING`, each log begins with a shard summary,
old completed chunks print `skip`, and the DONE count eventually increases. A base array
line may remain `PENDING (Priority)` while some of its elements run; that is normal.
Downstream jobs should be `PENDING (Dependency)` until their predecessor completes.

Check live memory with `sstat`, not only the requested memory shown by `squeue`. The
2026-08-14 layer-0 jobs reached 167,749,128 KiB, effectively the full 160 GiB allocation,
so future layer-0 submissions request 180 GiB. Do not lower that allocation merely to start
more shards; preserve memory headroom and let Slurm determine placement.

## 6. Diagnose a stop before resuming

Use Slurm accounting first:

```bash
sacct -j "$FIRST_JOB" -X \
  -o JobIDRaw,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS
```

Then inspect the affected stage log. Common states have different fixes:

- `OUT_OF_MEMORY`: lower per-node process concurrency or request more memory.
- `TIMEOUT`: extend the time limit or reshard the stage.
- `FAILED` with an ABISS traceback: fix the input/build/runtime problem first.
- `CANCELLED` or `DependencyNeverSatisfied`: find the failed upstream job.
- no Slurm record and all local processes vanished: it was not a persistent Slurm launch;
  session cleanup is the likely cause.

Never rebuild `lib/abiss/build/` while a decode uses it. Besides transient missing binaries,
this can mix DONE chunks from different builds and void reproducibility.

The ABISS manifest identity must cover the runtime scripts and compiled executables, not
only the repository commit. Local ABISS development commonly leaves edited scripts and
untracked `build/` binaries under an unchanged Git HEAD. The preparer hashes both runtime
surfaces; a changed reader or rebuilt executable therefore makes `--mode resume` refuse the
old run.

## 7. Resume safely

First ensure no old array element is still writing to the run. Then re-run the preparer with
`--mode resume` so manifest compatibility is checked.

The safest restart is to resubmit the whole chain:

```bash
set -o pipefail
WV="$WV" bash dev/zebrafinch/submit_wholevol_sharded.sh \
  | tee "$WV/slurm_submission_resume_$(date +%Y%m%d_%H%M%S).txt"
```

`run_wrapper.sh` skips every valid DONE chunk, so completed work is retained. Use
`START_AT=<stage>` only when every earlier stage is known complete, for example:

```bash
START_AT=me_L2 WV="$WV" bash dev/zebrafinch/submit_wholevol_sharded.sh
```

Valid stage names follow the submission output: `ws_L0` through `ws_L5`, `remapws`,
`nuccomp`, `me_L0` through `me_L5`, and `remapagg`. Do not skip `nuccomp` for a
nucleus-aware decode.

## 8. Completion checks

A run is complete only when the final job reports `COMPLETED`, not merely when it disappears
from `squeue`:

```bash
FINAL_JOB=$(cat "$WV/final_jobid.txt")
sacct -j "$FINAL_JOB" -X -o JobIDRaw,JobName,State,ExitCode,Elapsed,MaxRSS
```

For nucleus-aware runs, require a nonempty competition manifest. Require the final
precomputed segmentation metadata and then run the canonical NERL evaluation. Preserve:

- `precomputed/`;
- `param` and runtime secrets parameter JSON;
- replay `manifest.json` and `runtime.json`;
- `slurm_submission*.txt` and `final_jobid.txt`;
- the nucleus competition manifest;
- the canonical evaluation JSON.

After the output and score are verified, `scratch/` and `work/` are resume intermediates and
may be removed only as a separate, explicitly reviewed cleanup operation.

## Evaluation standard

Every whole-volume number quoted anywhere — lessons, papers, comparisons — is produced this
way. A NERL without its convention and its merge threshold is not a result.

```bash
python dev/zebrafinch/wholevol_nerl.py \
  --seg "$WV/seg_<name>/precomputed/seg/seg_<name>" \
  --merge-threshold 50 --out "$WV/nerl_funlib_mt50.json"
python dev/zebrafinch/wholevol_nerl.py \
  --seg "$WV/seg_<name>/precomputed/seg/seg_<name>" \
  --merge-threshold 0  --out "$WV/nerl_funlib_mt0.json"
```

Or, as one Slurm cell per threshold, `dev/zebrafinch/sbatch_score_eval_matrix.sh` with
`SEG`, `OUT`, `MT`.

Both JSONs also carry the NISB-challenge skeleton VOI (`voi_split` / `voi_merge` /
`voi_sum`), which is threshold-free — see "Skeleton VOI" below.

**Convention: funlib-matched, voxel units.** That is `wholevol_nerl.py`'s DEFAULT
(`--canonical`): the graph comes from `skel_to_erlgraph(skels)` with **no**
`skeleton_resolution`, so edge lengths are in voxels. This is the convention the external
FFN reference is in. The alternative `--nm` mode builds the graph with
`skeleton_resolution=(20,9,9)`; because NERL is length-weighted it reweights z- against
xy-running neurites and reads **~0.03 lower**. `--nm` exists only to reproduce the crop
comparator and is never comparable to the FFN reference or to any number in the lessons.

**Report BOTH `mt=50` and `mt=0`, always.** The merge threshold is how many nodes a
false merge may contain before it is counted. ABISS flood-fills every voxel and so creates
incidental border contacts, which `mt=50` forgives and `mt=0` does not; `mt=50 >= mt=0` for
the same segmentation, and the two can move differently. Quoting one alone invites exactly
the comparison error recorded below.

**Fixed inputs.** Skeletons `/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`
(50 skeletons, 500,845 nodes); bbox `[0, 0, 0, 10664, 10912, 5700]`. Do not vary these when
producing a comparable number.

**External reference: FFN, funlib, voxel units — `mt50 = 0.538003`, `mt0 = 0.525766`.**
Authoritative copy in `reports/ffn_reference.json` (which also carries the oracle ceilings,
`oracle_mt50 = 0.925546` and `oracle_mt0 = 0.813775`); see `lessons.md` L122 and the table at
`lessons.md:4952`. Compare each of our thresholds against the matching FFN threshold.

The claim that the public FFN score is an `mt0` number is **wrong** and was itself a corrected
bug — 0.538003 reproduces bit-exactly at mt5 and mt50 and never at mt0, where the same LUT
gives 0.525766. Some older notes still carry the mt0 phrasing; L122 supersedes them.

**Comparisons must be matched.** Compare only numbers sharing convention, merge threshold,
skeleton set and bbox. An A/B between two decodes is valid only when both arms were scored
by the same invocation, and a run whose `param` `AFF_PATH` differs from its comparator is a
different experiment regardless of what the run directory is named.

### Skeleton VOI — the NISB challenge metric

Quote it next to NERL. `wholevol_nerl.py` now emits `voi_split` / `voi_merge` / `voi_sum` on
every run, and `dev/zebrafinch/score_skeleton_voi.py` recomputes both reference arms from
their cached node LUTs (seconds, no whole-volume read) with a guard that re-derives each
arm's published NERL first — a wrong LUT or a wrong node order (L122) aborts instead of
quoting a plausible VOI.

**Definition, matched to the challenge.** `lib/banis/metrics.py:30` calls
`funlib.evaluate.rand_voi(gt_ids, pred_ids)` over the GT skeleton nodes: GT is the node's
neuron, pred is the seg label under the node, `voi_split` is over-segmentation and
`voi_merge` under-segmentation, in bits. Lower is better.
`connectomics.metrics.nerl.skeleton_voi` is a numpy port validated against that kernel and
re-checked against it on every scoring run.

**It is threshold-free.** VOI counts nodes — no edge lengths, no merge tolerance — so a
segmentation has exactly ONE VOI. It does not take an `mt=50` / `mt=0` pair, and it is immune
to the voxel-vs-nm convention that moves NERL by 0.03.

**GT labels are `node_skeleton_index + 1`, not the skeleton id.** funlib drops nodes whose GT
label is 0, and `test_50` contains a neuron literally named `"0"` (5,481 nodes) that a naive
`rand_voi(id, pred)` call deletes in silence. VOI is invariant to GT relabelling, so `+1`
costs nothing and scores all 50 neurons; the naive call reads `voi_sum` 1.854919 (FFN) /
2.547715 (ours) instead, moving the gap by 0.013.

**Measured 2026-08-18** (`reports/skeleton_voi.json`; same skeletons and bbox as above):

| arm | voi_split | voi_merge | **voi_sum** | NERL mt50 | NERL mt0 | node coverage |
|---|---:|---:|---:|---:|---:|---:|
| FFN (`em_erl/results/j0126/node_lut.h5`) | 1.728622 | 0.127173 | **1.855795** | 0.538003 | 0.525766 | 97.61% |
| ours, `wholevol_arm0_native96_nuc_matchguard` (ABISS + nucleus instance mask) | 2.542840 | 0.019130 | **2.561970** | 0.481614 | 0.287184 | 99.67% |
| **Δ ours − FFN** | **+0.814218** | −0.108043 | **+0.706175** | −0.056389 | −0.238582 | +2.06 pp |

**How to read it — the merge column is a coverage artifact, do not claim it.** funlib keeps
predicted background as a real label, so every unassigned node pools into one "segment"
spanning all 50 neurons and lands in `voi_merge`. FFN leaves 2.4% of nodes on background
against ABISS's 0.3%, and that difference *is* the entire merge column: dropping
background-predicted nodes collapses `voi_merge` to **0.000047** (FFN) and **0.001573**
(ours), i.e. ours becomes marginally worse, and the split gap widens to **+0.911132**. So the
honest statement is that at node level neither arm carries mass-weighted merge error and the
whole VOI gap is splits.

This is also why VOI does not replace the both-thresholds NERL rule: VOI is node-mass
weighted, so the sub-50-node phantom merges that cost us 0.194 NERL between mt=50 and mt=0
are nearly free in `voi_merge`. VOI reads fragmentation; strict-threshold ERL reads
contamination. Report both.

## GT hygiene — no ground truth inside a non-oracle algorithm

A decode number is only a claim about generalization if no test-set information reached the
thing being scored. "Oracle", "ceiling" and "headroom" figures are exempt **and must be
labelled as such**; everything that produces or selects a segmentation is not.

### The three tiers

| tier | contents | may be read by |
|---|---|---|
| **T0 test** | `test_50_skeletons.h5`; every per-chunk crop of it under `chunk_gt_skel/` | evaluation only, scored **once** at the end |
| **T1 dev** | `valid_12`; FFN pseudo-GT (`data/ffn_pseudogt`) | model/parameter selection |
| **T2 free** | affinity, nucleus channel, keep mask, segment sizes, RAG, prediction skeletons | anything, including the shipped algorithm |

Anything a production run reads must be T2. Anything that *chooses* a production
parameter must be T1 or T2. Note `valid_12` shares 2 identical neurons with `test_50`
(28≡103, 49≡100), so T1 is not perfectly clean either — say so when it matters.

The FFN tissue keep-mask (`tissue_border_keep_mask_full.zarr`) is T2 by this rule — it is
another method's output, not GT — but it is an **external dependency worth +0.089 NERL**
(see `error_analysis_arm0_96.md` §7). Declare it whenever independence is claimed.

### How to audit (the checks that actually catch it)

1. **Never compare GT sets by key name.** `chunk_gt_skel/*.h5` uses `<skel_id>_<cc_index>`
   (`'19_0'`), `test_50_skeletons.h5` uses `'19'`. A string comparison reports "no overlap"
   on files that are the same neurons. Compare **vertices**: crops are chunk-local, so add
   `chunk_index * 1008` and test set-membership against the global skeleton.
2. Grep the launch path (`wholevol_prepare.py`, `submit_wholevol_sharded.sh`,
   `sbatch_*.sh`, `lib/abiss/`) for skeleton/GT reads. Scoring attached *after* the final
   remap is fine; a read *before* it is not.
3. For any tuner, find the line that returns the objective and trace which GT it reads.
4. Check that a rule justified in `lessons.md` by a named GT skeleton did not ship. Those
   sentences ("caliber ratio 0.16 → vetoed, saves skel28 0.655") are analysis, and are only
   a leak if the constant reached the pipeline.

### Audit of 2026-08-17

**Clean — verified T2:**

- Launch path (`wholevol_prepare.py`, `submit_wholevol_sharded.sh`, `sbatch_abiss_shard.sh`,
  `sbatch_nucleus_competition.sh`) reads no skeletons. `sbatch_finalize_arm0_native96.sh`
  touches GT only by attaching `wholevol_nerl.py` **after** the final remap.
- `lib/abiss/scripts/nucleus_competition.py` reads no GT.
- The **611/651 match-guard is GT-free**: those are source/CC3D **nucleus** IDs, and the gate
  asserts two detected nuclei land in distinct segments. The *site* was localized with
  pseudo-GT, but the *rule* uses no GT. Using GT to find a bug is fine; shipping GT is not.
- The slenderness / caliber-ratio vetoes discussed in `lessons.md` did **not** ship — no
  match in `lib/abiss/src` or `lib/abiss/scripts`.

**LEAK — confirmed, T0 used for selection:**

`abiss_tune.py` optimizes decode parameters against **test_50**.

- `abiss_tune.py:1026` returns `summary["metrics"]["human"]["mt50"]["base"]["linear_nerl"]`;
  the declared headline is `"human linear-L NERL at merge_threshold=50"`.
- `"human"` resolves to `abiss_tuning.yaml: human_gt_dir: chunk_gt_skel`.
- Those crops are test_50. Verified by exact geometry, not by name: `chunk_z4_y6_x1.h5` holds
  base ids `19, 23, 24, 43, 46`, and with chunk size 1008, **100%** of `'19_0'`'s vertices are
  exactly `test_50['19']`. `chunk_z2_y10_x7.h5` likewise reproduces `test_50['0']` exactly.
  The other tuning chunks (`z1_y7_x3`, `z2_y4_x1`) are 16–19 test_50 ids plus a few 10x ids.
- The yaml's whole-volume rung does say "select using valid_12 and score test_50 once", which
  is the right protocol — but the earlier per-chunk racing rungs that decide which trials
  survive are scored on test_50, so the funnel is contaminated upstream of the clean rung.

**Consequence.** This does not invalidate a reported score: 0.481614 is an honest
measurement of a fixed segmentation. It does mean watershed/agglomeration parameters were
chosen with test-set visibility, so the number is an **optimistically biased estimate of
generalization**, and comparisons to FFN — which never saw these skeletons — are not matched
in that respect. The magnitude is unmeasured and is plausibly the size of the effects being
chased: a per-chunk parameter "gain" of +0.021 on this dataset became **−0.031** under honest
cross-validation, and matchguard's whole reported gain is +0.0121.

**Required before the next parameter claim:** rebuild the tuning rungs on T1 only
(`data/ffn_pseudogt`, and `valid_12` minus the 2 shared neurons), and re-derive
`AGG_THRESHOLD` there. The matchguard run's `AGG_THRESHOLD = 0.3` currently has no recorded
T1 provenance, and `aggsweep/a1_thr025.yaml` motivates its arm by a test-set statistic
("targets the 230.7 fragments/skeleton"). Record the provenance of every shipped parameter
in the run directory.

**Not audited:** `lib/abiss/src` C++ beyond the veto grep; the affinity training data split.

## Incident record: 2026-08-14

The first `arm0_96 + nucleus competition` whole-volume attempt was launched with the local
backend and seven GNU Parallel workers from a Codex-managed execution session. It stopped
after 245 successful atomic chunks. The original log contained no traceback, worker error,
OOM, disk error, or kernel OOM event; it ended between successful chunks and the entire
launcher process tree disappeared. This is the signature of the hosting session reaping a
local process, not an ABISS data failure.

The run was migrated to `submit_wholevol_sharded.sh`. Its existing DONE flags were reused,
and the competitive nucleus job was inserted between watershed remap and constrained
agglomeration. The prevention is simple: prepare locally, execute whole-volume work through
Slurm, and record every submitted job ID.

## Incident record: 2026-08-14, incomplete NFS chunk-store listings

The first sharded `arm0_96 + nucleus competition` resume completed atomic watershed L0 but
failed in composite L1 with `ws2: something is wrong in merge`. Five one-CPU reproductions
failed identically, ruling out memory pressure and TBB concurrency. Each failed composite
contained one atomic child whose saved boundary affinity was partly or wholly zero at a
1008-voxel HDF5 seam. Fresh reads matched the verified baseline.

Cause: every worker used `os.listdir()` to rediscover the 726-file affinity store. Under the
large Slurm fan-out, a few NFS directory listings omitted a file. `_ChunkedH5Array` treated an
unlisted file as an intentionally absent tile and silently returned zeros, producing locally
valid but mutually incompatible watershed chunks.

Prevention:

- production HDF5 chunk stores derive their complete deterministic grid from the ABISS `BBOX`;
- workers open `chunk_z*_y*_x*.h5` directly and never enumerate that directory;
- a genuinely missing required file raises instead of zero-filling;
- any L0 produced by the old reader is contaminated until proven otherwise. Start a fresh
  namespace and regenerate L0; do not resume its composite layers.

## Incident record: 2026-08-15, conda activation under `nounset`

The first scheduled nucleus-competition jobs exited in two to four seconds before running
Python. `activate-binutils_linux-64.sh` reads `ADDR2LINE` while it may be unset, and the
wrapper had enabled `set -u` before sourcing conda. Slurm reported `FAILED (1:0)` with only
`ADDR2LINE: unbound variable` in the log; every downstream `afterok` job was then canceled.

Prevention: nucleus and shard wrappers must use `set -eo pipefail`, explicitly `set +u`,
source the `pytc` environment, and only then enable `set -u`. When recovering, submit the
nucleus stage as a new Slurm job. After it completes and its manifest/territory files are
validated, restart at `START_AT=me_L0`; the canceled dependency chain does not revive by
itself, and watershed work must not be rerun.

## Incident record: 2026-08-16, window override without output suffix

The first full native-window affinity array used the correct `[48,96,96]` window but the
tutorial omitted `default.decoding.save_suffix`. Chunked raw prediction naming still reads
that suffix even when decoding is disabled, so the job targeted the already-complete
default `[144,144,144]` store and printed `already exists, skipping`. The array was stopped;
the default store already had all 726 chunks, so no wrong-window prediction was written.

Prevention: the arm0_96 tutorial must set both the window and
`save_suffix: zebrafinch_chunk_raw_grid1008_halo72_win48x96x96`. Always inspect the first
live shard through the line that prints its output directory before trusting a large array.

## Incident record: 2026-08-16, competitive labels lost during final remap

A corrected nucleus-aware gate reached the top aggregation level with zero
`load_conflict_collisions`, retained distinct owner-1 and owner-4 records, and wrote a
nucleus-rejected edge. Nevertheless, the materialized segmentation still assigned source
nuclei 611 and 651 to one segment. The graph was correct; the output volume was not.

Cause: competitive territories and deterministic owner labels were applied by
`cut_chunk_agg.py` while constructing the RAG. The final aggregation remap independently
ran `cut_chunk_remap.py`, reloaded the unsplit watershed from `WS_PATH`, and applied the
hierarchy's remap table to those original labels. New competitive labels therefore existed
inside the graph but not in the volume being remapped. The writer silently collapsed the
split back to the original parent's representative.

Prevention:

- every deterministic label transform used to build the RAG must be replayed on the exact
  source cutout passed to the final remap writer;
- `cut_chunk_remap.py` must call the same `apply_nucleus_competition` implementation as the
  atomic aggregation path before `ws3` applies remaps;
- zero repair territories must be a no-op, and every qualified soma piece belonging to one
  repaired owner must use one shared protected label rather than a label per watershed piece;
- changing either path changes the ABISS runtime hash and requires fresh gates;
- a clean internal collision count is necessary but insufficient. Require
  `fused_source_pairs == 0` and distinct dominant final segments for the known 611/651 pair
  in the materialized output;
- retain an integration test proving that the remap writer saves the refined cutout rather
  than the base watershed cutout.

The acceptance sequence is now: verify the competition manifest, verify hierarchy nucleus
records and rejected edges, materialize the final segmentation, then run the nucleus-instance
audit and canonical funlib NERL at merge threshold 50. Do not promote a run before the last
two checks.

## Related technical records

- `lesson_abiss.md`: ABISS fidelity, build/runtime bugs, and the whole-volume runner.
- `lesson_efficiency.md`: measured memory/concurrency choices and resharding results.
- `lesson_soma_linking.md`: nucleus/soma identity strategy.
- `seuron_reproduction.md`: reproduction and scoring details.
