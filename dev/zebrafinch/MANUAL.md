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

## Related technical records

- `lesson_abiss.md`: ABISS fidelity, build/runtime bugs, and the whole-volume runner.
- `lesson_efficiency.md`: measured memory/concurrency choices and resharding results.
- `lesson_soma_linking.md`: nucleus/soma identity strategy.
- `seuron_reproduction.md`: reproduction and scoring details.
