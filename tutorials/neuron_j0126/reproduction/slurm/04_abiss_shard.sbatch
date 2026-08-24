#!/bin/bash
#SBATCH --job-name=j0126-shard
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${RUNTIME_JSON:?}"
: "${PYTHON:?}"
: "${ABISS_LIBRARY_PREFIX:?}"
: "${OP:?}"
: "${LAYER:?}"
: "${STAGE_ENV:?}"

export PATH="$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$ABISS_LIBRARY_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_USE_FILE_LOCKING=FALSE

read_runtime() {
  "$PYTHON" -c "import json; print(json.load(open('$RUNTIME_JSON'))['$1'])"
}

workdir=$(read_runtime workdir)
abiss_home=$(read_runtime abiss_home)
secrets=$(read_runtime secrets)
cloud_volume_dir=$(read_runtime cloud_volume_dir)
shards=${NSHARD:-1}
shard=${SLURM_ARRAY_TASK_ID:-0}
cpus=${SLURM_CPUS_PER_TASK:-1}

export WORKER_HOME="$abiss_home"
export SECRETS="$secrets"
export STAGE="$STAGE_ENV"
export OVERLAP="${OVERLAP:-0}"
export META="${META:-}"
if [ -n "$cloud_volume_dir" ]; then
  export CLOUD_VOLUME_DIR="$cloud_volume_dir"
fi
export AIRFLOW_TMP_DIR="$workdir/.airflow_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$shard"
mkdir -p "$AIRFLOW_TMP_DIR"
rm -f "$AIRFLOW_TMP_DIR"/.cpulock_*

cd "$workdir"
task_list=".shard_${OP}_L${LAYER}_${shard}of${shards}.txt"
awk -v n="$shards" -v s="$shard" 'NR % n == s' "${LAYER}.txt" > "$task_list"
task_count=$(wc -l < "$task_list")
echo "shard=$shard/$shards op=$OP layer=$LAYER tasks=$task_count cpus=$cpus host=$(hostname)"
if [ "$task_count" -eq 0 ]; then
  exit 0
fi

export TMPDIR="$AIRFLOW_TMP_DIR/parallel"
mkdir -p "$TMPDIR"
parallel --halt 2 -j "$cpus" --tmpdir "$TMPDIR" --compress \
  "$abiss_home/scripts/run_wrapper.sh" . "$OP" {} < "$task_list"
echo "complete shard=$shard/$shards op=$OP layer=$LAYER"
