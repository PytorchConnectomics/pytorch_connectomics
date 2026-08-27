#!/bin/bash
"${BASH_VERSION:+true}" 2>/dev/null || exit 2
set -euo pipefail

: "${RUNTIME_JSON:?}"
: "${RUN_ROOT:?}"
: "${PYTHON:?}"
: "${ABISS_LIBRARY_PREFIX:?}"
: "${SKELETONS:?}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
slurm_dir="$script_dir/slurm"
partition=${PARTITION:-medium}
prefix=${JOB_PREFIX:-j0126}
dependency=${INIT_DEP:-}
top_mip=$("$PYTHON" -c "import json; print(json.load(open('$RUNTIME_JSON'))['top_mip'])")
mkdir -p "$RUN_ROOT/logs"
submission_log="$RUN_ROOT/submission.tsv"
printf 'stage\tjob_id\tdependency\tpartition\n' > "$submission_log"

layer_resources() {
  case "$1" in
    0) echo "80 19 180G" ;;
    1) echo "24 16 130G" ;;
    2) echo "16 4 96G" ;;
    3) echo "14 2 96G" ;;
    4) echo "8 1 120G" ;;
    *) echo "1 1 300G" ;;
  esac
}

submit_layer() {
  local name=$1 op=$2 layer=$3 stage_env=$4
  local shards cpus memory job_id
  read -r shards cpus memory <<<"$(layer_resources "$layer")"
  local dependency_arg=()
  if [ -n "$dependency" ]; then
    dependency_arg=(--dependency="afterok:$dependency")
  fi
  job_id=$(sbatch --parsable "${dependency_arg[@]}" \
    --partition="$partition" \
    --job-name="${prefix}_${name}" \
    --array="0-$((shards - 1))" \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --output="$RUN_ROOT/logs/%x_%A_%a.out" \
    --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",PYTHON="$PYTHON",ABISS_LIBRARY_PREFIX="$ABISS_LIBRARY_PREFIX",OP="$op",LAYER="$layer",STAGE_ENV="$stage_env",NSHARD="$shards" \
    "$slurm_dir/04_abiss_shard.sbatch")
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\n' "$name" "$job_id" "$dependency" "$partition" \
    >> "$submission_log"
  echo "$name -> $job_id${dependency:+ after $dependency}"
  dependency=$job_id
}

submit_nucleus() {
  local enabled nucleus_job dependency_arg=()
  enabled=$("$PYTHON" -c \
    "import json; r=json.load(open('$RUNTIME_JSON')); p=json.load(open(r['param'])); print(int(bool(p.get('NUC_PATH')) and p.get('NUC_COMPETITION_ENABLED', True)))")
  if [ "$enabled" != 1 ]; then
    echo "nucleus competition disabled"
    return 0
  fi
  if [ -n "$dependency" ]; then
    dependency_arg=(--dependency="afterok:$dependency")
  fi
  nucleus_job=$(sbatch --parsable "${dependency_arg[@]}" \
    --partition="${NUC_PARTITION:-long}" \
    --job-name="${prefix}_nucleus" \
    --output="$RUN_ROOT/logs/%x_%j.out" \
    --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",PYTHON="$PYTHON",ABISS_LIBRARY_PREFIX="$ABISS_LIBRARY_PREFIX" \
    "$slurm_dir/04_nucleus_competition.sbatch")
  nucleus_job=${nucleus_job%%;*}
  printf '%s\t%s\t%s\t%s\n' nucleus_competition "$nucleus_job" "$dependency" "${NUC_PARTITION:-long}" >> "$submission_log"
  echo "nucleus_competition -> $nucleus_job${dependency:+ after $dependency}"
  dependency=$nucleus_job
}

echo "Submitting $prefix full replay, top_mip=$top_mip, partition=$partition"
for layer in $(seq 0 "$top_mip"); do
  operation=composite_chunk_ws
  if [ "$layer" -eq 0 ]; then operation=atomic_chunk_ws; fi
  submit_layer "ws${layer}" "$operation" "$layer" ws
done
submit_layer remapws remap_chunk_ws 0 ws
submit_nucleus
for layer in $(seq 0 "$top_mip"); do
  operation=composite_chunk_me
  if [ "$layer" -eq 0 ]; then operation=atomic_chunk_me; fi
  submit_layer "me${layer}" "$operation" "$layer" agg
done
submit_layer remapagg remap_chunk_agg 0 agg

segmentation=$("$PYTHON" -c \
  "import json; from urllib.parse import unquote; print(unquote(json.load(open('$RUNTIME_JSON'))['seg_path']).removeprefix('file://'))")
evaluation_job=$(sbatch --parsable \
  --dependency="afterok:$dependency" \
  --partition=short \
  --job-name="${prefix}_eval" \
  --output="$RUN_ROOT/logs/%x_%j.out" \
  --export=ALL,REPOSITORY="$("$PYTHON" -c "import json; print(json.load(open('$RUNTIME_JSON'))['repository'])")",PYTHON="$PYTHON",SEGMENTATION="$segmentation",NODE_LUT=,SKELETONS="$SKELETONS",EVALUATION_JSON="$RUN_ROOT/evaluation.json" \
  "$slurm_dir/05_evaluate.sbatch")
evaluation_job=${evaluation_job%%;*}
printf '%s\t%s\t%s\t%s\n' evaluate "$evaluation_job" "$dependency" short >> "$submission_log"
printf '%s\n' "$dependency" > "$RUN_ROOT/final_decode_jobid.txt"
printf '%s\n' "$evaluation_job" > "$RUN_ROOT/evaluation_jobid.txt"
echo "final decode -> $dependency; evaluation -> $evaluation_job"
