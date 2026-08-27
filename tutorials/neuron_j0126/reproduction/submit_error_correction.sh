#!/bin/bash
set -euo pipefail

: "${REPOSITORY:?}"
: "${PYTHON:?}"
: "${CONFIG:?}"
: "${RUN_ROOT:?}"
: "${SKELETONS:?}"
: "${SEGMENTATION:?Final v7 output segmentation path}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/slurm/06_error_correction_stage.sbatch"
dependency=${INIT_DEP:-}
prefix=${JOB_PREFIX:-j0126ec}
mkdir -p "$RUN_ROOT/logs"
submission_log="$RUN_ROOT/submission.tsv"
printf 'stage\tjob_id\tdependency\tpartition\n' > "$submission_log"

resources() {
  case "$1" in
    sizes) echo "medium 8 64G 04:00:00 single" ;;
    skeletonize) echo "long 8 64G 2-00:00:00 array" ;;
    skeletons) echo "long 8 240G 2-00:00:00 single" ;;
    contacts) echo "long 2 64G 1-00:00:00 array" ;;
    contact_graph) echo "long 8 180G 12:00:00 single" ;;
    candidates) echo "medium 8 96G 12:00:00 single" ;;
    junction_scope) echo "medium 4 64G 12:00:00 single" ;;
    junction_features) echo "long 8 240G 2-00:00:00 single" ;;
    boundary) echo "medium 8 128G 12:00:00 single" ;;
    resolve) echo "medium 4 96G 12:00:00 single" ;;
    prepare_output) echo "short 4 32G 02:00:00 single" ;;
    postprocess) echo "long 1 16G 1-00:00:00 array" ;;
    verify) echo "short 4 32G 04:00:00 single" ;;
    *) return 2 ;;
  esac
}

submit_stage() {
  local stage=$1 partition cpus memory wall mode job_id
  read -r partition cpus memory wall mode <<<"$(resources "$stage")"
  local dependency_arg=()
  local array_arg=()
  if [ -n "$dependency" ]; then
    dependency_arg=(--dependency="afterok:$dependency")
  fi
  if [ "$mode" = array ]; then
    array_arg=(--array="0-79%40")
  fi
  job_id=$(sbatch --parsable \
    "${dependency_arg[@]}" \
    "${array_arg[@]}" \
    --partition="$partition" \
    --job-name="${prefix}_${stage}" \
    --cpus-per-task="$cpus" \
    --mem="$memory" \
    --time="$wall" \
    --output="$RUN_ROOT/logs/%x_%A_%a.out" \
    --export=ALL,REPOSITORY="$REPOSITORY",PYTHON="$PYTHON",CONFIG="$CONFIG",STAGE="$stage",NUM_TASKS=80 \
    "$runner")
  job_id=${job_id%%;*}
  printf '%s\t%s\t%s\t%s\n' "$stage" "$job_id" "$dependency" "$partition" \
    >> "$submission_log"
  echo "$stage -> $job_id${dependency:+ after $dependency}"
  dependency=$job_id
}

for stage in \
  sizes skeletonize skeletons contacts contact_graph candidates junction_scope \
  junction_features boundary resolve prepare_output postprocess verify; do
  submit_stage "$stage"
done

evaluation_job=$(sbatch --parsable \
  --dependency="afterok:$dependency" \
  --partition=short \
  --job-name="${prefix}_eval" \
  --output="$RUN_ROOT/logs/%x_%j.out" \
  --export=ALL,REPOSITORY="$REPOSITORY",PYTHON="$PYTHON",SEGMENTATION="$SEGMENTATION",SKELETONS="$SKELETONS",EVALUATION_JSON="$RUN_ROOT/evaluation.json" \
  "$script_dir/slurm/05_evaluate.sbatch")
evaluation_job=${evaluation_job%%;*}
printf '%s\t%s\t%s\t%s\n' evaluate "$evaluation_job" "$dependency" short >> "$submission_log"
printf '%s\n' "$dependency" > "$RUN_ROOT/final_jobid.txt"
printf '%s\n' "$evaluation_job" > "$RUN_ROOT/evaluation_jobid.txt"
echo "error correction -> $dependency; evaluation -> $evaluation_job"
