#!/bin/bash
#SBATCH --job-name=j0126-ec
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail
: "${REPOSITORY:?}"
: "${PYTHON:?}"
: "${CONFIG:?}"
: "${STAGE:?}"

export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPOSITORY"
arguments=(--config "$CONFIG" --stage "$STAGE")
case "$STAGE" in
  skeletonize|contacts|postprocess)
    : "${SLURM_ARRAY_TASK_ID:?}"
    arguments+=(
      --task-id "$SLURM_ARRAY_TASK_ID"
      --num-tasks "${NUM_TASKS:-80}"
    )
    ;;
esac
if [ "${OVERWRITE:-0}" = 1 ]; then
  arguments+=(--overwrite)
fi
exec "$PYTHON" scripts/run_error_correction.py "${arguments[@]}"
