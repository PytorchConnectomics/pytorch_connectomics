#!/bin/bash
#SBATCH --job-name=j0126-eval
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPOSITORY:?}"
: "${PYTHON:?}"
: "${SKELETONS:?}"
: "${EVALUATION_JSON:?}"
if [ -n "${SEGMENTATION:-}" ] && [ -n "${NODE_LUT:-}" ]; then
  echo "set only one of SEGMENTATION or NODE_LUT" >&2
  exit 2
elif [ -n "${SEGMENTATION:-}" ]; then
  source_args=(--segmentation "$SEGMENTATION")
elif [ -n "${NODE_LUT:-}" ]; then
  source_args=(--node-lut "$NODE_LUT")
else
  echo "SEGMENTATION or NODE_LUT is required" >&2
  exit 2
fi

export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPOSITORY"
exec "$PYTHON" tutorials/neuron_j0126/reproduction/evaluate.py \
  "${source_args[@]}" \
  --skeletons "$SKELETONS" \
  --output "$EVALUATION_JSON"
