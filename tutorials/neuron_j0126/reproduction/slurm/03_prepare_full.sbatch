#!/bin/bash
#SBATCH --job-name=j0126-prepare
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPOSITORY:?}"
: "${PYTHON:?}"
: "${CONFIG:?}"
: "${OUT_ROOT:?}"
: "${RUN_NAME:?}"
: "${RUNTIME_JSON:?}"

export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPOSITORY"
exec "$PYTHON" tutorials/neuron_j0126/reproduction/prepare.py \
  --config "$CONFIG" \
  --out-root "$OUT_ROOT" \
  --name "$RUN_NAME" \
  --runtime-json "$RUNTIME_JSON" \
  --mode "${RUN_MODE:-fresh}"
