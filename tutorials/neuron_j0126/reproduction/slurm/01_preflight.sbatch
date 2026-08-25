#!/bin/bash
#SBATCH --job-name=j0126-preflight
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPOSITORY:?}"
: "${PYTHON:?}"
: "${AFFINITY_CHUNKS:?}"
: "${AFFINITY_INDEX:?}"
: "${KEEP_MASK:?}"
: "${NUCLEUS:?}"
: "${SKELETONS:?}"
: "${ABISS_HOME:?}"
: "${PYTC_PREFIX:?}"
: "${STORAGE_ROOT:?}"
: "${PREFLIGHT_JSON:?}"

export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPOSITORY"
exec "$PYTHON" \
  tutorials/neuron_j0126/reproduction/preflight.py \
  --affinity-chunks "$AFFINITY_CHUNKS" \
  --affinity-index "$AFFINITY_INDEX" \
  --keep-mask "$KEEP_MASK" \
  --nucleus "$NUCLEUS" \
  --skeletons "$SKELETONS" \
  --abiss-home "$ABISS_HOME" \
  --pytc-prefix "$PYTC_PREFIX" \
  --storage-root "$STORAGE_ROOT" \
  --output "$PREFLIGHT_JSON" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
