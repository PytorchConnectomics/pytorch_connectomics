#!/bin/bash
#SBATCH --job-name=j0126-smoke
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

# Frozen ABISS derives its ID namespace and hierarchy from the full-volume
# geometry. Cropped multi-chunk boxes can therefore fail in later RAG stages
# even when watershed succeeds. A single native chunk at the origin is the
# reliable binary/runtime smoke; the full-volume regression is the semantic
# gate for hierarchy composition.
read -r -a bbox <<<"${BBOX_XYZ:-0 0 0 512 512 256}"
if [ "${#bbox[@]}" -ne 6 ]; then
  echo "BBOX_XYZ must contain six integers" >&2
  exit 2
fi

export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPOSITORY"
exec "$PYTHON" scripts/run_seuron_provenance.py \
  --config "$CONFIG" \
  --out-root "$OUT_ROOT" \
  --name "$RUN_NAME" \
  --exec-bbox "${bbox[@]}" \
  --mode fresh \
  --stages watershed remap_watershed \
  --execute
