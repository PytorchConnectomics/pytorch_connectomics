#!/bin/bash
#SBATCH --job-name=j0126-ffn-mip0
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${REPOSITORY:?}"
: "${PYTHON:?}"
: "${EM_ERL_HOME:?}"
: "${SKELETONS:?}"
: "${NODE_LUT:?}"
: "${ERL_PICKLE:?}"
: "${EVALUATION_JSON:?}"

mkdir -p "$(dirname "$NODE_LUT")" "$(dirname "$ERL_PICKLE")" \
  "$(dirname "$EVALUATION_JSON")" "${CACHE_DIR:?}"
export HDF5_USE_FILE_LOCKING=FALSE

cd "$EM_ERL_HOME"
"$PYTHON" -u examples/eval_j0126.py \
  --gt-skeleton "$SKELETONS" \
  --merge-threshold 5 \
  --num-workers "${SLURM_CPUS_PER_TASK:-16}" \
  --mip 0 \
  --cache-dir "$CACHE_DIR" \
  --lut "$NODE_LUT" \
  --output-path "$ERL_PICKLE"

cd "$REPOSITORY"
"$PYTHON" tutorials/neuron_j0126/reproduction/evaluate.py \
  --node-lut "$NODE_LUT" \
  --skeletons "$SKELETONS" \
  --output "$EVALUATION_JSON"
sha256sum "$NODE_LUT" > "$NODE_LUT.sha256"
