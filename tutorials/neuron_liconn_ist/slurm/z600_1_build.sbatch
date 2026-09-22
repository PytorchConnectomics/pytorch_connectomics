#!/bin/bash
# 600 nm gate step 1: build the Z-degraded val image (145 -> 104 planes, exact
# area average along Z, XY untouched). Card MSIDEPLOY-MODEL-002.
#
# Streams 2-3 input planes per output plane, so peak RSS is a few hundred MB;
# 24G is pure headroom for zarr/blosc buffers. No GPU.
#SBATCH --job-name=z600_build
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_1_build.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_1_build.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
/usr/bin/time -v python tutorials/neuron_liconn_ist/z600_build_degraded.py \
  --out /projects/weilab/dataset/liconn/pytc/final_proofread_z600/val_z600
echo "DONE end=$(date -Is)"
