#!/bin/bash
# Affinity QC montages for all eight volumes, against the in-domain IST val run.
#SBATCH --job-name=moe_qc
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%j_5_qc.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%j_5_qc.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from the repo root
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
python tutorials/neuron_liconn_moe/qc_affinity.py
echo "DONE end=$(date -Is)"
