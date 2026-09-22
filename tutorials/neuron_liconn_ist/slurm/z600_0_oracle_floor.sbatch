#!/bin/bash
# 600 nm gate step 0: the oracle floor + the 1.3889x label-damage numbers.
# Card MSIDEPLOY-MODEL-002. No GPU. Runs BEFORE the inference job, because if the
# oracle floor is comparable to the ~0.01 VOI noise floor the expensive half is
# not interpretable as a statement about the model.
#
# MEMORY. The scoring is the same `voi`/`adapted_rand` the 0.9130 baseline used,
# which materialises both volumes and a float64 `data` array over 2.08 Gvox:
#   gt uint64 16.6 GB + oracle uint64 16.6 GB + contingency `data` f64 16.6 GB
#   + coo row/col int64 33.2 GB + the csc copy ~33 GB  ~= 120 GB, plus transients.
# The measured peak of the equivalent whole-val scoring path was 147 GB. 400G is
# ~3x that and matches what wholeval_sweep.sbatch already runs at on the cb nodes.
# NOT the `weilab` partition: it rejects any job without a --gres=gpu request.
#SBATCH --job-name=z600_oracle
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_0_oracle_floor.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_0_oracle_floor.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"; free -g | head -2
/usr/bin/time -v python tutorials/neuron_liconn_ist/z600_oracle_floor.py \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/z600_oracle_floor.json"
echo "DONE end=$(date -Is)"
