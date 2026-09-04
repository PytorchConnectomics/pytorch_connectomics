#!/bin/bash
# Step 0: resample each moe volume onto the checkpoint's [24,18,18] nm grid.
# Array index = position in tutorials/neuron_liconn_moe/volumes.py::PENDING.
#SBATCH --job-name=moe_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_0_prepare.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_0_prepare.out

# No `set -u`: conda activate trips on unbound ADDR2LINE.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from the repo root
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) task=$SLURM_ARRAY_TASK_ID start=$(date -Is)"
python tutorials/neuron_liconn_moe/run_prepare.py --index "$SLURM_ARRAY_TASK_ID" --overwrite
echo "DONE end=$(date -Is)"
