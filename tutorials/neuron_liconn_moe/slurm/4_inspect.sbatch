#!/bin/bash
# Diagnostic: are the biggest segments merge chains or somata? (shape, not size)
# Array index = position in volumes.py::VOLUMES (all 8, including the published one).
#SBATCH --job-name=moe_inspect
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_4_inspect.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_4_inspect.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from the repo root
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

VOL=$(python -c "import sys;sys.path.insert(0,'tutorials/neuron_liconn_moe');import volumes as V;print(list(V.VOLUMES)[$SLURM_ARRAY_TASK_ID])")
echo "host=$(hostname) volume=$VOL start=$(date -Is)"
python tutorials/neuron_liconn_moe/inspect_top_segments.py --volume "$VOL"
echo "DONE end=$(date -Is)"
