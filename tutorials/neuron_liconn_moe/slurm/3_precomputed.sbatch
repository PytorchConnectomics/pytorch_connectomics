#!/bin/bash
# Step 4: build the neuroglancer precomputed layer + meshes for one volume.
# Upload is NOT done here -- it needs an interactive `gcloud auth login`.
#SBATCH --job-name=moe_ng
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_3_precomputed.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_3_precomputed.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from the repo root
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

VOL=$(python -c "import sys;sys.path.insert(0,'tutorials/neuron_liconn_moe');import volumes as V;print(V.PENDING[$SLURM_ARRAY_TASK_ID])")
echo "host=$(hostname) volume=$VOL start=$(date -Is)"
python tutorials/neuron_liconn_moe/upload_seg_precomputed.py --volume "$VOL" --create --downsample --mesh --parallel 16
echo "DONE end=$(date -Is)"
