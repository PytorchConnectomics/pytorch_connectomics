#!/bin/bash
# 36 nm route B step 1: build final_proofread_z36/{train_o0,train_o1,train_o2,val_z36}
# from the mip0 export (Z decimation x3, XY 2x2 block mean / label stride), then
# measure the label damage. Card MSIDEPLOY-MODEL-002 Stage 2; brief_36nm.md.
#
# Memory: 8 workers x (96x1024x1024 uint64 seg 0.8 GB + img 0.1 GB + outputs) ~ 10 GB;
# label damage holds one 24 nm plane at a time plus the (id, plane, count) table.
# No GPU.
#SBATCH --job-name=z36_build
#SBATCH --partition=short
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z36_1_build.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z36_1_build.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
/usr/bin/time -v python tutorials/neuron_liconn_ist/z36_build.py \
  --out /projects/weilab/dataset/liconn/pytc/final_proofread_z36 --workers 8
python tutorials/neuron_liconn_ist/z36_label_damage.py \
  --json "$REPO/outputs/neuron_liconn_ist/z36/label_damage.json"
echo "DONE end=$(date -Is)"
