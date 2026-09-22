#!/bin/bash
# 600 nm gate, CONTROL inference: the 18 nm checkpoint on val_z600rep -- the same
# Z information loss as the primary degraded volume, on the baseline's own
# 145-plane grid and window geometry. Card MSIDEPLOY-MODEL-002, Stage 1.
# Its VOI is directly comparable to 0.9130: no re-grid, no oracle floor.
#
# Sizing is the baseline's: value_accumulator (1,3,145,4290,3345) fp16 12.5 GB,
# weight 4.2 GB, plus the transient clamp copy. 120G is ample.
#SBATCH --job-name=z600_affrep
#SBATCH --partition=weilab
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_2b_affinity_rep.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_2b_affinity_rep.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
RUN="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436"
CKPT="$RUN/checkpoints/step=00200000.ckpt"
OUT="$RUN/test_step=00200000/val_z600rep/raw_x1_ch0-1-2.h5"
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_z600rep.yaml \
  --mode test --checkpoint "$CKPT"
ls -la "$RUN/test_step=00200000/val_z600rep/"
ls -la "$RUN/test_step=00200000/val/raw_x1_ch0-1-2.h5"
python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$OUT"
echo "DONE end=$(date -Is)"
