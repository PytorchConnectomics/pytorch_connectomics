#!/bin/bash
#SBATCH --job-name=z600_smoke
#SBATCH --partition=weilab
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_2_affinity_smoke.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_2_affinity_smoke.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
RUN="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436"
CKPT="$RUN/checkpoints/step=00200000.ckpt"
echo "host=$(hostname) start=$(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_z600_smoke.yaml \
  --mode test --checkpoint "$CKPT"
ls -la "$RUN/test_step=00200000/"
python tutorials/neuron_liconn_ist/slurm/check_affinity.py \
  "$RUN/test_step=00200000/val_z600_smoke/raw_x1_ch0-1-2.h5"
echo "DONE end=$(date -Is)"
