#!/bin/bash
# 36 nm step 4: finetune eb2 at [36,18,18] nm. Card MSIDEPLOY-MODEL-002 Stage 2;
# brief_36nm.md. Config header (3_finetune_z36.yaml) has the recipe and why.
#
# ONE GPU, exactly: eb2 = effective batch 2 = batch_size 2 x 1 GPU. `long` (5 d);
# 100k steps is ~26 h at the base run's 50k/13 h.
#
# After launch, confirm in the log that the external weights loaded with
# 0 missing / 0 unexpected keys -- a wrong prefix trains from random init with
# only a warning.
#SBATCH --job-name=z36_ft
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --constraint="vr40g|vr80g"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z36_4_finetune.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z36_4_finetune.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2
python scripts/main.py --config tutorials/neuron_liconn_ist/3_finetune_z36.yaml \
  system.num_gpus=1 system.num_workers=16
echo "DONE end=$(date -Is)"
