#!/bin/bash
# eb8 step 1: IST LICONN held-out val -> affinity, from the 4-GPU 200k checkpoint.
#
# Output location is NOT set here. checkpoint_dispatch derives it from the
# checkpoint path (nearest YYYYMMDD_HHMMSS ancestor), so this writes to
#   outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/
#     test_step=00200000/val/raw_x1_ch0-1-2.h5
# alongside the existing ExPID96/99 inference dirs, overwriting none of them.
#
# The eb2 counterpart is the same run's test phase from 2026-07-30, at
# .../liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/ --
# the same `val` leaf, for the same reason.
#SBATCH --job-name=eb8_val_aff
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_eb8_1_affinity.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_eb8_1_affinity.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

RUN="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545"
CKPT="$RUN/checkpoints/step=00200000.ckpt"
# The per-volume leaf is `val` -- taken from the DATA PATH's last component
# (params.data.val = .../final_proofread/val), NOT from test.data.test.name,
# which is `liconn_ist_val`. Assuming the latter cost a 41-minute job its exit
# status on 2026-09-16 (the inference itself had already succeeded).
OUT="$RUN/test_step=00200000/val/raw_x1_ch0-1-2.h5"

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity.yaml \
  --mode test --checkpoint "$CKPT"

python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$OUT"
echo "DONE end=$(date -Is)"
