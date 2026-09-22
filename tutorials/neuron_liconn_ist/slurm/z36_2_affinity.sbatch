#!/bin/bash
# 36 nm step 2: val affinity for one checkpoint on val_z36 (and, with SECONDARY=1,
# on the 33.3 nm area-averaged val_z600 too). Card MSIDEPLOY-MODEL-002 Stage 2.
#
#   CKPT=<ckpt> sbatch z36_2_affinity.sbatch               # val_z36 only
#   CKPT=<ckpt> SECONDARY=1 sbatch z36_2_affinity.sbatch   # + val_z600
#
# Output leaf comes from the data path: <ckpt run>/test_step=<step>/val_z36/ and
# .../val_z600/. For the eb2 baseline, val_z600 already exists (z600_2_affinity) --
# do NOT pass SECONDARY=1 with the eb2 checkpoint.
#
# Sizing as z600_2_affinity (97 planes padded to 129 vs 104 padded to 128).
#SBATCH --job-name=z36_aff
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint="vr40g|vr80g"
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z36_2_affinity.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z36_2_affinity.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
: "${CKPT:?set CKPT}"
RUN="$(dirname "$(dirname "$CKPT")")"
STEP="$(basename "$CKPT" .ckpt)"
echo "host=$(hostname) start=$(date -Is) ckpt=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_z36.yaml \
  --mode test --checkpoint "$CKPT"
ls -la "$RUN/test_$STEP/val_z36/"
python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$RUN/test_$STEP/val_z36/raw_x1_ch0-1-2.h5"

if [ "${SECONDARY:-0}" = 1 ]; then
  python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_z600.yaml \
    --mode test --checkpoint "$CKPT"
  ls -la "$RUN/test_$STEP/val_z600/"
  python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$RUN/test_$STEP/val_z600/raw_x1_ch0-1-2.h5"
fi
echo "DONE end=$(date -Is)"
