#!/bin/bash
# 600 nm gate step 2: the released 18 nm checkpoint on the Z-degraded val volume.
# Card MSIDEPLOY-MODEL-002, Stage 1 step 2. Same weights as the 0.9130 baseline;
# the only thing that changes is the input.
#
# WHY THE `weilab` a10 AND NOT `short`: at launch time `short` had 32 queued
# l40s jobs of this user's own and no running ones, while gdw001 was fully idle
# with 4 free a10s. An a10 run that starts now beats an l40s run that starts
# whenever. Note `weilab` REJECTS any job without a --gres=gpu request.
#
# SIZING, from the lazy whole-volume accumulator on the PADDED 128x4290x3345
# (see 1_affinity_z600.yaml note 2):
#   value_accumulator  (1,3,128,4290,3345) fp16  11.0 GB
#   weight_accumulator (1,1,...)           fp16   3.7 GB
#   clamp_min copy of the weight map       fp16   3.7 GB  (freed after the divide)
# ~19 GB + runtime + the ~9 GB h5 write. 120G is ~5x headroom.
#
# The smoke run (job 3031066, 104x640x640) already proved the pad/crop path:
# affinity came back (3, 104, 640, 640), quartiles 0.250/0.512/0.669 against the
# eb2 18 nm reference 0.26/0.53/0.68, max 0.786.
#SBATCH --job-name=z600_aff
#SBATCH --partition=weilab
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_2_affinity.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_2_affinity.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
RUN="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436"
CKPT="$RUN/checkpoints/step=00200000.ckpt"
# Leaf is `val_z600`, from the DATA PATH. NOT `val` -- that is the published
# 18 nm affinity behind the baseline and must not be overwritten.
OUT="$RUN/test_step=00200000/val_z600/raw_x1_ch0-1-2.h5"
echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; free -g | head -2
ls -la "$RUN/test_step=00200000/val/raw_x1_ch0-1-2.h5"

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_z600.yaml \
  --mode test --checkpoint "$CKPT"

echo "--- inference returned ---"
ls -la "$RUN/test_step=00200000/val_z600/"
ls -la "$RUN/test_step=00200000/val/raw_x1_ch0-1-2.h5"
# Non-degeneracy gate (card acceptance): std > 0.01 and max <= 0.95. A max at or
# above 0.95 would mean the activation changed and every inherited threshold is void.
python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$OUT"
echo "DONE end=$(date -Is)"
