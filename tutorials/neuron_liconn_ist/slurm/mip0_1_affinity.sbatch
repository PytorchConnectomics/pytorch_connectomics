#!/bin/bash
# mip0 step 1: IST LICONN held-out val at SOURCE resolution -> affinity, from the
# 4-GPU mip0 200k checkpoint. First time this checkpoint has ever touched a
# volume that has ground truth.
#
# Output location is NOT set here. checkpoint_dispatch derives it from the
# checkpoint path (nearest YYYYMMDD_HHMMSS ancestor), so this writes to
#   outputs/liconn_final_banis_plus_mip0/20260908_164220/
#     test_step=00200000/val/raw_x1_ch0-1-2.h5
# alongside the four existing ExPID96/99 moe dirs, overwriting none of them.
# The leaf is `val`, from the DATA PATH's last component, NOT from
# test.data.test.name (`liconn_ist_mip0_val`). Assuming the latter cost job
# 3013750 its exit status on 2026-09-16 after its inference had succeeded, and
# cost 3014237_0 the same on the moe side.
#
# SIZING. Volume is (290, 8580, 6690) = 16.65 Gvox, exactly 8x the 18 nm val.
# Whole-volume lazy accumulation holds, at peak:
#   value_accumulator  (1,3,290,8580,6690) fp16   99.9 GB
#   weight_accumulator (1,1,...)           fp16   33.3 GB
#   clamp_min copy of the weight map       fp16   33.3 GB   (freed after divide)
# = ~166 GB + runtime overhead. normalize_weighted_accumulator divides IN PLACE
# (window.py:275-294), so there is no second copy of the 100 GB value array.
# 400G matches what the moe mip0 volumes ran at and leaves ~2x headroom.
# This is NOT the 1.2 TB figure -- that is the ABISS decode in step 2, which is
# a different problem with a different answer.
#
# Window count is ~2.9x the moe ExPID99 run (70.4k vs 24.6k at stride 64), which
# took 87 min; expect ~4-5 h including the ~100 GB h5 write.
#SBATCH --job-name=mip0_val_aff
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_mip0_1_affinity.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_mip0_1_affinity.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

RUN="$REPO/outputs/liconn_final_banis_plus_mip0/20260908_164220"
CKPT="$RUN/checkpoints/step=00200000.ckpt"
OUT="$RUN/test_step=00200000/val/raw_x1_ch0-1-2.h5"

echo "host=$(hostname) start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /projects | tail -1

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_mip0.yaml \
  --mode test --checkpoint "$CKPT"

echo "--- inference returned, listing the run dir ---"
ls -la "$RUN/test_step=00200000/"

# Non-degeneracy gate. The eb2 18 nm val reference is p25/p50/p75 0.26/0.53/0.68;
# this is a different model on a different grid, so the quartiles are context,
# not a pass condition. The pass conditions are std > 0.01 and max <= 0.95.
python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$OUT"
echo "DONE end=$(date -Is)"
