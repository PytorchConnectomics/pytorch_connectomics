#!/bin/bash
# mip0 step 1 on the weilab partition's A10s.
#
# WHY NOT `short`: on 2026-09-20 every GPU in `short` was allocated (g002-g018 and
# gb001 all at N/N) and five GPU nodes were down, so job 3025968 pended with no
# start estimate. gdw001 in the `weilab` partition had 2 of 4 a10s free, ~383 GB
# allocatable, and MaxTime=UNLIMITED. The a10 is slower than an l40s, but an
# 8-hour run that starts now beats a 4-hour run that starts whenever.
#
# Output location is NOT set here -- checkpoint_dispatch derives it from the
# checkpoint path, so this writes to
#   outputs/liconn_final_banis_plus_mip0/20260908_164220/
#     test_step=00200000/val/raw_x1_ch0-1-2.h5
# The leaf is `val`, from the DATA PATH, not test.data.test.name.
#
# MEMORY, from the whole-volume lazy accumulator on 16.65 Gvox:
#   value_accumulator  (1,3,290,8580,6690) fp16   99.9 GB
#   weight_accumulator (1,1,...)           fp16   33.3 GB
#   clamp_min copy of the weight map       fp16   33.3 GB (freed after the divide)
# ~170 GB + runtime. normalize_weighted_accumulator divides IN PLACE, so there is
# no second copy of the 100 GB value array. 300G is ~50% headroom and fits inside
# what gdw001 has free. This is NOT the ~785 GB decode figure -- different step.
#SBATCH --job-name=mip0_val_aff
#SBATCH --partition=weilab
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_mip0_1_affinity_a10.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_mip0_1_affinity_a10.out

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
free -g | head -2

python scripts/main.py --config tutorials/neuron_liconn_ist/1_affinity_mip0_a10.yaml \
  --mode test --checkpoint "$CKPT"

echo "--- inference returned, listing the run dir ---"
ls -la "$RUN/test_step=00200000/"

# Non-degeneracy gate: std > 0.01 and max <= 0.95 (a max above the scale_sigmoid
# range would mean the activation changed and every inherited threshold is void).
# The eb2 18 nm quartiles 0.26/0.53/0.68 are context, not a pass condition --
# different model, different grid.
python tutorials/neuron_liconn_ist/slurm/check_affinity.py "$OUT"
echo "DONE end=$(date -Is)"
