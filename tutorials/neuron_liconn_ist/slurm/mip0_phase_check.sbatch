#!/bin/bash
# Robustness check on the 9 nm -> 18 nm DOWNSAMPLE, at the measured optimum.
#
# The headline mip0 number (mt 0.47 -> VOI 0.9701 vs the 18 nm baseline's 0.9130)
# is taken by point-sampling the 9 nm segmentation onto the 18 nm grid at phase 0,
# which is the exact inverse of the GT's np.repeat upsample. But point-sampling can
# drop a predicted fragment that happens to be thin at 9 nm, and a dropped fragment
# reads as a split. If that artifact were large it would inflate the gap.
#
# So: score the SAME decode at phase 0 and phase 1 (opposite corners of every
# 2x2x2 block). Both are equally valid samplings of the same segmentation onto
# the same grid, so their spread bounds how much the downsample itself contributes.
# Spread << 0.057 means the gap is not a downsampling artifact.
#
# One threshold only (0.47, the optimum), so this is ~1 watershed, not a sweep.
#SBATCH --job-name=mip0_phase
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=1500G
#SBATCH --exclude=gb001
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_mip0_phase_check.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_mip0_phase_check.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"
AFF="$REPO/outputs/liconn_final_banis_plus_mip0/20260908_164220/test_step=00200000/val/raw_x1_ch0-1-2.h5"

python tutorials/neuron_liconn_ist/mip0_wholeval_sweep.py \
  --affinity "$AFF" \
  --label mip0phase \
  --thresholds 0.47 \
  --ws-high 94% --ws-low 20% \
  --factor 2 --phases 0,1 \
  --ws-binary "$REPO/lib/abiss/build64/ws64" \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss_work/mip0phase" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/mip0_phase_check.json"

echo "--- phase 0 at mt 0.47 was 0.9701 in the sweep (job 3028274); the 18 nm"
echo "--- baseline at the same settings is 0.9130 (eb2) / 0.9188 (eb8)."
echo "DONE end=$(date -Is)"
