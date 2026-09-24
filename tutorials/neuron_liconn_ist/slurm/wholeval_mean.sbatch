#!/bin/bash
# Whole-volume confirmation of `mean`, before it replaces `max` in 2_abiss.yaml.
#
# WHY mean AT ALL. Donglai, 2026-09-17: the val GT contains errors of its own, so
# sub-0.05 VOI differences at the operating point are not worth optimising, and
# the decode should instead leave the most room for later error correction to
# recover. That makes the oracle ceiling the objective, not VOI -- and on cubes
# `mean` beats `max` on the ceiling by 6.5% (eb2) and 12.8% (eb8) while losing
# 0.040 / 0.011 VOI at the operating point.
#
# Cube numbers are crop-biased and comparable only across settings, so this is the
# whole-volume confirmation the card requires before any adoption.
#
# GRID. Cube optima were mt 0.10 (VOI) and 0.30/0.18 (ceiling) in PROBABILITY
# space. Crops were measured to misplace the optimum LOW by 1-2 grid steps, so
# the grid extends upward rather than bracketing the cube values symmetrically.
#
# `mean` is not monotone-invariant, so --uncompress is mandatory; the script
# refuses without it. Seeding stays at the adopted 94% / 0.0, and percentile
# seeding is itself invariant, so it is identical in both spaces.
#SBATCH --job-name=wval_mean
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%A_%a_wval_mean.out
# Submit from the repo root: --output above and REPO below are both
# relative to the submit directory.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${PYTC_CONDA:?export PYTC_CONDA to your miniconda prefix (no conda shell hook on the login node)}"
source "$PYTC_CONDA/bin/activate" "${PYTC_ENV:-pytc}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
EB2="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
EB8="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/test_step=00200000/val/raw_x1_ch0-1-2.h5"
case "$SLURM_ARRAY_TASK_ID" in
  0) AFF="$EB2"; L=eb2 ;;
  1) AFF="$EB8"; L=eb8 ;;
esac
echo "host=$(hostname) label=$L start=$(date -Is)"; free -g | head -2
python tutorials/neuron_liconn_ist/wholeval_sweep.py --affinity "$AFF" --label "${L}_mean" \
  --ws-high "94%" --ws-low 0.0 --merge-function mean --uncompress \
  --thresholds 0.10,0.18,0.30,0.45,0.60 --oracle \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_wholeval_mean.json"
echo "DONE end=$(date -Is)"
