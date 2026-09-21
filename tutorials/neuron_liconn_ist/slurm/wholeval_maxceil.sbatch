#!/bin/bash
# max's whole-volume ORACLE CEILING, extended past the edge it stopped at.
#
# WHY THIS EXISTS. The mean-vs-max ceiling comparison cannot be drawn without it.
# max's whole-volume ceiling was measured only at mt 0.44/0.47/0.50 (job 3016173)
# and was STILL FALLING at 0.50, the top of that grid: eb2 0.4057 -> 0.3016 ->
# 0.2829. Comparing mean's interior ceiling optimum against a max value pinned at
# a range edge is exactly the error this card was opened to stop making. On cubes
# max's ceiling turned at 0.50, but crops were measured to misplace optima LOW
# (R5), so the whole-volume turn may sit higher and max's true ceiling lower.
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
#SBATCH --job-name=wval_maxceil
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_logs/neuron_liconn_ist/%A_%a_wval_maxceil.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
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
python tutorials/neuron_liconn_ist/wholeval_sweep.py --affinity "$AFF" --label "${L}_maxceil" \
  --ws-high "94%" --ws-low 0.0 --merge-function max \
  --thresholds 0.50,0.53,0.56,0.59 --oracle \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_wholeval_maxceil.json"
echo "DONE end=$(date -Is)"
