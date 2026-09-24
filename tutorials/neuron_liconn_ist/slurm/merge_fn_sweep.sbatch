#!/bin/bash
# Merge-criterion experiment: does `mean` or `p75` beat `max`, and is undoing the
# scale_sigmoid a precondition or an improvement?
#
# WHY NOW. Threshold tuning is exhausted on this volume: mt is already optimal at
# 0.47, ws_high is inert, ws_low bought 0.003 VOI. But VOI ~0.91 against an
# oracle-merge ceiling of ~0.30 says ~67% of the remaining error is false splits
# that better agglomeration of the SAME fragments would fix. That is a criterion
# problem, not a threshold problem. `max` merges on the single strongest voxel of
# a contact surface; `mean` and `p75` weigh the whole surface.
#
# THE CONTROL IS THE POINT OF TASKS 0/1 AND 4/5. `max` is monotone-invariant, so
# max-on-compressed and max-on-uncompressed with thresholds mapped through
# p = sigmoid(logit(v)/0.2) MUST agree row for row. If they do not, the
# uncompression is wrong and the mean/p75 rows mean nothing. Mapped grid:
#   v 0.44 -> p 0.230443     v 0.47 -> p 0.354179     v 0.50 -> p 0.500000
#
# `mean` and `p75` are NOT monotone-invariant, so they are only run uncompressed:
# scoring them on sigmoid(0.2*logit(p)) would measure an arbitrary criterion.
# Their grids are deliberately wide -- a mean over a contact surface sits well
# below its max, so the optimum is expected far lower, and the range is unknown.
#
# Seeding is the adopted 94% / 0.0 everywhere. Percentile seeding is itself
# monotone-invariant, so it is identical in both spaces and the merge function is
# the only thing varying.
#SBATCH --job-name=merge_fn
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%A_%a_merge_fn.out
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
MAXC="0.44,0.47,0.50"
MAXU="0.230443,0.354179,0.500000"
MEANU="0.002,0.005,0.01,0.02,0.05,0.10,0.18,0.30,0.45,0.60"
P75U="0.005,0.02,0.05,0.10,0.18,0.30,0.45,0.60,0.75,0.88"
case "$SLURM_ARRAY_TASK_ID" in
  0) AFF="$EB2"; L=eb2; FN=max;  TH="$MAXC";  U="";             SLUG=max_compressed ;;
  1) AFF="$EB2"; L=eb2; FN=max;  TH="$MAXU";  U="--uncompress"; SLUG=max_prob ;;
  2) AFF="$EB2"; L=eb2; FN=mean; TH="$MEANU"; U="--uncompress"; SLUG=mean_prob ;;
  3) AFF="$EB2"; L=eb2; FN=p75;  TH="$P75U";  U="--uncompress"; SLUG=p75_prob ;;
  4) AFF="$EB8"; L=eb8; FN=max;  TH="$MAXC";  U="";             SLUG=max_compressed ;;
  5) AFF="$EB8"; L=eb8; FN=max;  TH="$MAXU";  U="--uncompress"; SLUG=max_prob ;;
  6) AFF="$EB8"; L=eb8; FN=mean; TH="$MEANU"; U="--uncompress"; SLUG=mean_prob ;;
  7) AFF="$EB8"; L=eb8; FN=p75;  TH="$P75U";  U="--uncompress"; SLUG=p75_prob ;;
esac
echo "host=$(hostname) label=$L fn=$FN slug=$SLUG start=$(date -Is)"
python tutorials/neuron_liconn_ist/merge_fn_sweep.py --affinity "$AFF" --label "${L}_${SLUG}" \
  --merge-function "$FN" --merge-thresholds "$TH" $U \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_mergefn_${SLUG}.json"
echo "DONE end=$(date -Is)"
