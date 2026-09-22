#!/bin/bash
# Per-model merge-threshold sweep on IST LICONN val slabs, eb2 and eb8 on the
# SAME slabs and the same threshold grid.
#
# WHY BOTH, not just eb8. The 0.47 both models were scored at was tuned on eb2,
# so comparing them there favours eb2 by construction. Sweeping each model on
# identical slabs gives each its own optimum, which is the comparison the card
# asks for. Slab VOI is biased high on splits (whole-val is 0.176 worse, nearly
# all split), but that bias applies to both models, so it does not distort the
# *comparison* -- it does mean neither number may be quoted as a score.
#
# The grid runs BELOW eb2's previous 0.47 optimum on purpose: slab tuning biases
# high, and eb8 splits more than eb2 at a matched threshold, so its optimum
# should sit lower still. The script warns if the best value lands on an end of
# the range.
#SBATCH --job-name=ist_mt_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%A_%a_mt_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%A_%a_mt_sweep.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

EB2="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
EB8="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/test_step=00200000/val/raw_x1_ch0-1-2.h5"
OUT="$REPO/outputs/neuron_liconn_ist/mt_sweep"

case "$SLURM_ARRAY_TASK_ID" in
  0) AFF="$EB2"; LABEL=eb2 ;;
  1) AFF="$EB8"; LABEL=eb8 ;;
  *) echo "bad array index" >&2; exit 2 ;;
esac

echo "host=$(hostname) label=$LABEL start=$(date -Is)"
python tutorials/neuron_liconn_ist/sweep_merge_threshold.py \
  --affinity "$AFF" --label "$LABEL" \
  --slabs 3 \
  --merge-thresholds 0.32,0.35,0.38,0.41,0.44,0.47,0.50,0.53,0.56 \
  --json "$OUT/${LABEL}_slab_sweep.json"
echo "DONE end=$(date -Is)"
