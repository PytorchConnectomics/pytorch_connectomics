#!/bin/bash
# Whole-volume confirmation of the cube-derived ws_low finding.
#
# WHAT THIS DECIDES. The round-2 cube grid showed ws_high is INERT over 79%..1.0
# (bit-identical numbers) and that the only seeding lever is ws_low, worth ~1.3-1.5%
# on the oracle ceiling and ~0.002 VOI -- the latter well inside the 0.01 noise
# floor. Cube numbers are crop-biased and comparable only across settings, so the
# ws_low change cannot be adopted into 2_abiss.yaml on cube evidence.
#
# This runs the incumbent ws_low 20% against ws_low 0.0 on the WHOLE volume, both
# models, with --oracle so the ceiling (where the effect actually lives) is
# visible. ws_high stays at the incumbent 94% because it has no measured effect.
# mt is re-swept 0.44/0.47/0.50 rather than carried over, since different seeding
# gives different fragments and therefore potentially a different merge optimum.
#
# Memory: 3 thresholds, freed as scored, plus ~50 GB transient for the oracle
# key/sort at 2.08 Gvox. The 5-threshold run without oracle peaked at 208 GB.
#SBATCH --job-name=wval_wslow
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_logs/neuron_liconn_ist/%A_%a_wval_wslow.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
EB2="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
EB8="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/test_step=00200000/val/raw_x1_ch0-1-2.h5"
case "$SLURM_ARRAY_TASK_ID" in
  0) AFF="$EB2"; L=eb2; LOW="20%"; SLUG=low20 ;;
  1) AFF="$EB2"; L=eb2; LOW="0.0";  SLUG=low00 ;;
  2) AFF="$EB8"; L=eb8; LOW="20%"; SLUG=low20 ;;
  3) AFF="$EB8"; L=eb8; LOW="0.0";  SLUG=low00 ;;
esac
echo "host=$(hostname) label=$L ws_low=$LOW start=$(date -Is)"; free -g | head -2
python tutorials/neuron_liconn_ist/wholeval_sweep.py --affinity "$AFF" --label "${L}_${SLUG}" \
  --ws-high "94%" --ws-low "$LOW" --thresholds 0.44,0.47,0.50 --oracle \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_wholeval_${SLUG}.json"
echo "DONE end=$(date -Is)"
