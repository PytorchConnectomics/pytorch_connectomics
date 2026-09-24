#!/bin/bash
# Does the TRADITIONAL ABISS seeding (high ~0.999 / low ~0.001 -- force nothing,
# discard nothing) give a better oracle ceiling than this pipeline*s 94%/20%?
#
# Reading lib/abiss/src/ws/basic_watershed.hpp:76-92, `low` is a DISCARD
# threshold (a voxel with max affinity <= low gets no connection) and `high` is a
# FORCE-CONNECT threshold (>= high adds unconditional connections on top of the
# steepest one). So low should be as low as possible, and high as HIGH as
# possible, for the best ceiling -- the opposite of the direction the round-2
# grid searched. This 2x2 tests that directly and separates the two parameters.
#
#   high=1.0  -- above the data max (eb2 0.823, eb8 0.885): no forced connections
#   low=0.0   -- discard nothing
#   94% / 20% -- the incumbent, kept in the grid so the run can say if it lost
#
# Absolute (non-%) values pass through _resolve_threshold unchanged.
#SBATCH --job-name=seed_extremes
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%A_%a_seed_extremes.out
# Submit from the repo root: --output above and REPO below are both
# relative to the submit directory.
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
: "${PYTC_CONDA:?export PYTC_CONDA to your miniconda prefix (no conda shell hook on the login node)}"
source "$PYTC_CONDA/bin/activate" "${PYTC_ENV:-pytc}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
case "$SLURM_ARRAY_TASK_ID" in
  0) AFF="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"; L=eb2 ;;
  1) AFF="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/test_step=00200000/val/raw_x1_ch0-1-2.h5"; L=eb8 ;;
esac
echo "host=$(hostname) label=$L start=$(date -Is)"
python tutorials/neuron_liconn_ist/cube_sweep.py --affinity "$AFF" --label "$L" \
  --ws-high 1.0,94% --ws-low 0.0,20% \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${L}_cube_seed_extremes.json"
echo "DONE end=$(date -Is)"
