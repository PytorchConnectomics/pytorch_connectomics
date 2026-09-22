#!/bin/bash
# 600 nm gate steps 3-4: decode the degraded-val affinity with ABISS, sweep the
# merge threshold, and score on BOTH grids. Card MSIDEPLOY-MODEL-002.
#
# 0.47 is NOT inherited. The affinity distribution shifts with the input -- the
# smoke crop came back p25/p50/p75 0.250/0.512/0.669 against the 18 nm reference
# 0.26/0.53/0.68 -- so the operating point is not transferable, and the baseline's
# own published optimum sat at the LOW edge of its 0.47-0.59 sweep. The grid here
# brackets it on both sides. ws_high/ws_low stay at the baseline's 94%/20% so the
# comparison is not confounded with MSIDEPLOY-JOB-001's ws_low study.
#
# Memory: 1.49 Gvox. The affinity is 17.9 GB as float32 and `ws` peaks around
# 126 GB (176 GB measured at 2.08 Gvox, scaled). Segmentations are consumed
# through the on_batch_result sink, so only ONE (11.9 GB) is held rather than six.
# With the GT (16.6 GB), its 104-plane copy (11.9 GB) and the scoring transients
# the peak is ~250 GB; 480G is what the equivalent baseline sweep ran at.
#SBATCH --job-name=z600_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_3_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_3_sweep.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
AFF="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val_z600/raw_x1_ch0-1-2.h5"
echo "host=$(hostname) start=$(date -Is)"; free -g | head -2
python tutorials/neuron_liconn_ist/z600_wholeval_sweep.py \
  --affinity "$AFF" --label z600 --source-grid z600 \
  --thresholds 0.38,0.41,0.44,0.47,0.50,0.53 \
  --workdir "$REPO/outputs/neuron_liconn_ist/abiss/ws_scratch_z600" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/z600_wholeval_sweep.json"
echo "DONE end=$(date -Is)"
