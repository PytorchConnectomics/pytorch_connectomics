#!/bin/bash
# mip0 step 2: ABISS watershed + GT-free merge-threshold sweep on a mip0 volume.
#
# Sizing is the reason this is not the mip1 sweep file. Native 18x at mip0 is
# 1.7-6.4 Gvox, against 0.4-1.6 at mip1, and TWO separate limits bite there.
#
# 1. THE uint32 WATERSHED CAP, which no --mem can move. The stock
#    lib/abiss/build/ws cannot index more than 2^31 voxels in one invocation.
#    Three of the four mip0 volumes are over it (249% and 297% for the two
#    cerebellum volumes, 145% for ExPID96_2ndgel_S1_40XW001_18x); only
#    ExPID96_S1_40XW002_18x (78.7%) fits, which is exactly why it is the only
#    one the 2026-09-16 array produced. sweep_merge_threshold.py now measures
#    the padded chunk and selects lib/abiss/build64/ws64 when it must; nothing
#    needs setting here. See its WS_UINT32_CAP block.
#
# 2. MEMORY. ws64 peaks at 57.9 GB/Gvox (measured, IST job 3028181) -> ~310 GB
#    at 5.3 Gvox, and that is now the whole story. It was not: at 480G this
#    OOMed, because run_abiss_volume.py carried ~160 GB of python arrays across
#    the watershed AND then accumulated every threshold's segmentation before
#    writing any (8 x 5.3 Gvox x 8 B = 341 GB), for a ~520 GB peak in which the
#    watershed was the SMALLER term. Three frees fixed it -- the percentile
#    affinity copy, the XYZC copy across the subprocess, and streaming each
#    segmentation out as it is read (`on_batch_result`) -- verified
#    bit-identical to the pre-change output on all three thresholds of a
#    96x384x384 crop of this volume's own affinity.
#    Peak is now ~342 GB at 5.3 Gvox and ~400 GB at 6.4, and it no longer grows
#    with the size of the threshold grid. 480G is the ask again, and this time
#    it holds.
#
# NODE CLASS. cb001-cb020 are 2 TB / 96-core CPU nodes in `short` and are what
# this should land on. gb001 (the shared h200) is excluded to leave it for work
# that actually needs a GPU. Do NOT reach for run_abiss_chunk.py instead: it
# agglomerates with mean and cannot do max, so it would change the decoder and
# the resolution at once.
#SBATCH --job-name=mip0_sweep
#SBATCH --partition=short
# No --gres: this is CPU work.
#SBATCH --cpus-per-task=16
#SBATCH --mem=480G
#SBATCH --exclude=gb001
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_mip0_2_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_mip0_2_sweep.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

export MOE_GRID=mip0
export MOE_CKPT="$REPO/outputs/liconn_final_banis_plus_mip0/20260908_164220/checkpoints/step=00200000.ckpt"
export MOE_OUT_ROOT="$REPO/outputs/neuron_liconn_moe/mip0_eb8"
export MOE_GCS_KIND=mip0_eb8

VOLS=(${MIP0_VOLUMES:-ExPID96_2ndgel_S1_40XW001_18x ExPID96_S1_40XW002_18x ExPID99_18x_2_cerebellum_1Byqvupl ExPID99_18x_2_cerebellum_1m2f9z4J})
VOL="${VOLS[$SLURM_ARRAY_TASK_ID]}"
[ -z "$VOL" ] && { echo "no volume at index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
free -g | head -2
python tutorials/neuron_liconn_moe/sweep_merge_threshold.py --volume "$VOL"
echo "DONE end=$(date -Is)"
