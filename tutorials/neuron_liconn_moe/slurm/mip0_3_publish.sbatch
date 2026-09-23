#!/bin/bash
# mip0 step 3: build the precomputed layer + meshes for one mip0 segmentation,
# then upload it to <family>/mip0_eb8/ and verify object count + bytes.
#
# One job rather than eb8's build/upload split: compute nodes reach GCS with the
# stored credential (verified 2026-09-16, see eb8_4_upload.sbatch). Memory is
# sized for the 18x volumes at mip0 (up to 6.4 Gvox; _read_seg holds the whole
# label volume, 8 B/vox before the uint32 cast).
#
# The gcloud parallelism cap is load-bearing: without it rsync workers can die
# and the parent hang forever with a partial layer (memory gcloud_rsync_ulimit).
#
#   sbatch --array=0-2 --export=ALL,MIP0_VOLUMES="volA volB volC" mip0_3_publish.sbatch
#SBATCH --job-name=mip0_publish
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --exclude=gb001
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_mip0_3_publish.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_mip0_3_publish.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export CLOUDSDK_STORAGE_PROCESS_COUNT=1 CLOUDSDK_STORAGE_THREAD_COUNT=8
cd "$REPO"

export MOE_GRID=mip0
export MOE_CKPT="$REPO/outputs/liconn_final_banis_plus_mip0/20260908_164220/checkpoints/step=00200000.ckpt"
export MOE_OUT_ROOT="$REPO/outputs/neuron_liconn_moe/mip0_eb8"
export MOE_GCS_KIND=mip0_eb8

VOLS=(${MIP0_VOLUMES:-ExPID96_2ndgel_S1_40XW001_18x ExPID96_S1_40XW002_18x ExPID99_18x_2_cerebellum_1Byqvupl ExPID99_18x_2_cerebellum_1m2f9z4J})
VOL="${VOLS[$SLURM_ARRAY_TASK_ID]}"
[ -z "$VOL" ] && { echo "no volume at index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
python - "$VOL" <<'PY'
import sys
sys.path.insert(0, "tutorials/neuron_liconn_moe")
import volumes as V
print("spacing_zyx ->", V.plan(sys.argv[1])["spacing_zyx"])
print("gcs dest    ->", V.gcs_folder(sys.argv[1]))
PY

python tutorials/neuron_liconn_moe/upload_seg_precomputed.py \
  --volume "$VOL" --create --downsample --mesh --parallel 16
python tutorials/neuron_liconn_moe/upload_seg_precomputed.py --volume "$VOL" --upload --verify
echo "DONE end=$(date -Is)"
