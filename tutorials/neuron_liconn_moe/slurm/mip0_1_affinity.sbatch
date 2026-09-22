#!/bin/bash
# mip0 step 1: apply the 4-GPU mip0 checkpoint to one natively-staged 18x volume.
#
# Everything routes off three env vars (see volumes.py): MOE_GRID picks the
# prepared tree and plan(); MOE_CKPT picks the checkpoint AND, through
# TEST_OUT, where the affinity lands; MOE_OUT_ROOT scopes the working tree and
# the GCS kind. make_volume_config.py then writes the per-volume override with
# the volume's ACTUAL spacing [22.22, 9.03, 9.03] -- not the model's nominal
# [12,9,9]. That matters twice: the model is applied at the scale the data is
# really at, and the published layer overlays the native image group.
#
# The mip0 checkpoint trained with erosion=4 (XY-only erosion on a 2x-upsampled
# label, reproducing the mip1 baseline's erosion=2 separation). Nothing here
# changes that; it is a property of the checkpoint.
#SBATCH --job-name=mip0_aff
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_mip0_1_affinity.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_mip0_1_affinity.out

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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CFG=$(python tutorials/neuron_liconn_moe/make_volume_config.py --volume "$VOL")
echo "config: $CFG"; cat "$CFG"

python scripts/main.py --config "$CFG" --mode test --checkpoint "$MOE_CKPT"

python - "$VOL" <<'PY'
import os, sys
sys.path.insert(0, "tutorials/neuron_liconn_moe")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py, numpy as np, volumes as V
p = V.affinity_h5(sys.argv[1])
with h5py.File(p, "r") as f:
    d = f["main"]
    print(f"affinity {p} {d.shape} {d.dtype} {os.path.getsize(p)/1e9:.2f} GB")
    s = np.asarray(d[:, d.shape[1] // 2]).astype(np.float32)
q = np.percentile(s, [25, 50, 75])
print(f"mid-plane p25/p50/p75 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  (mip1 in-domain ref 0.26/0.53/0.68)")
assert s.std() > 0.01, "affinity mid-plane is constant -- do NOT decode this"
PY
echo "DONE end=$(date -Is)"
