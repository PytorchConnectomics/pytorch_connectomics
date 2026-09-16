#!/bin/bash
# Step 1: apply the IST-LICONN banis+ 200k checkpoint to one prepared moe volume.
# Array index = position in tutorials/neuron_liconn_moe/volumes.py::PENDING.
#SBATCH --job-name=moe_aff
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_1_affinity.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_1_affinity.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from the repo root
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

VOL=$(python -c "import sys;sys.path.insert(0,'tutorials/neuron_liconn_moe');import volumes as V;print(V.PENDING[$SLURM_ARRAY_TASK_ID])")
CKPT=$(python -c "import sys;sys.path.insert(0,'tutorials/neuron_liconn_moe');import volumes as V;print(V.CKPT)")
CFG=$(python tutorials/neuron_liconn_moe/make_volume_config.py --volume "$VOL")

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/main.py --config "$CFG" --mode test --checkpoint "$CKPT"

python - "$VOL" <<'PY'
import sys, os
sys.path.insert(0, "tutorials/neuron_liconn_moe")
import h5py, numpy as np, volumes as V
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
p = V.affinity_h5(sys.argv[1])
with h5py.File(p, "r") as f:
    d = f["main"]
    print(f"affinity {p} {d.shape} {d.dtype} {os.path.getsize(p)/1e9:.2f} GB")
    s = np.asarray(d[:, d.shape[1] // 2]).astype(np.float32)
q = np.percentile(s, [25, 50, 75])
print(f"mid-plane affinity p25/p50/p75 = {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  (IST val 0.26/0.53/0.68)")
assert s.std() > 0.01, "affinity mid-plane is constant"
PY
echo "DONE end=$(date -Is)"
