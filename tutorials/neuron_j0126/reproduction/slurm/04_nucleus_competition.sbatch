#!/bin/bash
#SBATCH --job-name=j0126-nucleus
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${RUNTIME_JSON:?}"
: "${PYTHON:?}"
: "${ABISS_LIBRARY_PREFIX:?}"

export PATH="$(dirname "$PYTHON"):$PATH"
export LD_LIBRARY_PATH="$ABISS_LIBRARY_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_USE_FILE_LOCKING=FALSE
abiss_home=$("$PYTHON" -c \
  "import json; print(json.load(open('$RUNTIME_JSON'))['abiss_home'])")
param=$("$PYTHON" -c \
  "import json; print(json.load(open('$RUNTIME_JSON'))['param'])")
cloud_volume_dir=$("$PYTHON" -c \
  "import json; print(json.load(open('$RUNTIME_JSON'))['cloud_volume_dir'])")
if [ -n "$cloud_volume_dir" ]; then
  export CLOUD_VOLUME_DIR="$cloud_volume_dir"
fi
export PARAM_JSON="$param"

"$PYTHON" -u "$abiss_home/scripts/nucleus_competition.py" "$param"
manifest=$("$PYTHON" -c \
  "import json; print(json.load(open('$param'))['NUC_COMPETITION_MANIFEST'])")
"$PYTHON" -c \
  "import json; p=json.load(open('$manifest')); assert p['manifest_type']=='abiss_nucleus_competition'; print(f\"verified {len(p['repairs'])} nucleus repairs\")"
