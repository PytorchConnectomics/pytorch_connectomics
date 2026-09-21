#!/bin/bash -l
set -euo pipefail

ROOT=""; PARTITION=""; GPU_PARTITION=""; ACCOUNT=""; LAUNCHER="slurm"
BBOX=""; TRAIN="true"; PASSTHRU=()
ABISS_REPO="https://github.com/PytorchConnectomics/ABISS.git"
REPO_URL="${J0126_REPO_URL:-https://github.com/PytorchConnectomics/pytorch_connectomics.git}"
REPO_REF="${J0126_REPO_REF:-master}"
ABISS_COMMIT="452efa5f87f9d3cb241891ee44010d966a33b316"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --gpu-partition) GPU_PARTITION="$2"; shift 2;;
    --account) ACCOUNT="$2"; shift 2;;
    --launcher) LAUNCHER="$2"; shift 2;;
    --bbox) BBOX="$2"; shift 2;;
    --no-train) TRAIN="false"; shift;;
    --check|--dry-run) PASSTHRU+=("$1"); shift;;
    *) echo "unknown option: $1"; exit 2;;
  esac
done
[ -n "$ROOT" ] || { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
[ -n "$GPU_PARTITION" ] || GPU_PARTITION="$PARTITION"

say() { echo "[$(date +%H:%M:%S)] $*"; }
ENVDIR="$ROOT/env"; DATA="$ROOT/data"; OUT="$ROOT/out"
mkdir -p "$ROOT" "$DATA" "$OUT"

if [ ! -x "$ENVDIR/bin/python" ]; then
  MAMBA="$ROOT/micromamba"
  if [ ! -x "$MAMBA" ]; then
    say "fetching micromamba"
    curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$ROOT" --strip-components=1 bin/micromamba
  fi
  say "creating the environment (python 3.11, ~10 min)"
  "$MAMBA" create -y -p "$ENVDIR" -c conda-forge \
    python=3.12 cmake make cxx-compiler boost=1.82 tbb tbb-devel zlib parallel
fi
PY="$ENVDIR/bin/python"
export PATH="$ENVDIR/bin:$PATH"

if [ -f "$HERE/../../connectomics/runtime/abiss_chunk.py" ]; then
  REPO="$(cd "$HERE/../.." && pwd)"
  say "using the checkout this script lives in: $REPO"
else
  REPO="$ROOT/pytorch_connectomics"
  if [ ! -d "$REPO/.git" ]; then
    say "cloning $REPO_URL ($REPO_REF)"
    git clone --quiet --branch "$REPO_REF" "$REPO_URL" "$REPO"
  fi
fi

if ! "$PY" -c "import connectomics, omegaconf, em_erl, tinybrain, chunkiterator" 2>/dev/null; then
  say "installing python dependencies (~10 min)"
  (cd "$REPO" && "$PY" -m pip install -q -e .)
  export GIT_TERMINAL_PROMPT=0
  "$PY" -m pip install -q lightning cloud-volume tensorstore kimimaro zarr h5py tinybrain \
    "git+https://github.com/PytorchConnectomics/MedNeXt.git" \
    "git+https://github.com/seung-lab/chunk_iterator#egg=chunk-iterator" \
    "git+https://github.com/PytorchConnectomics/em_erl.git"
fi

if [ ! -x "$REPO/lib/abiss/build/acme" ]; then
  say "building ABISS at $ABISS_COMMIT (~10 min)"
  mkdir -p "$REPO/lib"
  [ -d "$REPO/lib/abiss/.git" ] || git clone --quiet "$ABISS_REPO" "$REPO/lib/abiss"
  git -C "$REPO/lib/abiss" checkout --quiet "$ABISS_COMMIT"
  cmake -S "$REPO/lib/abiss" -B "$REPO/lib/abiss/build" -DCMAKE_BUILD_TYPE=Release \
    -DBOOST_ROOT="$ENVDIR" -DEXTRACT_SIZE=ON -DBUILD_TESTING=ON >/dev/null
  cmake --build "$REPO/lib/abiss/build" --parallel 8 >/dev/null
  ctest --test-dir "$REPO/lib/abiss/build" --output-on-failure >/dev/null
fi

say "writing $REPO/tutorials/neuron_j0126/params.yaml"
"$PY" - "$REPO" "$DATA" "$OUT" "$LAUNCHER" "$ACCOUNT" "$PARTITION" "$GPU_PARTITION" \
      "$TRAIN" "$BBOX" <<'PYEOF'
import sys, re
from pathlib import Path
repo, data, out, launcher, account, part, gpu_part, train, bbox = sys.argv[1:10]
p = Path(repo) / "tutorials" / "neuron_j0126" / "params.yaml"
s = p.read_text()
s = re.sub(r"^(    repository: ).*$", r"\1" + repo, s, count=1, flags=re.M)
s = re.sub(r"^(    dataset_root: ).*$", r"\1" + data, s, count=1, flags=re.M)
s = re.sub(r"^(    output_root: ).*$", r"\1" + out, s, count=1, flags=re.M)
s = re.sub(r"^(    launcher: ).*$", r"\1" + launcher, s, count=1, flags=re.M)
s = re.sub(r"^(    account: ).*$", r'\1"%s"' % account, s, count=1, flags=re.M)
s = re.sub(r"^(    em_bbox: ).*$", r"\1[%s]" % ", ".join(bbox.split()), s, count=1, flags=re.M)
lines = s.splitlines(True)
section = None
for i, line in enumerate(lines):
    m = re.match(r"^  (\w+):\s*$", line)
    if m:
        section = m.group(1)
    if line.startswith("    slurm_partition:"):
        chosen = gpu_part if section in ("train", "inference") else part
        lines[i] = '    slurm_partition: "%s"\n' % chosen
    if section == "train" and line.startswith("    enabled:"):
        lines[i] = "    enabled: %s\n" % train
p.write_text("".join(lines))
print("  repository  ", repo)
print("  dataset_root", data)
print("  output_root ", out)
print("  train       ", train)
print("  em_bbox     ", bbox or "[] (whole volume)")
PYEOF

say "running the pipeline"
cd "$REPO"
exec "$PY" scripts/run_j0126.py ${PASSTHRU[@]+"${PASSTHRU[@]}"}
