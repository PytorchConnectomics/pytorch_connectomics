#!/bin/bash
#SBATCH --job-name=j0126-build
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${ABISS_HOME:?Set ABISS_HOME to the pinned ABISS checkout}"
: "${PYTC_PREFIX:?Set PYTC_PREFIX to the pytc conda environment}"

# Compute nodes expose compiler wrappers through ccache, while their inherited
# XDG runtime directory can be owned by another node. Keep all ccache writes in
# the writeable checkout so configuration does not fail before compilation.
export CCACHE_DIR="$ABISS_HOME/.ccache"
export CCACHE_TEMPDIR="$ABISS_HOME/.ccache/tmp"
export CMAKE_PREFIX_PATH="$PYTC_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$PYTC_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$CCACHE_TEMPDIR"

head=$(git -C "$ABISS_HOME" rev-parse HEAD)
expected=452efa5f87f9d3cb241891ee44010d966a33b316
if [ "$head" != "$expected" ]; then
  echo "ABISS commit $head does not match frozen commit $expected" >&2
  exit 2
fi

cmake -S "$ABISS_HOME" -B "$ABISS_HOME/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="$PYTC_PREFIX" \
  -DBUILD_TESTING=ON
cmake --build "$ABISS_HOME/build" --parallel "${SLURM_CPUS_PER_TASK:-8}"
ctest --test-dir "$ABISS_HOME/build" --output-on-failure

git -C "$ABISS_HOME" status --short --branch
echo "ABISS build complete: $ABISS_HOME/build"
