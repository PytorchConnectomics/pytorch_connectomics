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

# The frozen reference was built with the conda compiler, Boost 1.82, and
# oneTBB (libtbb.so.12). Merely setting CMAKE_PREFIX_PATH is not sufficient:
# CMake may otherwise select the host compiler, Boost 1.85, and legacy
# libtbb.so.2, which produces a binary that fails or stalls during mean-edge
# agglomeration.
cc="$PYTC_PREFIX/bin/x86_64-conda-linux-gnu-cc"
cxx="$PYTC_PREFIX/bin/x86_64-conda-linux-gnu-c++"
boost_dir="$PYTC_PREFIX/lib/cmake/Boost-1.82.0"
tbb_dir="$PYTC_PREFIX/lib/cmake/TBB"
for required in "$cc" "$cxx" "$boost_dir/BoostConfig.cmake" "$tbb_dir/TBBConfig.cmake"; do
  if [ ! -e "$required" ]; then
    echo "Missing frozen-toolchain dependency: $required" >&2
    exit 2
  fi
done

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
  -DCMAKE_C_COMPILER="$cc" \
  -DCMAKE_CXX_COMPILER="$cxx" \
  -DBOOST_ROOT="$PYTC_PREFIX" \
  -DBoost_DIR="$boost_dir" \
  -DTBB_DIR="$tbb_dir" \
  -DEXTRACT_SIZE=ON \
  -DBUILD_TESTING=ON
cmake --build "$ABISS_HOME/build" --parallel "${SLURM_CPUS_PER_TASK:-8}"
ctest --test-dir "$ABISS_HOME/build" --output-on-failure

cache="$ABISS_HOME/build/CMakeCache.txt"
require_cache_value() {
  local key=$1 expected_value=$2
  if ! awk -v key="$key" -v expected="$expected_value" '
    index($0, key ":") == 1 {
      separator = index($0, "=")
      if (separator && substr($0, separator + 1) == expected) found = 1
    }
    END { exit(found ? 0 : 1) }
  ' "$cache"; then
    echo "CMake cache does not pin ${key}=${expected_value}" >&2
    grep -E "^${key}:" "$cache" >&2 || true
    exit 3
  fi
}
require_cache_value CMAKE_C_COMPILER "$cc"
require_cache_value CMAKE_CXX_COMPILER "$cxx"
require_cache_value Boost_DIR "$boost_dir"
require_cache_value TBB_DIR "$tbb_dir"
require_cache_value EXTRACT_SIZE ON

for binary in ws agg; do
  dynamic=$(readelf -d "$ABISS_HOME/build/$binary")
  if ! grep -Fq 'Shared library: [libtbb.so.12]' <<<"$dynamic"; then
    echo "$binary is not linked to the frozen oneTBB soname libtbb.so.12" >&2
    exit 4
  fi
  if grep -Eq 'libtbb\.so\.2]|libboost_.*\.so\.1\.85\.0]' <<<"$dynamic"; then
    echo "$binary picked up an incompatible system/Boost runtime" >&2
    grep 'Shared library:' <<<"$dynamic" >&2
    exit 4
  fi
done

git -C "$ABISS_HOME" status --short --branch
sha256sum "$ABISS_HOME/build/ws" "$ABISS_HOME/build/agg"
"$cxx" --version | head -1
grep -E '^(CMAKE_CXX_COMPILER|Boost_DIR|TBB_DIR):' "$cache"
readelf -d "$ABISS_HOME/build/agg" | grep -E 'NEEDED|RPATH|RUNPATH'
echo "ABISS build complete: $ABISS_HOME/build"
