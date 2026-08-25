#!/bin/bash
#SBATCH --job-name=j0126-agg-check
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
: "${SOURCE_CHUNK:?Set SOURCE_CHUNK to a failed atomic mean-edge work directory}"
: "${VALIDATION_DIR:?Set VALIDATION_DIR to a new output directory}"
: "${ABISS_HOME:?Set ABISS_HOME to the rebuilt pinned checkout}"
: "${ABISS_LIBRARY_PREFIX:?Set ABISS_LIBRARY_PREFIX to the pinned conda environment}"

if [ -e "$VALIDATION_DIR" ]; then
  echo "Refusing to reuse validation directory: $VALIDATION_DIR" >&2
  exit 2
fi
mkdir -p "$VALIDATION_DIR"

for name in aff.raw seg.raw chunkmap.data param.txt; do
  if [ ! -f "$SOURCE_CHUNK/$name" ]; then
    echo "Missing validation input: $SOURCE_CHUNK/$name" >&2
    exit 2
  fi
  cp --reflink=auto "$SOURCE_CHUNK/$name" "$VALIDATION_DIR/$name"
done

export LD_LIBRARY_PATH="$ABISS_LIBRARY_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1
cd "$VALIDATION_DIR"
mkdir chunked_rg

sha256sum "$ABISS_HOME/build/agg" > binary.sha256
readelf -d "$ABISS_HOME/build/agg" | grep -E 'NEEDED|RPATH|RUNPATH' > binary.dynamic.txt
chunk_tag=$(basename "$SOURCE_CHUNK")
/usr/bin/time -v "$ABISS_HOME/build/acme" param.txt "$chunk_tag" \
  > acme.stdout.txt 2> acme.stderr.txt

mv "edges_${chunk_tag}.data" input_rg.data
cat boundary_*_"${chunk_tag}".data > frozen.data
test -s ns.data
test -s ongoing_seg_size.data
touch ongoing_semantic_labels.data ongoing_nuclei_labels.data
/usr/bin/time -v "$ABISS_HOME/build/agg" "${AGG_THRESHOLD:-0.3}" \
  input_rg.data frozen.data ns.data > agg.stdout.txt 2> agg.stderr.txt

test -s residual_rg.data
test -s remap.data
touch SUCCESS
echo "Pinned ABISS agg completed the regression chunk: $VALIDATION_DIR"
