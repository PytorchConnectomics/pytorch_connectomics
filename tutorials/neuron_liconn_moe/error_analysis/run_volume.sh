#!/bin/bash
# No-GT error analysis for one moe run directory. Shared by SLURM and the image.
#
#   LICONN_EA_RUN=<run dir> run_volume.sh sizes
#   LICONN_EA_RUN=<run dir> run_volume.sh shard <i> <num_shards>
#   LICONN_EA_RUN=<run dir> run_volume.sh finish <num_shards>
#   LICONN_EA_RUN=<run dir> run_volume.sh reports    # catalog + sidecar + report only
#   LICONN_EA_RUN=<run dir> run_volume.sh all [num_shards]      # sequential, one host
#
# Settings are the published ExPID96 S1 run's (pieces_chain.sh, 2026-09-17):
# 10 nm skeleton simplification, caliber gates 0.15/0.20 um, terminal windows
# 0.10/0.15 um, 0.4 um pieces. --max-voxels 5e6 reproduces its giant-label skip
# (S1 skipped exactly one label, 64008 at 12.3 M voxels). The gates were
# calibrated on S1; caliber_survey.py output is kept per volume to check them.
set -eo pipefail
: "${LICONN_EA_RUN:?set LICONN_EA_RUN to the volume run directory}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT="$LICONN_EA_RUN/error_analysis"
FINE="$OUT/fine"
SKEL="$OUT/skeletons_fine.npz"
CPUS=${LICONN_EA_CPUS:-$(nproc)}
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$FINE"

sizes() {
    [ -f "$LICONN_EA_RUN/label_sizes.npz" ] && { echo "label_sizes.npz present"; return; }
    python "$HERE/label_sizes.py"
}

shard() {
    local i=$1 n=$2 leaf
    printf -v leaf "skeletons_shard%03d.npz" "$i"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$HERE/build_skeletons.py" \
        --parallel "$CPUS" --shard "$i" --num-shards "$n" \
        --simplification-nm 10 --max-voxels 5000000 \
        --output "$FINE/$leaf" --overwrite
}

finish() {
    local n=$1
    if [ -f "$OUT/error_analysis.json" ]; then
        echo "refusing: $OUT/error_analysis.json exists; move it aside first" >&2
        exit 1
    fi
    export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
    python "$HERE/merge_skeletons.py" --shards "$n" --directory "$FINE" --output "$SKEL"
    python "$HERE/caliber_survey.py" --skeletons "$SKEL" | tee "$OUT/caliber_survey.txt"
    python "$HERE/analyze.py" --skeletons "$SKEL" \
        --axon-max-radius-um 0.15 --dendrite-min-radius-um 0.20 \
        --head-window-um 0.10 --shaft-window-um 0.15 --piece-length-um 0.4
    python "$HERE/link_splits.py" --skeletons "$SKEL" --no-semantic-gate
    python "$HERE/plot_analysis.py"
    reports
}

# Everything downstream of error_analysis.json: the five-class catalog, the
# Neuroglancer sidecar that shows it, and the GT-free report. Safe to rerun.
reports() {
    python "$HERE/end_evidence.py"
    semantic
    python "$HERE/make_segment_properties.py"
    python "$HERE/make_report.py"
}

# The five-class catalog (axon, dendrite, glia_or_soma, blood_vessel,
# unclassified) built from error_analysis.json; re-verifies the segmentation
# hash and the full label histogram before writing.
semantic() {
    local layer title
    layer=$(cd "$HERE" && python -c "import volume as V; print(f'gs://{V.GCS_BUCKET}/{V.GCS_ANALYSIS_PREFIX}')")
    title="$(basename "$LICONN_EA_RUN") · $(basename "$(dirname "$LICONN_EA_RUN")") · initial semantic candidates"
    python "$HERE/../build_semantic_catalog.py" --volume-dir "$LICONN_EA_RUN" \
        --output "$LICONN_EA_RUN/semantic" --layer-uri "$layer" --title "$title"
}

echo "$(hostname) $1 $(basename "$LICONN_EA_RUN") started $(date -Is)"
case "$1" in
    sizes) sizes ;;
    shard) shard "$2" "$3" ;;
    finish) finish "$2" ;;
    semantic) semantic ;;
    reports) reports ;;
    all)
        n=${2:-16}
        sizes
        for ((i = 0; i < n; i++)); do shard "$i" "$n"; done
        finish "$n"
        ;;
    *) echo "unknown step: $1" >&2; exit 2 ;;
esac
echo "finished $1 $(date -Is)"
