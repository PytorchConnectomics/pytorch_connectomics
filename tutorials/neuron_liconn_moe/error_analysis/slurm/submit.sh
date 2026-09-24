#!/bin/bash
# Submit sizes -> 16 skeleton shards -> finish for each run directory given.
# Memory scales with voxel count from the S1 run (0.78 Gvox: shards peaked at
# 32 GB, the analysis at < 80 GB).
#
#   bash tutorials/neuron_liconn_moe/error_analysis/slurm/submit.sh <run dir>...
set -eo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOGS=$HERE/slurm/logs
mkdir -p "$LOGS"
ACTIVATE="source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc"
N=16
for RUN in "$@"; do
    RUN=$(cd "$RUN" && pwd)
    NAME=$(basename "$RUN")
    if [ -f "$RUN/error_analysis/error_analysis.json" ]; then
        echo "skip $NAME: error_analysis.json exists"
        continue
    fi
    SEG=$(ls "$RUN"/"$NAME"_seg_abiss_mt*.h5 | grep -v nogt)
    SCALE=$(python3 -c "import h5py,sys,math; s=h5py.File(sys.argv[1],'r')['main'].shape; print(max(1.0, math.prod(s)/7.76e8))" "$SEG")
    MEM_SHARD=$(python3 -c "import math;print(math.ceil(48*$SCALE))")
    MEM_FINISH=$(python3 -c "import math;print(math.ceil(80*$SCALE))")
    ENV="--export=ALL,LICONN_EA_RUN=$RUN"
    S=$(sbatch --parsable -p short -t 01:00:00 -c 4 --mem=24G $ENV -J "ea_sizes_$NAME" \
        -o "$LOGS/${NAME}_sizes_%j.log" --wrap "$ACTIVATE; bash $HERE/run_volume.sh sizes")
    K=$(sbatch --parsable -p short -t 06:00:00 -c 8 --mem="${MEM_SHARD}G" $ENV \
        --array=0-$((N - 1))%8 --dependency=afterok:$S -J "ea_skel_$NAME" \
        -o "$LOGS/${NAME}_skel_%A_%a.log" \
        --wrap "$ACTIVATE; bash $HERE/run_volume.sh shard \$SLURM_ARRAY_TASK_ID $N")
    F=$(sbatch --parsable -p short -t 04:00:00 -c 8 --mem="${MEM_FINISH}G" $ENV \
        --dependency=afterok:$K -J "ea_finish_$NAME" \
        -o "$LOGS/${NAME}_finish_%j.log" --wrap "$ACTIVATE; bash $HERE/run_volume.sh finish $N")
    echo "$NAME sizes=$S skel=$K finish=$F mem ${MEM_SHARD}G/${MEM_FINISH}G"
done
