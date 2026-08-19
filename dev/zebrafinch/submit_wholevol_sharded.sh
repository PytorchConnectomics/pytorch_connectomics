#!/bin/bash
# Orchestrate the whole-volume ABISS replay as a chain of SHARDED SLURM array jobs.
#
# ABISS pipeline = 4 stages; the two run_batch stages walk the chunk hierarchy
# (layer 0 atomic -> layer 1..top composite), the two remap stages touch layer 0 only:
#   1 watershed            layers 0..top  atomic/composite_chunk_ws   STAGE=ws
#   2 remap_watershed      layer 0        remap_chunk_ws              STAGE=ws
#   3 agglomerate_mean_edge layers 0..top atomic/composite_chunk_me   STAGE=agg
#   4 remap_agglomeration  layer 0        remap_chunk_agg             STAGE=agg
# Chunks WITHIN a layer are independent (run_layer.sh fans them across cores), so each
# layer becomes an array of shards; each layer waits on the previous via afterok.
# Higher layers hold bigger region graphs, so cpus drop and memory rises (mirroring
# abiss's own `parallel -j 8 / -j 4` throttling at layers >=2).
#
# Run prepare first (wholevol_prepare.py) so runtime.json + <layer>.txt exist.
set -eo pipefail
MAIN=/projects/weilab/weidf/lib/pytorch_connectomics
# WV selects the run directory, so the same launcher drives the precomputed-mirror
# run and the HDF5 chunk-store run without being copied.
WV=${WV:-$MAIN/dev/zebrafinch/wholevol}
RUNTIME_JSON=$WV/runtime.json
SH=$MAIN/dev/zebrafinch/sbatch_abiss_shard.sh
NUC_SCAN_SH=$MAIN/dev/zebrafinch/sbatch_nuccomp_scan.sh
NUC_FLOOD_SH=$MAIN/dev/zebrafinch/sbatch_nuccomp_flood.sh
NUC_MERGE_SH=$MAIN/dev/zebrafinch/sbatch_nuccomp_merge.sh
TOP=$(python -c "import json;print(json.load(open('$RUNTIME_JSON'))['top_mip'])")

# per-layer: shards cpus mem   (layer0 is the 10626-chunk bulk; upper layers are few+fat)
# Sized from the measured run (sacct on 2782527..2782540): the ABISS binaries are
# SINGLE-THREADED (no tbb::parallel / OpenMP / std::thread anywhere in src/), so one
# chunk = one core and concurrency is purely shards x cpus. The old table throttled
# cpus DOWN as layers rose (16/8/4/2/1), which left the upper layers -- 64% of total
# wall -- barely parallel at all: L4 ran 8 chunks 2-at-a-time for 33 min, L3 ran 27
# chunks 8-at-a-time.
#
# The throttle existed for MEMORY (a composite chunk's region graph grows with layer:
# ~8 GB/chunk at L0, ~16 at L2, ~20 at L3, ~39 at L4, ~98 at L5), so the fix is more
# NODES with FEW cpus each, not more cpus per node. Concurrency now covers each
# layer's whole task count where memory allows.
#
#   layer:  chunks   shards x cpus = concurrency
#     0     10626      80 x 19 = 1520
#     1      1452      24 x 16 =  384
#     2       216      16 x  4 =   64
#     3        27      14 x  2 =   28  (>= 27, one wave)
#     4         8       8 x  1 =    8  (= 8,  one wave)
#     5         1       1 x  1 =    1  IRREDUCIBLE -- single chunk, single thread.
#
# L5 is the Amdahl floor (~37 min for ws+me). Beating it would need a bigger
# CHUNK_SIZE, which comes from the provenance and would change the segmentation.
# Layer-0 concurrency raised 16 -> 19 to exploit mimalloc's measured 10.3% memory cut
# (7.37 -> 6.60 GB peak RSS, reproduced exactly across two independent builds). Memory is
# raised 130G -> 160G at the same time because the 6.60 GB figure is the peak of the LARGEST
# SINGLE PROCESS in a whole-chunk pipeline and that process may be `agg`, not the layer-0
# `ws`/`acme`. 19 x 6.6 = 125 GB if the saving applies at L0; 19 x 8.0 = 152 GB if it does
# not. Live jobs later reached 167,749,128 KiB -- effectively the full 160G cgroup limit --
# without failing. Use 180G for future runs to retain 19-way concurrency with headroom while
# still fitting the common ~191 GB nodes.
# L5 memory restored 200G -> 300G: cutting it to 200G made ws_L5 58% SLOWER (10.4 -> 16.4 min)
# on a stage that cannot be recovered by parallelism.
cfg_for() { case "$1" in
  0) echo "80 19 180G";; 1) echo "24 16 130G";; 2) echo "16 4 96G";;
  3) echo "14 2 96G";;  4) echo "8 1 120G";;  *) echo "1 1 300G";; esac; }

# START_AT=<stage name> resumes the chain (earlier stages already COMPLETED); chunk-level
# resume is free anyway since run_wrapper.sh skips chunks already flagged DONE.
START_AT=${START_AT:-}
STARTED=0
# INIT_DEP chains a whole run behind another job (e.g. a second affinity variant
# that should not contend with the first for the cluster).
DEP="${INIT_DEP:-}"
submit() { # $1=name $2=op $3=layer $4=stage_env
  if [ -n "$START_AT" ] && [ "$STARTED" = 0 ]; then
    if [ "$1" = "$START_AT" ]; then STARTED=1; else echo "  skip $1 (already done)"; return 0; fi
  fi
  read -r NS CPUS MEM <<<"$(cfg_for "$3")"
  local dep_arg=(); [ -n "$DEP" ] && dep_arg=(--dependency=afterok:"$DEP")
  local jid
  jid=$(sbatch --parsable "${dep_arg[@]}" --job-name="$1" --array=0-$((NS-1)) \
        --cpus-per-task="$CPUS" --mem="$MEM" \
        --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",OP="$2",LAYER="$3",STAGE_ENV="$4",NSHARD="$NS" \
        "$SH")
  jid=${jid%%;*}                      # --parsable may append ;cluster
  echo "  $1 (op=$2 layer=$3 shards=$NS cpus=$CPUS mem=$MEM) -> $jid${DEP:+ after $DEP}"
  DEP=$jid
}

submit_nucleus_competition() {
  local enabled max_units concurrency run_id
  enabled=$(python -c "import json; p=json.load(open('$RUNTIME_JSON')); q=json.load(open(p['param'])); print(int(bool(q.get('NUC_PATH')) and q.get('NUC_COMPETITION_ENABLED', True)))")
  [ "$enabled" = 1 ] || { echo "  skip nuccomp (disabled)"; return 0; }
  if [ -n "$START_AT" ] && [ "$STARTED" = 0 ]; then
    if [ "$START_AT" = nuccomp ]; then STARTED=1; else echo "  skip nuccomp (already done)"; return 0; fi
  fi
  max_units=$(python -c "import json; p=json.load(open('$RUNTIME_JSON')); q=json.load(open(p['param'])); print(int(q.get('NUC_MAX_UNITS', 64)))")
  [ "$max_units" -ge 1 ] || { echo "NUC_MAX_UNITS must be positive" >&2; return 1; }
  concurrency=${NUC_FLOOD_CONCURRENCY:-9}
  run_id=${NUC_COMPETITION_RUN_ID:-planv3-$(date -u +%Y%m%dT%H%M%SZ)-${BASHPID}}

  local dep_arg=(); [ -n "$DEP" ] && dep_arg=(--dependency=afterok:"$DEP")
  local scan_jid flood_jid merge_jid
  scan_jid=$(sbatch --parsable "${dep_arg[@]}" --job-name=nucscan \
        --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$run_id" \
        "$NUC_SCAN_SH")
  scan_jid=${scan_jid%%;*}
  flood_jid=$(sbatch --parsable --dependency=afterok:"$scan_jid" --job-name=nucflood \
        --array=0-$((max_units-1))%"$concurrency" \
        --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$run_id" \
        "$NUC_FLOOD_SH")
  flood_jid=${flood_jid%%;*}
  merge_jid=$(sbatch --parsable --dependency=afterok:"$flood_jid" --job-name=nucmerge \
        --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$run_id" \
        "$NUC_MERGE_SH")
  merge_jid=${merge_jid%%;*}
  echo "  nucscan -> $scan_jid${DEP:+ after $DEP}"
  echo "  nucflood array=0-$((max_units-1))%$concurrency -> $flood_jid after $scan_jid"
  echo "  nucmerge -> $merge_jid after $flood_jid"
  echo "  nuccomp run_id=$run_id"
  echo "  retry: sbatch --array=<failed indices> --export=ALL,RUNTIME_JSON=$RUNTIME_JSON,NUC_COMPETITION_RUN_ID=$run_id $NUC_FLOOD_SH"
  echo "  then submit $NUC_MERGE_SH with afterok on the retry job"
  DEP=$merge_jid
}

echo "submitting sharded whole-vol ABISS (top_mip=$TOP):"
for L in $(seq 0 "$TOP"); do
  [ "$L" = 0 ] && OP=atomic_chunk_ws || OP=composite_chunk_ws
  submit "ws_L$L" "$OP" "$L" ws
done
submit "remapws" remap_chunk_ws 0 ws
submit_nucleus_competition
for L in $(seq 0 "$TOP"); do
  [ "$L" = 0 ] && OP=atomic_chunk_me || OP=composite_chunk_me
  submit "me_L$L" "$OP" "$L" agg
done
submit "remapagg" remap_chunk_agg 0 agg
echo "final job: $DEP  (seg -> $(python -c "import json;print(json.load(open('$RUNTIME_JSON'))['seg_path'])"))"
echo "$DEP" > "$WV/final_jobid.txt"

# --- scoring ------------------------------------------------------------------------------
# The chain used to stop at remapagg, so an arm looked finished while its NERL did not exist:
# b4_mintag50 sat decoded-but-unscored for ~6 h because the scoring commands were only ever
# PRINTED as advice. Score both thresholds (MANUAL.md evaluation standard), chained on the
# final decode job so the numbers land without anyone remembering. SCORE=0 opts out.
if [ "${SCORE:-1}" != "0" ]; then
  SCORE_SH=$MAIN/dev/zebrafinch/sbatch_score_eval_matrix.sh
  if [ ! -f "$SCORE_SH" ]; then
    echo "  [score] WARNING: $SCORE_SH missing; scoring NOT chained." >&2
  else
    # runtime.json stores a CloudVolume URL; the scorer takes a filesystem path.
    SEG_LAYER=$(python -c "import json;print(json.load(open('$RUNTIME_JSON'))['seg_path'])" \
                 | sed 's#^file://##')
    for MT in 50 0; do
      # START_AT can skip every stage, leaving DEP empty; then score immediately.
      dep_arg=(); [ -n "$DEP" ] && dep_arg=(--dependency=afterok:"$DEP")
      sjid=$(sbatch --parsable "${dep_arg[@]}" --job-name="score_mt$MT" \
        --export=ALL,SEG="$SEG_LAYER",OUT="$WV/nerl_funlib_mt${MT}.json",MT=$MT "$SCORE_SH")
      sjid=${sjid%%;*}
      echo "  score mt=$MT -> $sjid${DEP:+ after $DEP}  -> $WV/nerl_funlib_mt${MT}.json"
    done
  fi
fi
