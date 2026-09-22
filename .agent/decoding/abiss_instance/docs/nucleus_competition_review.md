# ABISS competitive nucleus growth: robustness and efficiency review

## Verdict

The implementation is suitable for controlled cluster validation, but it is not yet ship-ready as
a general nucleus feature. The refactor removes the most expensive silent-failure modes: final
remap now has a real overlay oracle, every unit is tied to an immutable scan plan, canonical
publication is transactional, zero repairs is an acceptance failure for intervention arms, and all
five declared fail-closed conditions have executable coverage. Efficiency acceptance is deliberately
pending: neither 7--11 hour production baseline was run during implementation.

The minimum ship gate is: materialized `fused_source_pairs == 0`, nuclei 611/651 in distinct
dominant output segments, exact equality outside every repair box, exact already-clean-nucleus
report fields, reference-matching `nuc_cuts.data` counts at every hierarchy level, emitted-label and
array-throttle invariance, and failure-injection retry. Cross-arm effect-size claims additionally
require an `ABISS_NUC_MIN_TAGGED` sweep because win144 uses 50 and native96 uses 1024.

## Ranked robustness findings

Rank is probability of silently invalidating an experiment multiplied by cost of detection.

1. **R1, graph/volume divergence -- addressed in code, cluster gate still required.** The former
   test stubbed `apply_nucleus_competition` and could only prove that `cut_chunk_remap.py` called a
   symbol. The replacement drives `cut_chunk_remap.py` through the real manifest loader, real
   territory array, real internal-to-emitted translation, and real overlay, then checks the written
   `seg.raw`. This would catch B2. The whole-volume materialized check remains mandatory because a
   synthetic unit test cannot validate deployed runtime hashes or output assembly.
2. **R7, partial fail-closed behavior -- addressed.** Missing unanimous seeds, overlapping scopes
   for one parent, generated-id collisions, incompatible watershed identities, and invalid nucleus
   coordinates are all reachable. The first, second, fourth, and fifth are realistic configuration
   or data errors; a SHA-256 id collision is theoretical but cheap to reject. Scan, flood, and merge
   write only into `.nuccomp-runs/<run-id>`; merge validates all records before replacing the
   canonical manifest, so failure leaves the previous publication intact. Unit tests cover all five
   aborts plus stale-plan rejection and preservation of the last publication.
3. **R2, silent no-op -- addressed as observability, not correctness.** Zero units remains a valid
   segmentation no-op. Scan and merge emit loud warnings, the final manifest records
   `zero_repairs: true`, and `nucleus_acceptance_report.py` reports a completed zero-repair
   intervention as `FAIL`. An array index above the unit count exits zero by design.
4. **R5, permissive untagged bridge ownership -- unresolved determinism risk.** The owning path is
   `lib/abiss/src/agg/mean_aggl.cpp::nuc_can_merge` at the pre-merge veto, followed by
   `nuc_join`. The heap comparator compares only edge weight; it has no endpoint tie-break. A
   synthetic RAG with `PROPER(1) -- NONE -- PROPER(2)` must be run with reversed equal-weight edge
   insertion and with alternate hierarchy/shard layouts. The ownership oracle is that the `NONE`
   node remaps to the same proper identity in every case. Divergence means exclusion is stable but
   ownership is order-dependent. No cannot-link algorithm change was authorized in this refactor,
   so this remains a ship gate rather than being silently declared deterministic.
5. **R3, bounded-ROI containment -- handled honestly, not proved.** The scan does not prove that a
   parent watershed id has no disconnected material beyond its repair box. Every repair and the
   final manifest therefore records `separation_claim: local_only`. A global claim requires the
   outside-box materialization comparison.
6. **R4, parameter non-equivalence -- unresolved experimental-design limitation.** All cross-arm
   results must state that `ABISS_NUC_MIN_TAGGED` is 50 on win144 and 1024 on native96. There is no
   evidence for a parameter-independent effect size until the planned sweep ties this 20x change to
   outcomes.
7. **R6, `z4_y6_x1` null -- unresolved with a falsifiable test.** Score the same chunk once with the
   local scorer and once with the whole-volume node LUT filtered to nodes whose coordinates fall in
   `z4_y6_x1`. If local scoring is blind to cross-chunk merges, the whole-volume-restricted score
   must be lower. Equality leaves the null unexplained and blocks shipping; the predicted difference
   explains and closes it. Both scores must use funlib matching, voxel units, and state the merge
   threshold.

Native96's recorded per-skeleton evidence is encouraging but not dispositive: with funlib matching
in voxel units it improves 3 skeletons and regresses 1 at mt=50 (skel28, -16,274 ERL, -11.5% of its
GT length), and improves 5 with zero regressions at mt=0. The aggregate scores remain
native96 nucleus 0.481614 versus naive 0.469513 at mt=50, and 0.287184 versus 0.267679 at mt=0;
win144 nucleus 0.468030 versus naive 0.444376 at mt=50, and 0.318718 versus 0.304262 at mt=0.

## Efficiency plan

The code confirms that instrumentation must precede further optimization: `scan_geometry` and
`map_to_watershed` are separate whole-volume passes, while each flood is bounded. The final
`stage_report.json` records both scans separately, every `t_i`, `sum(t_i)`, `max(t_i)`, merge time,
and the serial/array critical-path model.

1. Run the serial and fixed-capacity array baselines below. The array preserves determinism by
   preallocating internal ids from `(parent, anchor)`, selecting a winner by largest pooled voxel
   count then lowest anchor id, and validating the exact mapping domain. Expected speedup is
   `(T_nonflood + sum(t_i)) / (T_nonflood + max(t_i) + scheduler_overhead)`. Balanced nine-unit
   illustrations are 1.10x at flood fraction 0.1, 1.55x at 0.4, and 2.65x at 0.7. PASS requires
   measured critical path no more than 1.15x predicted; an array slower than serial is a regression
   and leaves serial as default. Determinism risk is low after emitted-label and throttle gates.
2. Preserve retryability even if speedup is small. Rerun only failed indices and submit merge as a
   new job after that retry; exact `plan_digest` matching prevents an old unit from satisfying
   fan-in. Failure injection must prove all other `terr_*.npz` files stay byte-identical.
3. Use the measured selection rule. If `map_to_watershed` is the largest phase and at least 40%,
   shard nuclei by id range, use chunk-grouped watershed reads, and reduce integer histograms. If
   `scan_geometry` is largest and at least 40%, shard nucleus-mask z-slabs and reduce integer counts,
   sums, and bbox unions. If flood dominates, stop: the array already targets it. Otherwise record
   the table and build neither. For a selected scan fraction `p` and `s` effective shards, the ideal
   ceiling is `1 / (1 - p + p/s)` before I/O contention. Integer reductions preserve determinism;
   store-read concurrency is the operational risk.
4. Capture `MaxRSS` for scan and every flood before changing the 128 GB request. Flood workers are
   single-CPU because the Python/skimage path is not made parallel by requesting eight CPUs.

Scan sharding is designed but deferred. Geometry shards use source-mask z-slabs and associative
integer reductions. Mapping shards use nucleus-id ranges and emit `(nucleus_id, watershed_id,
count)` triples whose keyed sums are order-independent.

## Supported watershed reuse repair

`connectomics/runtime/abiss_chunk.py` now resolves `CHUNKMAP_INPUT` independently and defaults it to
`CHUNKMAP_OUTPUT` only when absent. An explicit input therefore survives payload materialization and
can point at the completed watershed chunkmap; a 126 MB copy is no longer necessary. Turning reuse
into a public workflow flag remains out of scope. A reuse run must still freeze the authoritative
watershed manifest digest and keep `WS_PATH`, `WS_MANIFEST`, and `CHUNKMAP_INPUT` read-only.

## Frozen production input identities

These were computed read-only with the final fingerprint command before cluster execution. The
fresh parameter copies below intentionally have different parameter hashes because their manifest
namespaces differ.

| Input | native96 (`ABISS_NUC_MIN_TAGGED=1024`) | win144 (`ABISS_NUC_MIN_TAGGED=50`) |
|---|---|---|
| source param SHA-256 | `fe91ed78d1b66ef6ef1d8f15bf1342fd73d148c073357f355f0fe43b9e434d1b` | `978322488ee319cd4c4898c3b91572a844b6685d8b56da6bae0b352d58f16599` |
| nucleus SHA-256 / bytes | `13c982231584ccda1b525778e3235cddd68a3a538cd6fd4bf03c7393a499d44b` / 47,005,708 | same |
| watershed manifest SHA-256 | `341e738bd0dd61e089172165e8d47431d41e8abbd6a918d98bb3e9ba277dd77b` | `313344a824857e85b8083f8cefb6674ba9595f311f017bde1214f837b2adee93` |
| affinity index SHA-256 / chunks | `f80045ea47c8b633c15252dc73f95dd458871a291c22eca5e35aa5028293966a` / 726 | `b0a6e6e598d7cd0960c8f18183d34bc8c060ba423f9a961c483aea379e467a4b` / 726 |
| ABISS build identity | `git:452efa5f87f9d3cb241891ee44010d966a33b316:runtime-sha256:9cdbe0f48b7916e89d68736980fb2590ec20a1c89435780d4407a6596c4de62b` | same |
| ABISS Python sources SHA-256 | `9885684dc0b9a90174c551ab9a9139558ccc96e157de03c300cb6f12991042db` | same |

Expected fresh param hashes after the preparation command are:

| Arm | serial | array throttle 1 | array throttle 9 |
|---|---|---|---|
| native96 | `02290807b23187ef045bf7a24383c6e9d4eb175a94f56b170558122a8c7613fe` | `5a33acf996f0dcd8a851fbd41d22d6628f3c0440d3b4050fff73448cc710d972` | `69566b089621bc2bc751f79cd6234cba423ef1496888e16a7f673b5ed388e1c3` |
| win144 | `a7d5e5defff93dca8b74cad9dde2e6d03997016c938ca7ac24132482a6b4c6ad` | `ef7c6f2cbad47079f7f1852e2f17889b1c519001e526f49f13db0414305ec9c8` | `a4eda44255fa8ee65e7db1cab9b27b7b5fcce9f5a8315f4751fcd56b268140e0` |

## Exact cluster baseline commands

Run from the repository root. The preparation refuses to reuse a namespace.

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTC_REPO=/projects/weilab/weidf/lib/pytorch_connectomics
export NUC_BASE=$PYTC_REPO/dev/zebrafinch/nuccomp_planv3_baselines

prepare_nuccomp() {
  local source_runtime=$1 arm=$2 mode=$3 ws_manifest=$4 ws_sha=$5
  local out=$NUC_BASE/${arm}_${mode}
  test ! -e "$out" || { echo "namespace exists: $out" >&2; return 1; }
  mkdir -p "$out/nucleus_competition"
  python - "$source_runtime" "$out" "$ws_manifest" "$ws_sha" <<'PY'
import json, sys
from pathlib import Path
source_runtime = Path(sys.argv[1])
out = Path(sys.argv[2])
ws_manifest = Path(sys.argv[3])
ws_sha = sys.argv[4]
runtime = json.loads(source_runtime.read_text())
param = json.loads(Path(runtime["param"]).read_text())
param["NUC_COMPETITION_MANIFEST"] = str(out / "nucleus_competition/manifest.json")
param["WS_MANIFEST"] = str(ws_manifest)
param["WS_MANIFEST_SHA256"] = ws_sha
param["NUC_MAX_UNITS"] = 64
param_path = out / "param.json"
param_path.write_text(json.dumps(param, indent=2, sort_keys=True) + "\n")
runtime["param"] = str(param_path)
(out / "runtime.json").write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
PY
}

NATIVE_RUNTIME=$PYTC_REPO/dev/zebrafinch/wholevol_arm0_native96_nuc_matchguard/runtime.json
NATIVE_WS=$PYTC_REPO/dev/zebrafinch/wholevol_arm0_native96_nuc_matchguard/seg_arm0_native96_nuc_matchguard/manifest.json
WIN_RUNTIME=$PYTC_REPO/dev/zebrafinch/wholevol_arm096_nuc_competitive_v2/runtime.json
WIN_WS=$PYTC_REPO/dev/zebrafinch/wholevol_arm096_nuc_competitive_v2/seg_arm096_nuc_competitive_v2/manifest.json

for mode in serial array_t1 array_t9; do
  prepare_nuccomp "$NATIVE_RUNTIME" native96 "$mode" "$NATIVE_WS" 341e738bd0dd61e089172165e8d47431d41e8abbd6a918d98bb3e9ba277dd77b
  prepare_nuccomp "$WIN_RUNTIME" win144 "$mode" "$WIN_WS" 313344a824857e85b8083f8cefb6674ba9595f311f017bde1214f837b2adee93
done
sha256sum "$NUC_BASE"/*/param.json
```

Submit serial baselines:

```bash
for arm in native96 win144; do
  sbatch --export=ALL,RUNTIME_JSON="$NUC_BASE/${arm}_serial/runtime.json" \
    dev/zebrafinch/sbatch_nucleus_competition.sh
done
```

Submit array runs, preserving the printed ids per arm/mode:

```bash
submit_array() {
  local arm=$1 mode=$2 throttle=$3 run_id=${arm}-${mode}-planv3
  local runtime=$NUC_BASE/${arm}_${mode}/runtime.json
  local scan flood merge
  scan=$(sbatch --parsable --export=ALL,RUNTIME_JSON="$runtime",NUC_COMPETITION_RUN_ID="$run_id" dev/zebrafinch/sbatch_nuccomp_scan.sh); scan=${scan%%;*}
  flood=$(sbatch --parsable --dependency=afterok:"$scan" --array=0-63%"$throttle" --export=ALL,RUNTIME_JSON="$runtime",NUC_COMPETITION_RUN_ID="$run_id" dev/zebrafinch/sbatch_nuccomp_flood.sh); flood=${flood%%;*}
  merge=$(sbatch --parsable --dependency=afterok:"$flood" --export=ALL,RUNTIME_JSON="$runtime",NUC_COMPETITION_RUN_ID="$run_id" dev/zebrafinch/sbatch_nuccomp_merge.sh); merge=${merge%%;*}
  echo "$arm $mode scan=$scan flood=$flood merge=$merge run_id=$run_id"
}
submit_array native96 array_t1 1
submit_array native96 array_t9 9
submit_array win144 array_t1 1
submit_array win144 array_t9 9
```

Expected artifacts for each successful array run are:

```text
<namespace>/nucleus_competition/manifest.json
<namespace>/nucleus_competition/.nuccomp-runs/<run-id>/units.json
<namespace>/nucleus_competition/.nuccomp-runs/<run-id>/scan_report.json
<namespace>/nucleus_competition/.nuccomp-runs/<run-id>/terr_<unit>_<parent>.npz
<namespace>/nucleus_competition/.nuccomp-runs/<run-id>/unit_<unit>.json
<namespace>/nucleus_competition/.nuccomp-runs/<run-id>/flood_<index>.report.json
<namespace>/nucleus_competition/.nuccomp-runs/<run-id>/stage_report.json
```

Native96 must report 8 contact units and zero bridges. Win144 must report 9 units, including parent
`72199226020331523` with anchors `[611, 651]`. Any difference fails semantic equality.

## Cluster verification gates

1. Compare reference and refactored repairs after translating reference markers through
   `marker_labels` and refactored internal ids through each repair's explicit `emitted_id`.
   Expanded post-overlay label arrays must be byte-identical; `terr_*.npz` identity is not expected.
2. Require semantic manifest equality: territory set, parent, bbox, factor, anchor set, and pooled
   voxel counts. Check the expected 8/9 unit counts above.
3. Compare array throttle 1 and 9 manifests and emitted arrays. They must be identical after removing
   timestamps, run ids, paths, timing, and fingerprints whose only difference is the fresh param.
4. Inject failure in unit 8 of win144. Save `sha256sum` for every other `terr_*.npz`, rerun only
   index 8 with the same run id, verify those hashes are unchanged, then submit merge separately:

   ```bash
   retry=$(sbatch --parsable --array=8 --export=ALL,RUNTIME_JSON="$NUC_BASE/win144_array_t9/runtime.json",NUC_COMPETITION_RUN_ID=win144-array_t9-planv3 dev/zebrafinch/sbatch_nuccomp_flood.sh); retry=${retry%%;*}
   sbatch --dependency=afterok:"$retry" --export=ALL,RUNTIME_JSON="$NUC_BASE/win144_array_t9/runtime.json",NUC_COMPETITION_RUN_ID=win144-array_t9-planv3 dev/zebrafinch/sbatch_nuccomp_merge.sh
   ```

5. Parse `stage_report.json` and apply the efficiency PASS/FAIL rule above. This gate is pending and
   must not be inferred from unit-test success.
6. Run `nucleus_shell_contamination.py --tol 0.0` on each materialized output. Require zero fused
   source pairs and distinct dominant segments for 611/651. For every nucleus absent from
   `units.json`, require exact equality of `dominant_segment`, `total_mask_voxels`, and the complete
   ordered `segments` list `(seg, voxel count)` against reference.
7. Compare material outside the union of repair boxes voxel-for-voxel and parse `nuc_cuts.data` at
   every hierarchy level. Outside-box material must be unchanged and rejected-edge counts must equal
   counts frozen from the completed references before launching the refactor.
8. Report every NERL with funlib matching, voxel units, and both mt=50 and mt=0. Never compare
   `--nm` results to the FFN voxel-unit reference.

Risk retained by design: the affinity fingerprint hashes the authoritative index and records the
chunk count, not every multi-gigabyte chunk. An in-place chunk edit that preserves both can escape
detection. The already-clean-nucleus gate also compares report fields rather than all 58 GB of
voxels.
