# Nucleus Competition Review

## Validation Status and Ship Verdict

**Do not read this document as validation of the refactor.** The focused unit suite and the
published-volume realization subgate have run. Full verification gates 2 through 8 remain unrun,
including emitted-label equivalence, reference-manifest semantics, shard-count invariance,
production failure injection, efficiency, retry recovery, and the other downstream segmentation
checks. The 7--11 hour cluster baselines have not been launched and gate 6 has not passed.

The implementation is therefore **not cleared to ship as behavior-preserving**. It is ready for an
operator to execute the frozen comparison and cluster procedures below. R5 and R6 also remain open
ship blockers until their named experiments run.

## Scope Record

The scan/flood/merge implementation follows plan v3 with three explicit scope amendments:

- `connectomics/runtime/abiss_chunk.py` now retains an explicit `CHUNKMAP_INPUT` and defaults it to
  `CHUNKMAP_OUTPUT` only when unset. This two-line production repair was outside plan v3's declared
  file list. The planner explicitly accepted it because it removes the documented 126 MB chunkmap
  copy workaround without overriding user configuration.
- `dev/zebrafinch/sbatch_nuccomp_scan.sh` and `sbatch_nuccomp_merge.sh` were also outside the
  declared file list. They are retained as the natural launchers required by the declared
  three-operation split; the flood launcher alone cannot form the dependency chain.
- `NUC_MAX_UNITS=64` is confirmed policy. Scan logs observed units as `observed/64`, fails closed
  when the observed count is greater than 64, and never truncates the unit plan.

Production `lib/abiss/scripts/nucleus_overlay.py` remains strict: it accepts schema 2.0 completed
publications only. Schema 1.2 now has a no-recomputation migration to that boundary; schema 1.0
labels are opaque legacy ids and remain supported solely by the read-only comparison harness.

## Legacy Publication Migration

Before relaunching a consumer of an existing schema-1.2 publication, run this exact command from
the repository root:

```bash
python lib/abiss/scripts/migrate_nucleus_competition.py \
  --manifest /absolute/path/to/nucleus_competition/manifest.json
```

The command infers the authoritative watershed manifest, hashes it, preserves the exact original
as `manifest.schema-1.2.<digest>.json`, writes a plan sidecar, and atomically publishes schema 2.0.
It does not rerun scan, flood, or merge and does not rewrite territory `.npz` files; legacy arrays remain
`int32` marker indices and are declared as `territory_encoding: marker_index`. This is the path for
relaunching `dev/zebrafinch/aggsweep/b3_gate025` without repeating the 11-hour nucleus-competition
stage. The production rejection message prints this command with the rejected manifest's absolute
path.

Fresh and migrated publications declare `identity.scope: nucleus`, retired adjudicated parents,
`residue_disposition: parent_retained`, and the exact deterministic mint descriptor. Validators
enforce those declarations rather than an assumed largest-territory convention. The ledger records
each repaired parent, every canonicalized single-owner source, and each many-to-one consolidation.
Parent ids may remain on unadjudicated residue, so their absence in the published volume is not a
criterion.

## R1--R7 Review

- **R1, zero repairs:** `zero_repairs` is durable in the manifest and stage report. The acceptance
  reporter judges a completed intervention with zero repairs as failure. Overlay execution still
  runs ownership filtering and canonicalization, and the publication has a nonempty `reason`.
- **R2, fail-closed publication:** invalid ids, overlapping scopes, missing seeds, incompatible
  watershed identity, and incomplete or stale fan-in abort before canonical publication. The last
  successful `manifest.json` remains in place.
- **R3, spatial scope:** every repair declares `separation_claim: local_only`. No global separation
  claim is allowed until the outside-repair-box regression gate passes.
- **R4, cross-arm evidence:** native96 and win144 use different `ABISS_NUC_MIN_TAGGED` values.
  Cross-arm effect sizes are descriptive and not a controlled algorithm comparison.
- **R5, ownership order:** the owner is
  `lib/abiss/src/agg/mean_aggl.cpp::nuc_can_merge` plus its attachment ordering. Run a synthetic RAG
  with one untagged object bridging two `PROPER` identities, reverse edge iteration, and vary shard
  count. Any owner divergence blocks shipment. This experiment is not run.
- **R6, local scoring:** score `z4_y6_x1` with the local per-chunk scorer and with the whole-volume
  node LUT restricted to that chunk's nodes. A lower whole-volume-restricted score explains and
  closes the local null; equality leaves it open. This experiment is not run.
- **R7, retryability:** unit records are bound to the exact `plan_digest`. Recovery reruns only
  failed indices, then submits a new merge dependency. Gate 7 must still demonstrate unchanged
  sibling territory files and an identical final manifest.

Measured native96 evidence remains descriptive: at `min_tagged=50`, three skeletons improve and
one regresses (skel28: -16,274 ERL, -11.5% of its GT length); at `min_tagged=0`, five improve and
none regress. These observations do not validate the refactor or close R4--R6.

Canonicalization is a naming consolidation, not a refinement invariant. The measured native96
manifest retires 77 single-owner base segments into 15 nucleus labels, with up to seven sources in
one label. If a nucleus mask crosses two real cells, this can silently fuse them. The publication
ledger makes that behavior inspectable; it does not remove the underlying mask-quality risk.

## Published-Volume Realization Gate

The gate reads each run's own `nucleus_shell_contamination_tol0.json`, which was computed solely on
nucleus-mask voxels. For every repair unit it requires distinct dominant published segments, a
declared dominance threshold, and equality between each dominant segment and that nucleus's
declared emitted label:

```bash
python dev/zebrafinch/nucleus_acceptance_report.py \
  --realization-run dev/zebrafinch/wholevol_arm0_native96_nuc_matchguard \
  --realization-dominance 0.6
```

The focused regression reproduces the measured oracle: win144 realizes 0/9 repair units; native96
realizes 8/8 units and 15/15 unique owner labels. This discriminates the collapsed and realized
publications. Parent-id absence does not: parent ids are absent at nucleus-overlapping voxels in
both runs.

## Read-Only Legacy Comparison Harness

`dev/zebrafinch/compare_nucleus_competition.py` is comparison tooling, not a production import.
For schema 1.0 and 1.2 it reads `marker_labels` and the existing marker-valued territory arrays to
reproduce exactly what the legacy overlay emitted. For schema 1.2 it also applies the historical
single-owner canonicalization. For schema 2.0 it requires a valid completion marker, matching
`plan_digest`, verified `units.json`, and territory hashes. It never modifies a manifest,
territory, watershed, or reference run.

The completed, frozen references are:

| Parameterization | Reference | Legacy schema | Repairs | Bridges |
|---|---|---:|---:|---:|
| native96 | `wholevol_arm0_native96_nuc_matchguard` | 1.2 | 8 | 0 |
| win144 | `wholevol_arm096_nuc_competitive_v2` | 1.0 | 9 | 0 |

Run gates 2 and 3 from the repository root after the fresh schema-2.0 publications complete. Set
only the two candidate paths; all reference paths and expected semantics are fixed here:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
ROOT=/projects/weilab/weidf/lib/pytorch_connectomics
COMPARE="$ROOT/dev/zebrafinch/compare_nucleus_competition.py"

NATIVE_RUN="$ROOT/dev/zebrafinch/wholevol_arm0_native96_nuc_matchguard/seg_arm0_native96_nuc_matchguard"
NATIVE_REF="$NATIVE_RUN/nucleus_competition/manifest.json"
NATIVE_PARAM="$NATIVE_RUN/param"
NATIVE_CANDIDATE=/absolute/path/to/fresh-native96/nucleus_competition/manifest.json

WIN_RUN="$ROOT/dev/zebrafinch/wholevol_arm096_nuc_competitive_v2/seg_arm096_nuc_competitive_v2"
WIN_REF="$WIN_RUN/nucleus_competition/manifest.json"
WIN_PARAM="$WIN_RUN/param"
WIN_CANDIDATE=/absolute/path/to/fresh-win144/nucleus_competition/manifest.json

python "$COMPARE" labels \
  --param "$NATIVE_PARAM" \
  --reference-manifest "$NATIVE_REF" \
  --candidate-manifest "$NATIVE_CANDIDATE"
python "$COMPARE" manifests \
  --reference-manifest "$NATIVE_REF" \
  --candidate-manifest "$NATIVE_CANDIDATE" \
  --expected-units 8 --expected-bridges 0

python "$COMPARE" labels \
  --param "$WIN_PARAM" \
  --reference-manifest "$WIN_REF" \
  --candidate-manifest "$WIN_CANDIDATE"
python "$COMPARE" manifests \
  --reference-manifest "$WIN_REF" \
  --candidate-manifest "$WIN_CANDIDATE" \
  --expected-units 9 --expected-bridges 0 \
  --expect-unit 72199226020331523:611,651
```

The `labels` action reads the immutable base watershed in bounded blocks over every frozen reference
repair box, applies each publication independently, and requires byte-identical emitted arrays.
Any differing voxel fails with its exact block. The `manifests` action compares parent ids, anchor
sets, pooled voxel counts, and bridge records after removing only the internal-id representation
difference. It does not infer winners or rewrite legacy ids.

For a small diagnostic cutout, add the same fixed `--bbox X0 Y0 Z0 X1 Y1 Z1` to a `labels`
command. Omitting `--bbox` is the full gate-2 procedure and covers every reference repair box.

## Cluster Execution and Retry

Use a fresh run directory and run id for each parameterization. `runtime.json` must point to the
fresh parameter file while reusing the completed watershed through its authoritative manifest.

```bash
export RUNTIME_JSON=/absolute/path/to/fresh-run/runtime.json
export NUC_COMPETITION_RUN_ID=planv3-$(date -u +%Y%m%dT%H%M%SZ)

scan=$(sbatch --parsable \
  --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$NUC_COMPETITION_RUN_ID" \
  dev/zebrafinch/sbatch_nuccomp_scan.sh)
scan=${scan%%;*}
flood=$(sbatch --parsable --dependency=afterok:"$scan" --array=0-63%9 \
  --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$NUC_COMPETITION_RUN_ID" \
  dev/zebrafinch/sbatch_nuccomp_flood.sh)
flood=${flood%%;*}
sbatch --dependency=afterok:"$flood" \
  --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$NUC_COMPETITION_RUN_ID" \
  dev/zebrafinch/sbatch_nuccomp_merge.sh
```

If flood indices fail, leave `units.json` and every successful `terr_*.npz` in place:

```bash
retry=$(sbatch --parsable --array=<failed-indices> \
  --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$NUC_COMPETITION_RUN_ID" \
  dev/zebrafinch/sbatch_nuccomp_flood.sh)
retry=${retry%%;*}
sbatch --dependency=afterok:"$retry" \
  --export=ALL,RUNTIME_JSON="$RUNTIME_JSON",NUC_COMPETITION_RUN_ID="$NUC_COMPETITION_RUN_ID" \
  dev/zebrafinch/sbatch_nuccomp_merge.sh
```

## Verification Gates

| Gate | Criterion | Current status |
|---:|---|---|
| 1 | Focused unit suite | Passed locally: 26 tests |
| 2 | Reference and schema-2.0 emitted label arrays byte-identical | **Unrun** |
| 3 | Translated manifest semantics equal (native96 8/0, win144 9/0) | **Unrun** |
| 4 | Flood throttle 1 versus N produces identical merge | **Unrun** |
| 5 | Five injected failures preserve the last canonical manifest | **Unrun** |
| 6 | Measured array critical path within efficiency acceptance | **Unrun; not claimed** |
| 7 | Failed-unit-only retry preserves all sibling territories | **Unrun** |
| 8 | Realization 0/9 vs 8/8 and 15/15 reproduced; outside-box, clean-nucleus, and `nuc_cuts.data` checks **unrun** |
| 9 | CCC validation and unchanged HEAD | Coordinator validation pending for this version |

## Efficiency Decision

Use the stage report's measured values:

```text
T_serial = T_nonflood + sum(t_i)
T_array  = T_nonflood + max(t_i) + scheduler_overhead
```

Pass when measured critical path is at most `1.15 * predicted T_array`. If it exceeds
`T_serial`, keep serial as the default; retain the array only if gate 7 independently proves retry
value. Follow-up selection uses the measured dominant fraction: shard `map_to_watershed` at >=0.4,
shard `scan_geometry` at >=0.4, accept the array when flood dominates, otherwise build nothing.
Both scan sharding designs remain deferred.

Full schema-3.0 provenance (`inputs[].semantics`), required-capability negotiation, and a measured
native-id-space scan are deferred to a follow-up run. They are not implied by the schema-2.0
identity and ledger implemented here.
