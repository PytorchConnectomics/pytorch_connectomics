# Code v0
## Overview

Implemented Step 1 of the zebrafinch whole-volume neuron graph agglomeration: a
total global-label namespace, big-branch face continuation candidates, a
transitive nucleus-firewalled union-find, complete remap/certificate outputs,
and exactly-once streamed NERL remapping. All synchronous hard controls passed.
The real two-chunk smoke improves base NERL by `+0.04483134` at `tau_score=0.4`
with exactly flat merge-oracle NERL. The 600-chunk extraction/link/sweep is live
in the background and is intentionally not reported as complete.

## What Changed

- Repaired all 600 stale `decode_v1.chunks` links and added an idempotent
  substrate validator.
- Added the pinned corrected-nucleus marker sampler using the
  `soma_recon_wholevol.py` `caff=(g*CBSCALE)//POOL` mapping, sparse per-marker
  reads, a fixed 0.95 foreground gate, and nearest-foreground snapping.
- Additively extended big-branch extraction with a caliber proxy, exact face
  centroid, schema validation, selected/all batch modes, atomic archives, and
  bounded worker control.
- Added the whole/block graph linker: exact total namespace enumeration,
  non-circular raw cue score, mutual-best ranking, best-other-partner ambiguity
  deferral, marker-set firewall/quarantine, root-wise structural assertions,
  total remap, accepted-union metadata, and one certificate per candidate.
- Additively extended the existing oracle stitch evaluator with one or more
  total remaps. `R` is applied once to the sampled decode LUT; linked base and
  `branch_merge` both consume that same remapped LUT.
- Added runnable controls, sweep instructions, acceptance rules, and the
  pending whole-volume table to the README.

## Implementation Details

The node namespace contains every distinct nonzero `(chunk_key, local_label)`;
background remains implicit `0`. Zero-union output assigns one distinct nonzero
ID per node. Linked output compresses roots to contiguous `uint32` IDs and checks
that equal global IDs are exactly equal union-find roots. The evaluator converts
the existing `sample_variant` chunk-offset namespace back to `(chunk,label)`,
hard-fails on any uncovered sampled foreground label, and never remaps oracle
output a second time.

Candidates are generated once per physical shared face within 200 nm using
actual face-footprint centroids. Face IoU comes from the cached segmentation
planes. The raw score is `(IoU + 2*tangent + caliber + perp)/5`; tangent has a
0.5 floor, IoU has exactly zero floor, and caliber has no floor. Mutual-best and
ambiguity use raw score only. Ambiguity is the best other partner divided by the
edge score and is a separate `>0.8` defer gate.

Each union root carries its marker-ID set. A pre-existing segment with two or
more markers is quarantined as a fixed singleton. Every accepted union asserts
marker-set cardinality at most one, which blocks the transitive
`A -> unmarked -> B` path. Certificates capture current-root marker sets at
decision time and use the approved decision enum, including `same_root_noop`.

## Files Changed
| File | Purpose |
|---|---|
| `dev/zebrafinch/fix_decode_v1_symlinks.py` | New S0 idempotent 600-link repair and resolution assertion. |
| `dev/zebrafinch/sample_markers.py` | New pinned S2 marker mapping, sparse sampling, numeric gate, snap, and membership cache. |
| `dev/zebrafinch/big_branch_extract.py` | Additive caliber/face-centroid schema plus selected/all batch driver and validation. |
| `dev/zebrafinch/graph_link_whole.py` | New S1/S3/S4/S5/S6 namespace, linker, firewall, remap, certificate, and controls. |
| `dev/zebrafinch/oracle_stitch_decode_v1.py` | Additive exactly-once `--remap` evaluation; multiple remaps reuse one decode sample. |
| `dev/zebrafinch/graph_link_whole.README.md` | New controls, fixed sweep, oracle gate, and acceptance documentation. |
| `dev/zebrafinch/decode_v1.chunks/chunk_*.h5` (600 symlinks) | S0 targets changed from stale `../{key}_decode_v1.h5` to `../results/{key}_decode_v1.h5`. |
| `dev/zebrafinch/results/marker_membership.npz` | S2 membership and marker-gate metadata (435 in-box marker rows). |
| `dev/zebrafinch/crossings/*.npz` | S3 schema-v2 caches; two smoke archives completed and the 600-archive background refresh is running. |
| `dev/zebrafinch/results/graph_smoke_zero.{npz,certs.csv}` | Two-chunk total zero-union remap and empty certificate control. |
| `dev/zebrafinch/results/graph_smoke_tau{04,05,06,07,08}.{npz,certs.csv}` | Two-chunk linked remaps and full candidate certificates for the bounded sweep. |
| `dev/zebrafinch/graph_link_whole_run.log` | Live whole-volume background pipeline log. |
| `.agent/features/graph_agglomeration/artifacts/code_v0.md` | This CCC coder artifact. |

## Git Baseline
run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
current_head: e8844b3da0f0a992431c901e7e2034486e7a678b

## Verification

Environment for every Python/data command:

```text
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
```

S0 symlink repair and all-target assertion (real output):

```text
S0 symlink fix: changed=600 resolved=600/600
S0 PASS: every decode_v1 symlink target resolves
```

S2 pinned marker sampling (real final output):

```text
S2 sparse sampling: marker_sites=435 decode_chunks=304
S2 marker gate: source=/projects/weilab/dataset/zebrafinch/yl_cb_80nm_neuron.h5 ids=464 n_inbox=435 n_onfg=433 ratio=0.995402
S2 markers initially off foreground: [13, 539]
S2 snapped off-fg markers: [13, 539]; unresolved: []
S2 PASS: 0.995402 >= 0.950000
saved marker membership rows=435 -> /projects/weilab/weidf/lib/pytorch_connectomics/dev/zebrafinch/results/marker_membership.npz
```

An independent slabwise scan found 464 nonzero IDs and 465 stored values when
background is included; this reconciles the current file with the plan's quoted
count without weakening the required foreground ratio.

Firewall unit controls (real output):

```text
firewall transitive A->unmarked->B rejection: PASS
firewall marker-set union<=1: PASS
firewall synthetic >=2-marker quarantine: PASS
firewall unit tests: PASS
```

The certificate-level synthetic path also produced
`['accepted', 'firewall_conflict']` for the transitive case and
`['accepted', 'accepted', 'same_root_noop']` for a three-edge same-root case.

S1 exact foreground enumeration and zero-union assertions on the real smoke
subset (real output):

```text
S1 enumerate [1/2] z2_y5_x4: foreground_labels=4534 total=4534
S1 enumerate [2/2] z2_y5_x5: foreground_labels=5703 total=10237
S1 linked structural PASS: nodes=10237 roots=10237 total_nonzero=yes root_id_iff=yes background=0
S1 zero-union structural PASS: nodes=10237 distinct_ids=10237 foreground_coverage=complete background=0
S5 firewall PASS: accepted=0 conflicts=0 quarantined=0 decisions={'firewall_conflict': 0}
```

Real big-branch smoke extraction and independent schema check:

```text
extracting 2 chunks with 1 worker(s); reusing 0 schema-valid outputs
[z2_y5_x4] 161 big branches (>= 1M vox); 244 face-crossings, 10 CLEAN (perp>=0.5, true exits)
[z2_y5_x5] 174 big branches (>= 1M vox); 280 face-crossings, 6 CLEAN (perp>=0.5, true exits)
DONE: schema OK for 2 crossing files, 524 rows
real_seconds=87.977 user_seconds=75.097 sys_seconds=12.480
schema OK: 2 crossing files, 524 rows
```

The smoke linker generated 21 candidates (20 mutual-best). Accepted unions by
threshold were 14, 14, 14, 13, and 1 for 0.4 through 0.8 respectively; all 21
candidates are present in every certificate.

Small-block end-to-end NERL used block-cropped original `test_50_skeletons`,
`local_nerl_all_chunks.crop_split(..., 1000, (20,9,9))`, and
`local_nerl_all_chunks.chunk_nerl` with `[10,10,10]`, length threshold 1000,
merge threshold 1, and canonical `branch_merge`. Bounding box was
`[2016,5040,4032]:[3024,6048,6048]`; 64 local skeletons, 6,650 nodes, 79 missing,
and `gt_len_nm=906192.480` were scored. Real table:

| variant | accepted | base NERL | oracle NERL | Delta base | Delta oracle | oracle flat (`eps=1e-4`) | realized gain |
|---|---:|---:|---:|---:|---:|---|---|
| unlinked | 0 | 0.33200301 | 0.95691795 | +0.00000000 | +0.00000000 | PASS | n/a |
| zero | 0 | 0.33200301 | 0.95691795 | +0.00000000 | +0.00000000 | PASS | no-op expected |
| tau=0.4 | 14 | 0.37683435 | 0.95691795 | +0.04483134 | +0.00000000 | PASS | PASS |
| tau=0.5 | 14 | 0.37683435 | 0.95691795 | +0.04483134 | +0.00000000 | PASS | PASS |
| tau=0.6 | 14 | 0.37683435 | 0.95691795 | +0.04483134 | +0.00000000 | PASS | PASS |
| tau=0.7 | 13 | 0.36505605 | 0.95691795 | +0.03305304 | +0.00000000 | PASS | PASS |
| tau=0.8 | 1 | 0.33200301 | 0.95691795 | +0.00000000 | +0.00000000 | PASS | no measured gain |

No-edge result (real output):

```text
NO_EDGE_CONTROL PASS eps=0.0001: base_delta=+0.00000000 oracle_delta=+0.00000000
```

The smallest smoke operating point with positive realized gain and a flat
oracle is `tau_score=0.4`.

Focused code checks:

```text
py_compile/isort/flake8: PASS
whole length-threshold contract: PASS threshold0==threshold1000 (50 skeletons, 470450 nodes)
baseline_guard: PASS (HEAD and all 24 pre-existing dirty entries/diffs unchanged)
```

A final read-only audit of all six scoped source/documentation files reported no
blocking correctness findings across namespace/remap, candidate scoring,
ambiguity, firewall/quarantine, certificates, marker mapping, and exactly-once
evaluation.

SLURM commands are installed, but `sinfo` cannot reach the controller in this
sandbox (`Error creating slurm stream socket: Operation not permitted`). The
whole-volume run therefore uses `nohup` on the current node through persistent
execution handle `session_id=86756`.

Log path:

```text
/projects/weilab/weidf/lib/pytorch_connectomics/dev/zebrafinch/graph_link_whole_run.log
```

Exact launched command:

```bash
exec nohup nice -n 10 bash -lc '
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
set -eo pipefail
cd /projects/weilab/weidf/lib/pytorch_connectomics
echo "WHOLE RUN START $(date --iso-8601=seconds) host=$(hostname)"
python -u dev/zebrafinch/big_branch_extract.py --all --workers 1
python -u dev/zebrafinch/big_branch_extract.py --all --check-schema
python -u dev/zebrafinch/graph_link_whole.py --no-edges --out dev/zebrafinch/results/graph_link_remap_zero.npz --certs dev/zebrafinch/results/graph_link_zero.certs.csv
python -u dev/zebrafinch/graph_link_whole.py --namespace-from dev/zebrafinch/results/graph_link_remap_zero.npz --tau-score 0.4 --out dev/zebrafinch/results/graph_link_remap_tau04.npz --certs dev/zebrafinch/results/graph_link_certs_tau04.csv
python -u dev/zebrafinch/graph_link_whole.py --namespace-from dev/zebrafinch/results/graph_link_remap_zero.npz --tau-score 0.5 --out dev/zebrafinch/results/graph_link_remap_tau05.npz --certs dev/zebrafinch/results/graph_link_certs_tau05.csv
python -u dev/zebrafinch/graph_link_whole.py --namespace-from dev/zebrafinch/results/graph_link_remap_zero.npz --tau-score 0.6 --out dev/zebrafinch/results/graph_link_remap.npz --certs dev/zebrafinch/results/graph_link_certs.csv
python -u dev/zebrafinch/graph_link_whole.py --namespace-from dev/zebrafinch/results/graph_link_remap_zero.npz --tau-score 0.7 --out dev/zebrafinch/results/graph_link_remap_tau07.npz --certs dev/zebrafinch/results/graph_link_certs_tau07.csv
python -u dev/zebrafinch/graph_link_whole.py --namespace-from dev/zebrafinch/results/graph_link_remap_zero.npz --tau-score 0.8 --out dev/zebrafinch/results/graph_link_remap_tau08.npz --certs dev/zebrafinch/results/graph_link_certs_tau08.csv
python -u dev/zebrafinch/oracle_stitch_decode_v1.py --remap dev/zebrafinch/results/graph_link_remap_zero.npz dev/zebrafinch/results/graph_link_remap_tau04.npz dev/zebrafinch/results/graph_link_remap_tau05.npz dev/zebrafinch/results/graph_link_remap.npz dev/zebrafinch/results/graph_link_remap_tau07.npz dev/zebrafinch/results/graph_link_remap_tau08.npz
echo "WHOLE RUN COMPLETE $(date --iso-8601=seconds)"
' > /projects/weilab/weidf/lib/pytorch_connectomics/dev/zebrafinch/graph_link_whole_run.log 2>&1 < /dev/null
```

Launch verification from the live log (real output):

```text
WHOLE RUN START 2026-07-10T00:26:39-04:00 host=a002.m31.bc.edu
extracting 600 chunks with 1 worker(s); reusing 0 schema-valid outputs
[z0_y0_x0] 142 big branches (>= 1M vox); 155 face-crossings, 1 CLEAN (perp>=0.5, true exits)
```

The coordinator can monitor the path above and collect the baseline plus
`tau_score={0.4,0.5,0.6,0.7,0.8}` table after `WHOLE RUN COMPLETE` appears.

## Review Focus

- Confirm `R` is a total collision-free function, background is fixed, the
  zero-union assertion is node-wise, linked assertions are root-wise, and
  `TotalRemap.apply` is called exactly once before both linked scores.
- Confirm raw score has no ambiguity term, ambiguity uses the best *other*
  partner, IoU floor is zero, caliber has no floor, mutual-best/tie ordering is
  deterministic, and every physical candidate is certified once.
- Confirm marker sampling remains pinned, uses the exact `caff` mapping and
  fixed 0.95 gate, and that quarantine plus component marker-set union blocks
  transitive two-nucleus joins.
- Confirm all 600 crossing archives are schema-checked before whole linking and
  that default oracle behavior is unchanged without `--remap`.
- Confirm no prohibited source files, legacy decode/oracle helpers, or the 24
  unrelated dirty paths changed.

## Risks and Unknowns

- Whole-volume NERL is pending by design. The live run must still satisfy the
  single `eps=1e-4` oracle gate; the successful smoke is not a substitute for
  the 600-box acceptance table.
- The current corrected marker HDF5 contains 464 nonzero identities (465 unique
  stored values including background), while plan prose called all 465 values
  nonzero. The pinned source and measured 0.995402 mapping gate are recorded
  explicitly rather than fabricating the stale count.
- Markerless/off-test-50 false merges remain invisible to the sparse oracle.
  The certificate table and nucleus firewall reduce risk but do not make this a
  dense global merge-safety proof.
- Adaptive snap search uses complete nested coarse-grid cubes. A pathological
  marker far from all foreground could be expensive; both actual misses resolve
  nearby and the final run completed with no unresolved markers.
- Whole evaluation retains the existing canonical `crop600` behavior rather
  than introducing local `crop_split` semantics. Remap-bearing evaluation now
  explicitly uses the 1000-nm length threshold, which was verified to preserve
  the same 50 skeletons and 470,450 nodes on current test-50.
- The whole job is attached to persistent execution handle `86756` because the
  sandbox kills an ordinary detached child; its durable progress/result surface
  is `graph_link_whole_run.log`.

## Changes Since Previous Code Version

Initial implementation.
