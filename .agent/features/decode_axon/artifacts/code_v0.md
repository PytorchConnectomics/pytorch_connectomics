# Code v0

## Overview

Implemented the plan_v2 confident-region LiCONN decoder: strong 2D section decoding, bump-safe Prob-1 carve and bidirectional extension, parallel-vetoed and caliber-matched Prob-2 merge, and deterministic drop-only crumb cleanup. Weak recovery and crumb absorption remain explicit deferred stubs. No git commit was created.

## What Changed

- Added `decode_axon.py` with affinity/mask helpers, cached strong-section loading, staged orchestration, ablation flags, array-level metric reporting, HDF5 output, full-volume NERL wiring, smoke safeguards, deferred weak/absorb APIs, and synthetic self-tests.
- Extracted the canonical constrained section decoder into `decode_v2.decode_sections`; the legacy CLI calls the public function and retains its A/B/C/substrate outputs and evaluation behavior through the same private implementation's diagnostic detail capture.
- Extracted file-I/O-free Prob-1 carve and extension cores. The legacy carve path remains available through `bump_safe=False`; the shipped path validates the host and joined orphan profiles before committing.
- Changed the Prob-2 region graph to consume caller-supplied CZYX raw affinity and added a reusable merge core with the existing z-overlap veto plus optional mean-caliber matching.
- Added `valid_tube_metric.score_array`; path-based scoring now loads the HDF5 and delegates without changing metric semantics.

## Implementation Details

`run_pipeline` loads the requested affinity slab once, computes in-plane affinity once, slices the only global strong-section cache when present, and runs `decode_sections`. The 3D order is carve, bump-safe bidirectional extend, then guarded merge. `--no-prob1` skips carve and extension together; `--no-prob2` skips the region-graph merge. The shipped merge uses `area_tol=0.5` and always retains the original-label incomplete-tube and z-overlap checks.

Safe Prob-1 carving proposes a complete watershed carve without mutating the input. It requires removal of the targeted host bump, forbids new host or joined-orphan bump runs, and tracks accepted logical-component area profiles incrementally. `bump_safe=False` retains the original immediate-mutation/relink sequence and was verified against `decode_p1.h5`.

`prob1_extend` preserves the validated static face/z-range eligibility, biggest-first orphan order, bidirectional endpoint traversal, and incrementally updated bump areas. `prob2_merge` computes channel-correct raw boundary affinity from the passed slab, applies guards to original labels, and defines caliber as voxels divided by inclusive z extent.

Drop-only cleanup maps labels below 500 voxels to background without renumbering or merging. `--weak`, `build_weak_sections`, `weak_bridge`, `bump_safe_merge`, and `--crumb-absorb` raise the plan-specified `NotImplementedError` messages.

## Files Changed

| File | Purpose |
|---|---|
| `dev/mit_liconn/decode_axon.py` | New unified confident-region decoder, CLI, orchestration, cleanup, and self-tests |
| `dev/mit_liconn/decode_v2.py` | Reusable constrained `decode_sections` core and legacy CLI delegation |
| `dev/mit_liconn/decode_v4_split.py` | Reusable `prob1_carve` with safe/default and legacy/equality paths |
| `dev/mit_liconn/tube_extend.py` | Reusable validated bump-safe `prob1_extend` core |
| `dev/mit_liconn/decode_v3_merge.py` | In-memory region graph and guarded `prob2_merge` core |
| `dev/mit_liconn/valid_tube_metric.py` | Array-level valid-tube scoring and path delegation |

## Git Baseline

run_start_ref: e8844b3da0f0a992431c901e7e2034486e7a678b
current_head: e8844b3da0f0a992431c901e7e2034486e7a678b

## Verification

`conda run -n pytc python dev/mit_liconn/decode_axon.py --self-test`

```text
per-slice area + faces  0s
  CAND host=1 bump z4-4(peak 4) excess~12  above=2(end z3,a12) below=3(start z5,a12)

1 through-intruder candidates (0 rejected by bump-safe gate)  0s
per-slice area + faces  0s

0 through-intruder candidates (0 rejected by bump-safe gate)  0s
per-slice area + faces  0s
  CAND host=1 bump z4-4(peak 4) excess~12  above=2(end z3,a9) below=3(start z5,a9)
  REJECT host=1 z4-4: bump-safe gate

1 through-intruder candidates (1 rejected by bump-safe gate)  0s
per-slice area + faces  0s
  CAND host=1 bump z4-4(peak 4) excess~12  above=2(end z3,a9) below=3(start z5,a9)

1 through-intruder candidates  0s
1 orphans to extend (touch<2 faces, >= 2000 vox); gate=bumpsafe  0s
  accept orphan 2 (1sl into [1]): bumps 1->0 {1: (1, 0), 2: (0, 0)}
extended 1 orphan ends (0 rejected by gate), 1 slices carved from 1 hosts  0s
region graph: 1 adjacent pairs  0s
merged 1 end-to-end over-splits; skipped 0 parallel-axon pairs; skipped 0 caliber-mismatched pairs (aff>=0.5, contact>=4, zoverlap<=0.5)  0s
region graph: 1 adjacent pairs  0s
merged 0 end-to-end over-splits; skipped 1 parallel-axon pairs; skipped 0 caliber-mismatched pairs (aff>=0.5, contact>=4, zoverlap<=0.5)  0s
region graph: 1 adjacent pairs  0s
merged 1 end-to-end over-splits; skipped 0 parallel-axon pairs (aff>=0.5, contact>=4, zoverlap<=0.5)  0s
region graph: 1 adjacent pairs  0s
merged 0 end-to-end over-splits; skipped 0 parallel-axon pairs; skipped 1 caliber-mismatched pairs (aff>=0.5, contact>=4, zoverlap<=0.5)  0s
crumb cleanup: dropped 1 labels / 2 voxels (< 4)
decode_axon self-test: PASS
```

Full-volume extraction equality used `/tmp/check_decode_axon_extractions.py`, which streamed each expected HDF5 in eight-slice chunks, built actual-to-expected and expected-to-actual label maps, asserted both maps stayed one-to-one, and asserted background mapped to background. Candidate and tqdm progress lines are omitted below; the pasted terminal results are:

```text
$ PYTHONPYCACHEPREFIX=/tmp/codex-pyc conda run --no-capture-output -n pytc python /tmp/check_decode_axon_extractions.py
loaded affinity + affxy in 87s
45 through-intruder candidates  12s
PASS decode_p1.h5: shape=(800, 1024, 1024) labels=2557 partition_sha256=049773868c5e5d5a92e1794b7cc6ab26ab6509c7c0930b5a86bfb2f2748795f0
region graph: 1192 adjacent pairs  58s
merged 116 end-to-end over-splits; skipped 584 parallel-axon pairs (aff>=0.5, contact>=20, zoverlap<=0.5)  58s
PASS decode_p1p2.h5: shape=(800, 1024, 1024) labels=2441 partition_sha256=165a1c4b15e0c5a0f4de4896a41de728c86a516f9ff74a4dcfdffa67815f48d5
1076 orphans to extend (touch<2 faces, >= 2000 vox); gate=bumpsafe  33s
extended 27 orphan ends (97 rejected by gate), 215 slices carved from 25 hosts  283s
PASS decode_p1p2eg.h5: shape=(800, 1024, 1024) labels=2441 partition_sha256=aa210c6e7ee92889eab85a7c87d5574388852748519ecf2802de3141498823e8
PASS decode_v2_c_constrained.h5: shape=(800, 1024, 1024) labels=2579 partition_sha256=ed75cdafe7ed80be9849eb2aabe19c206871b73262dcd2a57d18a9971f355230
all extraction checks passed in 822s
```

This establishes the requested equalities: `prob1_carve(bump_safe=False)` equals `decode_p1`, `prob2_merge(area_tol=None)` equals `decode_p1p2`, `prob1_extend` equals `decode_p1p2eg`, and `decode_sections` equals `decode_v2_c_constrained`, all as relabeling-invariant voxel partitions.

The printed `partition_sha256` values are raw-label diagnostics, not the relabeling-invariant proof. The proof is the checker's bidirectional actual-to-expected and expected-to-actual one-to-one partition maps. After the final `decode_sections` purity refactor, its public wrapper was rerun independently against the full reference volume:

```text
$ PYTHONPYCACHEPREFIX=/tmp/codex-pyc conda run --no-capture-output -n pytc python /tmp/check_decode_sections_only.py
PASS decode_v2_c_constrained.h5: shape=(800, 1024, 1024) labels=2579 partition_sha256=ed75cdafe7ed80be9849eb2aabe19c206871b73262dcd2a57d18a9971f355230
final decode_sections wrapper passed in 305s
```

`conda run -n pytc python dev/mit_liconn/decode_axon.py --zslice 0:96 --tag _smoke`

```text
SMOKE — face counts are not full-volume comparable; --full is authoritative
loaded affinity (3, 96, 1024, 1024)  9s
loaded cached strong sections sections_thr0.3_z0-800.h5[0:96]
     t1_strong: segs(z>20|vol>10000)  371 | crosses(>=2 faces)  232 (62.5%) | VALID(cross&clean&single)  227 (61.2%, vol 84.6%) | bumps 10 | PARALLEL-merge segs 0
[Prob-1 carve]
per-slice area + faces  0s
2 through-intruder candidates (1 rejected by bump-safe gate)  1s
[Prob-1 extend]
180 orphans to extend (touch<2 faces, >= 2000 vox); gate=bumpsafe  4s
extended 3 orphan ends (23 rejected by gate), 11 slices carved from 3 hosts  37s
[Prob-2 merge]
region graph: 115 adjacent pairs  5s
merged 6 end-to-end over-splits; skipped 29 parallel-axon pairs; skipped 6 caliber-mismatched pairs (aff>=0.5, contact>=20, zoverlap<=0.5)  6s
         t2_3d: segs(z>20|vol>10000)  371 | crosses(>=2 faces)  233 (62.8%) | VALID(cross&clean&single)  229 (61.7%, vol 85.1%) | bumps 6 | PARALLEL-merge segs 0
crumb cleanup: dropped 72 labels / 18638 voxels (< 500)
         final: segs(z>20|vol>10000)  371 | crosses(>=2 faces)  233 (62.8%) | VALID(cross&clean&single)  229 (61.7%, vol 85.1%) | bumps 6 | PARALLEL-merge segs 0
saved /projects/weilab/weidf/lib/pytorch_connectomics/outputs/mit_liconn/DL288B_crop1/decode_axon_smoke.h5
decode_axon completed in 92s
```

Focused checks:

```text
$ conda run -n pytc python -m py_compile dev/mit_liconn/decode_axon.py dev/mit_liconn/decode_v2.py dev/mit_liconn/decode_v4_split.py dev/mit_liconn/tube_extend.py dev/mit_liconn/decode_v3_merge.py dev/mit_liconn/valid_tube_metric.py
(no output; exit 0)
$ conda run -n pytc black --fast --check dev/mit_liconn/decode_axon.py
All done! ✨ 🍰 ✨
1 file would be left unchanged.
$ conda run -n pytc isort --check-only dev/mit_liconn/decode_axon.py
(no output; exit 0)
$ conda run -n pytc flake8 --max-line-length=100 dev/mit_liconn/decode_axon.py
(no output; exit 0)
$ conda run -n pytc python dev/mit_liconn/decode_v2.py --self-test
decode_v2 self-test: PASS
$ conda run -n pytc python -c "... inspect/AST contract check ..."
(seg2d, affxy, *, fuse_iou=0.2, area_tol=0.5, next_overlap=0.5, min_len=3)
run_pipeline_calls_decode_sections= True
$ conda run -n pytc black --fast --check dev/mit_liconn/decode_axon.py dev/mit_liconn/decode_v2.py
would reformat dev/mit_liconn/decode_v2.py
1 file would be reformatted, 1 file would be left unchanged. (exit 1)
```

`git status --short` exactly matches `state/run_start.status` (`cmp` exit 0), and `HEAD` remains the run-start commit.

## Review Focus

- Confirm the new Prob-1 logical-component profile tracker enforces target-host bump removal and no new joined-orphan bump while leaving `bump_safe=False` on the behavior-equivalent legacy sequence verified as the same voxel partition.
- Check that the shipped 3D order and toggles are exactly carve then extend then merge, and that `area_tol=0.5` is additive to—not a replacement for—the original incomplete-tube and z-overlap guards.
- Review the `decode_sections` ownership split: its exact public wrapper is pure and returns only constrained `uint32` C, while the legacy CLI calls it and reruns the same private implementation with explicit A/B/C/substrate detail capture.
- Confirm full-cache slicing and in-memory affinity transport preserve global section labels and correct raw-affinity z offsets for slab runs.

## Risks and Unknowns

- Per instruction, the unified `--full` 800-slice pipeline was not run. Its final valid-tube gate and NERL remain unverified; only extraction fidelity and the required 96-slice smoke run were executed.
- Slab face counts are inherently non-authoritative and are labeled SMOKE in the CLI. The printed 0:96 valid-tube percentages cannot be compared to full-volume baselines.
- The shipped Prob-2 caliber gate is new relative to `decode_p1p2`; it can reject legitimate end-to-end joins. The full-volume quality impact must be evaluated by the coordinator.
- `prob1_extend` and `prob2_merge` retain their source scripts' assumption that the segmentation has at least one foreground label; an all-background input still fails in the inherited bounding-box helper. This does not affect the supplied cache or required verification volumes.
- Preserving both the exact, pure `decode_sections(...)` return contract and all legacy A/B/C/substrate diagnostics requires the legacy `decode_v2` CLI to execute the shared core twice. This does not affect `decode_axon`, but increases legacy CLI runtime and peak memory versus its pre-refactor implementation.
- The combined Black check reports one pre-existing assertion layout outside the changed `decode_v2.py` hunk. It was not reformatted to avoid unrelated churn; the new `decode_axon.py` passes Black, isort, and flake8.
- Weak recovery and crumb absorption are intentionally unavailable and raise `NotImplementedError`.
- `dev/` and `.agent/` are gitignored, so normal `git diff`/`git status` do not expose these source and artifact changes. Pre-existing tracked/untracked user changes remained untouched and status-identical to the CCC start snapshot.

## Changes Since Previous Code Version

Initial implementation.
