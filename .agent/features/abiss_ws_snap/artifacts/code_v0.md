# Code v0
## Overview

Implemented nucleus-mask snapping in the atomic watershed and transported the resulting nucleus
equivalence through every watershed hierarchy level. Multi-nucleus basins are split by a
deterministic affinity-priority seeded watershed; every tagged basin is then materialized as a
must-link in the same DSU used for atomic compaction and, separately, before hierarchy remap
emission. The no-`NUC_PATH` invariance gate passed.

No commit was created. The sibling run's global-nucleus-table working-tree changes were preserved
as the run baseline.

## What Changed

- `cut_chunk_ws.py` now creates a shape-checked `uint32` `nuc.raw` using the existing
  `nucleus_utils` coordinate transform, only when `NUC_PATH` is configured.
- Atomic watershed output tags every single-nucleus basin, splits every multi-nucleus basin inside
  its own bounding box, re-densifies labels, and independently reconstructs complete count words
  and boundary bits from the post-split labels.
- Nucleus IDs live in a packed side-channel record rather than the watershed label namespace, so
  the two ID spaces cannot collide.
- Atomic agglomeration performs the tag-group closure after ordinary `try_merge` processing and
  before `remaps`; tagged components are dust-exempt and cross-tag edges cannot enter the emitted
  dendrogram.
- Hierarchy processing loads child tag records, guards plateau/descent/ordinary/MST union sites,
  performs the same closure before building remaps, and aborts on conflicting records for one ID.
- Tagged interior components remain transportable below the top hierarchy level, including the
  zero-face path, and are finalized into `done_pre`/`done_post` at the top level. This prevents a
  correct tag from remaining permanently `ongoing` and missing final `ws3` remapping.
- Added `ABISS_WS_NUC_GUARD`, default `ON`. The `OFF` build disables only cross-tag compatibility
  checks; split inputs, tags, closures, and dust behavior remain enabled.

## Implementation Details

The seeded split uses every mask voxel as a fixed marker and floods only voxels belonging to the
original basin. Candidates are ordered by maximum bottleneck affinity, then nucleus ID and voxel
index for deterministic ties. Unreached basin voxels receive one untagged label. The code comments
state that this cut has no image support: the measured bottleneck is approximately 0.999 against a
0.3 merge threshold, so imposing it is defensible only because the mask is confirmed correct and
conservative.

`nuc_union` adds real sizes, ORs `on_border`, clears the losing count/tag, links the DSU, and moves
both count and tag to the representative actually chosen by Boost. Ordinary tagged+untagged unions
use the same ownership rule. Multi-threshold execution takes a fresh tag vector copy beside each
segmentation, region graph, and count copy.

At hierarchy levels, non-top tagged components are emitted in the count/tag/ongoing streams even
without physical boundary contact, allowing disconnected same-tag pieces to meet at a later
parent. The top-level flag added to `param.txt` changes only their final classification: after the
top closure they become real done remaps instead of an unused terminal side channel. A single-child
top level is processed rather than lifted so it receives the same finalization.

The existing atomic and composite shell tar globs already include `nuc_tag_<tag>.data`; no shell
change was needed to carry the new file. `remap_chunk_ws.sh` consumes the materialized hierarchy
remaps through its existing `merge_remaps.py -> ws3 -> upload_chunk.py` path, so the tag side file
is not required after hierarchy completion.

## Files Changed

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Add the default-ON guard seam and focused ON/OFF test targets. |
| `scripts/cut_chunk_ws.py` | Cut and validate the nucleus mask into the padded atomic watershed coordinate frame. |
| `scripts/merge_chunks_ws.py` | Merge/lift tag files, process a single-child top level, and pass top-level finalization state to `ws2`. |
| `src/ws/nucleus_tags.hpp` | Define the collision-free packed tag wire format and guard compatibility algebra. |
| `src/ws/nucleus_snap.hpp` | Implement basin tagging, seeded splitting, deterministic re-densification, and complete count/border reconstruction. |
| `src/ws/agglomeration.hpp` | Add representative-safe tagged unions, atomic tag closure, guard checks, tag compaction, dust protection, and MST vetoes. |
| `src/ws/atomic_chunk.cpp` | Load `nuc.raw`, invoke snapping before region-graph construction, copy tags per threshold, and emit tag records. |
| `src/ws/merge_chunks.cpp` | Consume/emit hierarchy tags, guard all four union sites, close tags before remaps, protect tagged dust, handle zero-face work, and finalize at the top. |
| `work/test/test_ws_nucleus.cpp` | Check split count words, independent border oracle, same-tag unconditional closure, dust survival, and atomic guard ON/OFF. |
| `work/test/test_ws_nucleus_atomic.py` | Exercise HDF5 mask ingress, the real `ws` binary, split/tag output, and multi-threshold tag copies. |
| `work/test/test_ws_nucleus_hierarchy.py` | Exercise zero-face closure, child/parent versus monolithic identity, parent tag/remap inspection, bridge guards, `ws3`, chunkmap, global-table, and `acme` tracing. |
| `scripts/cut_chunk_agg.py`, `scripts/set_env.py`, `src/seg/NucExtractor.hpp`, `src/seg/atomic_chunk_ME.cpp`, `scripts/nucleus_utils.py`, `scripts/build_nucleus_table.py`, and sibling tests | Pre-existing global-nucleus-table baseline consumed or preserved by this run; not reverted or re-owned here. |

## Git Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
current_head: 312bf54183faa94be1aff8ae246fcc211808f0bb

## Verification

- **V1 — PASS.** Command:
  `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u; export HDF5_USE_FILE_LOCKING=FALSE; work/test/run_v2_invariance.sh`
  Result: `NUC_TABLE unset` and `NUC_TABLE set without NUC_PATH` each reported
  `identical=38 differing=0 missing=0`; both reported zero extra files and zero nucleus log lines;
  stale nucleus environment passed; final line `run_v2_invariance: PASS`.
- **V2 — PASS.** Command:
  `python work/test/test_ws_nucleus_atomic.py --repo . --ws build/ws`
  Result: HDF5 mask ingress produced the padded `nuc.raw`; one real watershed basin split into two
  single-tag pieces; mask seeds mapped to different IDs; full counts summed to the output volume;
  both multi-threshold outputs retained tags `{11,22}`. Command `build/test_ws_nucleus` also passed
  the independent post-split count-word/border oracle, same-tag closure below the ordinary edge
  threshold, and tagged dust survival.
- **V3 — PASS.** Commands:
  `build/test_ws_nucleus && build/test_ws_nucleus_guard_off` and
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DABISS_WS_NUC_GUARD=OFF && cmake --build build --parallel 4 --target ws2 && python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 0`.
  Result: atomic tests passed with `guard=1` and `guard=0`; the real OFF `ws2` build merged the
  tag-to-untagged-to-other-tag plateau bridge. The tree was then reconfigured and rebuilt with
  `ABISS_WS_NUC_GUARD=ON`; ON kept two tag components. Same-tag closure merged components whose
  sizes exceeded the ordinary size threshold and whose edge was below the merge threshold.
- **V4 — PASS.** Command:
  `python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 1 --trace`.
  Result: zero-face same-tag IDs became remaps before emission; hierarchy-level tagged dust
  survived; child+parent identity equaled monolithic identity; parent tag records and
  `done_pre`/`done_post` remaps were inspected; a tag-to-untagged bridge did not cross-stitch the
  second tag.
- **V5 — PASS.** Same command as V4 with `--trace`.
  Result: known IDs were followed through parent remaps, `ws3`, `chunkmap.data`, the existing
  global nucleus-table reducer, and `acme`; the final records were two `PROPER` records mapping
  nuclei `{7,8}` to one supervoxel each. This fixture's chunkmap was the valid empty identity map
  because the chosen final representatives were already raw watershed IDs.
- **V6 — NOT RUN.** `/usr/bin/sbatch` is available, but this allowed worktree has no
  `dev/zebrafinch/nuc_z1_y7_x6` crop arm or `w2ctl`/`wssnap` inputs. Per the task, the coordinator
  must run the SLURM crop A/B. Therefore the eight-nucleus mass preservation, cannot-link and
  must-link invariants, 110,244-to-0 shared mass, 95% coverage, dominance, largest segment/root
  count, wall-clock, and whether real split pieces were eligible for ordinary refusion are all
  **NOT RUN / unknown**, not claimed.
- **Input contracts — PASS.** Command:
  `python work/test/test_input_contracts.py` with `HDF5_USE_FILE_LOCKING=FALSE`.
  Result: accepted integer/identity/resolution-aligned nucleus inputs, rejected oversized/float/
  negative/wrong-shape inputs, verified set-env gating, and ended `test_input_contracts: PASS`.
- **Build/source checks — PASS.** Command:
  `cmake --build build --parallel 4 --target ws ws2 ws3 acme test_ws_nucleus test_ws_nucleus_guard_off`
  completed, followed by both C++ test binaries and `git diff --check`. Focused Python checks ran
  Black, black-profile isort, and flake8 on the new test files, plus `py_compile` on the two changed
  pipeline scripts. Final output: `final build/status gate: PASS`; HEAD remained the run start SHA.

## Review Focus

- Confirm the atomic closure remains between the ordinary region-graph merge loop and `remaps`,
  and the hierarchy closure remains before its `remaps` vector and `done_pre`/`done_post` output.
- Inspect tag/count ownership after Boost chooses a representative, especially ordinary
  tagged+untagged unions and hierarchy plateau/descent unions.
- Check the below-top transport versus top finalization contract, including single-child and
  `face_size == 0` paths, for consistency with production hierarchy JSONs' `top_mip_level`.
- Check the seeded flood's affinity indexing, deterministic tie order, bbox restriction, and
  post-split boundary oracle against the watershed's Fortran-order coordinate convention.
- Verify that OFF only bypasses `ws_nuc_tags_compatible` and does not disable split/tag inputs,
  closure, or dust protection.

## Risks and Unknowns

- The seeded cut is deliberately unsupported by image evidence (approximately 0.999 bottleneck
  against 0.3); correctness depends on the accepted premise that the mask is correct and
  conservative. A bleeding mask can impose an unconditional false must-link.
- V6 was not run, so crop-level quality, runtime, runaway/shatter limits, and ordinary-refusion
  eligibility remain unknown. No claim is made that the feature fixes `worst3` until that A/B runs.
- The per-conflict flood allocates arrays over the original basin bounding box. Atomic chunk size
  bounds this, but a large sparse bbox can cost more memory than the basin voxel count alone.
- No terminal-plan requirement was dropped. One necessary hierarchy detail was made explicit while
  implementing: tagged interior components must be carried below the top but finalized at the top.
  Omitting that distinction would satisfy local tag tests while stranding final remaps—the exact
  fourth no-op risk identified in review.

## Changes Since Previous Code Version

Initial implementation.
