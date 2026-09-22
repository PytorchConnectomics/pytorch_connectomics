# Code v1
## Overview

Fixed every review_v0 finding in the detached `work2/abiss` tree. Hierarchy plateau and descent
unions are structural again and therefore unconditional; when either structural mechanism joins
two distinct real nucleus tags, the component now carries a persistent `0xFFFFFFFF` `CONFLICT`
tag instead of losing both identities or aborting later through a broken `descent[]` invariant.

`CONFLICT` is deliberately quarantined at watershed policy sites: with the guard enabled it cannot
merge with a real tag, an untagged component, or another conflict, and it is excluded from the
tag-group closure. This is stricter than the downstream agglomeration rule that permits
`CONFLICT + NONE`, because a conflict already proves that the watershed component contains an
unresolved cannot-link violation and must not become a growth seed.

No commit was created. The sibling run's global-nucleus-table baseline changes remain intact. V6
was not run because it requires the coordinator's SLURM crop A/B environment.

## What Changed

- Restored unconditional unions in both hierarchy structure passes: the `same` plateau loop and
  the `descent[v1] == val || descent[v2] == val` loop. Cross-tag vetoes remain only in
  `try_merge`, the tag-group policy closure, and MST construction.
- Added a reserved `ws_nuc_conflict == 0xFFFFFFFF` state. Differing real tags join to conflict;
  conflict absorbs every tag and is emitted and reloaded unchanged at later hierarchy levels.
- Made conflict incompatible with every policy merge in guard-ON builds and excluded it from
  same-tag closure, preventing the `CONFLICT + NONE` runaway-growth pattern.
- Added `ws nuc: conflict components N` per hierarchy chunk. It counts transitions caused by
  differing real tags joining, including duplicate real-tag records for one loaded component.
- Reserved `0xFFFFFFFF` at mask ingress and added a C++ defense-in-depth rejection so it cannot
  collide with a real nucleus ID.
- Preserved `nuc_tag.data` after normal reads and changed the empty-chunk pass-through from rename
  to copy, leaving the input available for reruns and post-mortem inspection.
- Added nucleus tags to both pre-existing `descent[]` invariant abort diagnostics.
- Reworked the hierarchy guard fixture so its bridge exercises policy edges only, then added
  independent cross-tag plateau and descent fixtures at `high_threshold=0.99999`, conflict-policy
  assertions, input preservation, conflict logging, and child-to-parent conflict transport.

## Implementation Details

The tag algebra now has three classes: `NONE` (`0`), real nucleus IDs, and `CONFLICT`
(`0xFFFFFFFF`). `join(real_a, real_b)` returns conflict when the IDs differ, and any join involving
conflict returns conflict. The packed wire record is unchanged at 12 bytes, so conflict uses the
existing tag field and requires no format migration. Python mask validation rejects the reserved
value before writing `nuc.raw`; `snap_nucleus_basins` rejects it again if a raw file bypasses that
validator.

At hierarchy structure sites, the code captures the two root tags for conflict accounting, calls
the representative-safe `nuc_union` unconditionally, and then performs the original plateau or
descent reconciliation. This preserves the watershed's required `descent[]` bookkeeping. The
ordinary region-graph `try_merge` and final MST continue to call `ws_nuc_tags_compatible`.

The guard-ON policy treats every conflict pairing as incompatible, including
`CONFLICT + NONE` and `CONFLICT + CONFLICT`. The tag-group closure processes only real tags, so the
sentinel cannot accidentally turn all conflicted components into one must-link group. Dust
protection and all hierarchy count/tag/ongoing emission conditions still treat conflict as nonzero,
which keeps it alive through remapping. Duplicate records for one supervoxel are joined through the
same algebra rather than aborting or choosing one identity.

The new structural fixture has two tagged basins and one untagged neighbor. Its plateau form uses a
face affinity equal to `0.99999`; its descent form uses `0.99998`, below the high threshold but
exactly equal to the recorded descent. Both tagged basins must structurally merge to one conflict.
With the guard on, the conflict cannot absorb the ordinary untagged neighbor; with the guard off,
that policy edge is allowed. A non-top plateau case is then fed to a parent and must re-emit the
same conflict sentinel.

The structural-versus-policy distinction does not create another exception inside atomic
`merge_segments`. That function has no `same` plateau pass, no `descent[]` array, and no
plateau-equivalent invariant bookkeeping. Its sequence is the ordinary `try_merge` policy loop,
the explicit real-tag must-link closure, remap/compaction, and policy-side MST construction. The
closure therefore remains correctly placed between `try_merge` and `remaps`; it skips conflict but
still unconditionally closes components carrying the same real nucleus ID.

## Files Changed

| File | Purpose |
|---|---|
| `src/ws/nucleus_tags.hpp` | Define the reserved conflict sentinel, absorbing join lattice, real-tag predicate, conflict-transition predicate, and strict guard-ON policy. |
| `src/ws/merge_chunks.cpp` | Restore structural unions, count and transport conflicts, retain tag inputs, print diagnostic tags, and preserve empty-chunk inputs by copying. |
| `src/ws/agglomeration.hpp` | Prevent the real-tag closure from treating all conflict sentinels as one nucleus group; retain strict policy checks in atomic `try_merge` and MST. |
| `src/ws/nucleus_snap.hpp` | Reject the reserved sentinel if a `nuc.raw` bypasses Python validation. |
| `scripts/nucleus_utils.py` | Reject real nucleus ID `0xFFFFFFFF` at shared mask ingress. |
| `work/test/test_ws_nucleus.cpp` | Test the conflict lattice, guard-ON quarantine, guard-OFF seam, and persistent conflict tag after compaction. |
| `work/test/test_ws_nucleus_hierarchy.py` | Add policy-only bridge coverage, structural plateau/descent conflict fixtures, retained-input checks, per-chunk counter checks, and hierarchy transport. |
| `work/test/test_input_contracts.py` | Assert that the reserved conflict ID is rejected as input. |

## Git Baseline

run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
current_head: 312bf54183faa94be1aff8ae246fcc211808f0bb

## Verification

- **V1 — PASS.** Command:
  `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u; export HDF5_USE_FILE_LOCKING=FALSE; work/test/run_v2_invariance.sh`
  Result: both `NUC_TABLE` unset and `NUC_TABLE` set without `NUC_PATH` reported
  `identical=38 differing=0 missing=0`, zero extra files, and zero nucleus log lines; stale nucleus
  environment passed; final line `run_v2_invariance: PASS`.
- **V2 — PASS.** Commands: `build/test_ws_nucleus` and
  `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u; export HDF5_USE_FILE_LOCKING=FALSE; python work/test/test_ws_nucleus_atomic.py --repo . --ws build/ws`.
  Result: the guard-ON C++ lattice/split/count/closure/dust tests passed; the real atomic binary
  split one basin into tags `{11,22}`, preserved complete counts, and retained both tags in both
  multi-threshold outputs. An earlier direct-environment invocation was not counted because its
  child `python` resolved outside `pytc` and lacked `cloudvolume`.
- **V3 — PASS.** Command:
  `build/test_ws_nucleus && build/test_ws_nucleus_guard_off && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DABISS_WS_NUC_GUARD=OFF && cmake --build build --parallel 4 --target ws2 && /projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 0`.
  Result: both C++ binaries passed. The real guard-OFF `ws2` merged the policy-only
  tag-to-untagged-to-other-tag bridge and retained `CONFLICT`; the plateau and descent structure
  cases also completed and retained conflict rather than being vetoed.
- **V4 — PASS.** Command:
  `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DABISS_WS_NUC_GUARD=ON && cmake --build build --parallel 4 --target ws2 ws3 acme && /projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 1`.
  Result: guard ON was restored; same-tag closure, dust, child/parent identity, and the policy
  bridge passed. Both new structural fixtures completed, reported one conflict, emitted
  `0xFFFFFFFF`, kept the ordinary untagged neighbor out, retained `nuc_tag.data`, and the plateau
  conflict survived a child-to-parent transport with a zero-new-conflicts parent log.
- **V5 — PASS.** Command:
  `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u; export HDF5_USE_FILE_LOCKING=FALSE; python work/test/test_ws_nucleus_hierarchy.py --repo . --ws2 build/ws2 --guard 1 --trace`.
  Result: known IDs passed through hierarchy remaps, `ws3`, the empty identity `chunkmap.data`,
  the global nucleus-table reducer, and `acme`; final line
  `V5 id trace: hierarchy remaps -> ws3 -> chunkmap -> global table -> acme PASS`.
- **New plateau/descent regression fixture — PASS.** Exercised by both V3 and V4 commands. The
  plateau pair meets at `0.99999 >= high_threshold`; the descent pair meets at `0.99998`, exactly
  its recorded descent. `ws2` completed in both cases, the tagged IDs shared one conflict result,
  and guard ON prevented subsequent policy growth into `NONE`.
- **V6 — NOT RUN.** It requires the coordinator's SLURM crop A/B. No crop masses, cannot-link or
  must-link counts, coverage, dominance, largest-segment/root-count measurements, wall-clock, or
  real-data conflict frequency are claimed.
- **Input contracts — PASS.** Command:
  `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u; export HDF5_USE_FILE_LOCKING=FALSE; python work/test/test_input_contracts.py`.
  Result: all prior contracts passed and the new `reserved_conflict` case rejected
  `0xFFFFFFFF` with the expected message; final line `test_input_contracts: PASS`.
- **Build/status/source gate — PASS.** Command:
  `cmake --build build --parallel 4 --target ws ws2 ws3 acme test_ws_nucleus test_ws_nucleus_guard_off && build/test_ws_nucleus && build/test_ws_nucleus_guard_off && /projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/isort --check-only --profile black work/test/test_ws_nucleus_hierarchy.py work/test/test_input_contracts.py && /projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/flake8 --max-line-length=100 work/test/test_ws_nucleus_hierarchy.py work/test/test_input_contracts.py && /projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python -m py_compile work/test/test_ws_nucleus_hierarchy.py work/test/test_input_contracts.py scripts/nucleus_utils.py && git diff --check && test "$(git rev-parse HEAD)" = 312bf54183faa94be1aff8ae246fcc211808f0bb && rg '^ABISS_WS_NUC_GUARD:BOOL=ON$' build/CMakeCache.txt`.
  Result: all targets built, both C++ binaries passed, source checks passed, HEAD stayed fixed, the
  final cache was guard ON, and the final line was `final build/status gate: PASS`.
- **Black formatting — PASS through the serial API.** Command:
  `/projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python -c 'from pathlib import Path; import black; files=[Path("work/test/test_ws_nucleus_hierarchy.py"),Path("work/test/test_input_contracts.py")]; changed=[black.format_file_in_place(path,fast=False,mode=black.FileMode(),write_back=black.WriteBack.CHECK) for path in files]; assert not any(changed), changed; print("Black serial API check: PASS")'`.
  Result: `Black serial API check: PASS`. The equivalent Black CLI reported both files unchanged
  but failed to terminate under the sandbox and was killed by `timeout` with status 124, so that
  CLI attempt is not claimed as a pass.

## Review Focus

- Confirm the plateau and descent loops contain no nucleus compatibility veto and still reconcile
  sizes, representative-owned tags, and `descent[]` through `nuc_union`.
- Review the deliberate conflict quarantine: guard-ON `try_merge` and MST reject every pairing
  involving conflict, and tag-group closure accepts only real nucleus IDs.
- Check that conflict remains dust-exempt, is written to the unchanged packed record, is absorbed
  on reload/join, and cannot decay to zero through representative swaps or hierarchy transport.
- Confirm the counter semantics: it counts real-tag mismatch transitions in the current chunk,
  while an already-conflicted child transported through a parent produces zero new transitions.
- Inspect the new fixtures' face encoding to ensure the policy bridge has no outer-face match, the
  plateau case is at the high threshold, and the descent case has one matching outer endpoint.
- Verify `nuc_tag.data` remains after both normal processing and the empty-work pass-through.

## Risks and Unknowns

- V6 remains NOT RUN, so the real `1_0_2_2` chunk's completion and the real-data conflict count are
  unknown until the coordinator reruns the crop.
- Strict conflict quarantine is intentionally conservative. Once a structural conflict exists,
  ordinary image-supported watershed merges cannot grow that component; this may preserve more
  components than the previous policy, but prevents a known-invalid component from absorbing
  untagged mass.
- The `ws nuc:` counter records conflict-creation transitions, not the number of distinct conflict
  components surviving at output; later structural unions can combine already-conflicted roots.
- `0xFFFFFFFF` is now unavailable as a real nucleus ID. Both shared ingress and the atomic C++ path
  reject it explicitly.
- The Black CLI nontermination appears environment-specific; its serial formatting API, isort,
  flake8, and `py_compile` all completed successfully.

## Changes Since Previous Code Version

- **review_v0 major: structural veto placement — fixed.** Removed the tag compatibility condition
  from both plateau and descent unions. `try_merge` and MST remain the discretionary guard sites.
- **review_v0 major: cross-tag structural result was silent — fixed.** Added the persistent
  `CONFLICT` sentinel and absorbing join algebra, per-chunk transition counter, unchanged wire
  transport, input collision rejection, and an explicit strict policy that blocks every
  conflict-involving policy merge and excludes conflict from tag-group closure.
- **review_v0 major: hierarchy fixture missed plateau/descent — fixed.** Added direct cross-tag
  plateau and descent face cases, completion/remap/tag/counter assertions, conflict-versus-`NONE`
  policy checks, retained-input assertions, and child-to-parent transport. The old bridge was
  corrected to be policy-only so V3 still tests `try_merge`/MST guard behavior rather than
  structure behavior.
- **review_v0 minor: `nuc_tag.data` deleted on read — fixed.** Removed the loader deletion and made
  empty-work pass-through copy instead of rename.
- **review_v0 minor: invariant abort omitted tags — fixed.** Both `This should not happen in a/b`
  messages now print the two endpoint nucleus tags.
- **review_v0 question: atomic structural-versus-policy distinction — confirmed not applicable.**
  Atomic `merge_segments` has no plateau/descent bookkeeping. Its real-tag closure remains
  policy-side between `try_merge` and `remaps`, and conflict is deliberately excluded from it.
