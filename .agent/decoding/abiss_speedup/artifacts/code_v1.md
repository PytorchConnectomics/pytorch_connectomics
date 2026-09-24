# Code v1
## Overview

Addressed all findings from `review_v0` in the isolated `work/abiss` copy and the
run-local profiling harness. The default `ABISS_ARCH=""` configuration now receives no
`-ffp-contract` option, so the FP behavior is unchanged by construction. A non-empty
`ABISS_ARCH` adds C++-only `-ffp-contract=off`, which is the opt-in case where a selected
architecture could enable FMA.

Both isolated deployable build variants were reconfigured and rebuilt successfully. All 17
system targets and all 17 mimalloc targets passed the asserted flag/definition audit. No
commit was created, no file below `work/abiss/scripts/` was modified, and no live build was
configured or rebuilt.

## What Changed

- Moved `-ffp-contract=off` inside the non-empty `ABISS_ARCH` branch and guarded it with
  `$<COMPILE_LANGUAGE:CXX>`.
- Added `work/abiss/.gitignore` entries for the three generated build trees.
- Added physical-path guards to the manifest capture, ME/WS composite copies, and ABISS
  profiling command. Paths resolving outside the CCC run or isolated repo are rejected
  before source or execution.
- Added `state/profiling/test_run_local_isolation.sh`, which positively resolves the
  run-local binaries and negatively tests live-checkout repository and build overrides.
- Documented the isolation contract and its test in the profiling README.
- Reconfigured and rebuilt the system and mimalloc variants, and regenerated the
  configure-only non-empty-architecture audit tree.

## Implementation Details

`ABISS_ARCH` remains empty by default. In that branch CMake leaves the inherited
architecture flags alone and does not add `-ffp-contract=off`. The generated default
`flags.make` files contain the inherited `-march=nocona -mtune=haswell` and no FP
contraction directive. When `ABISS_ARCH` is non-empty, CMake removes inherited
`-march`/`-mtune` tokens from the directory-scope `CMAKE_CXX_FLAGS`, appends exactly one
`-march=${ABISS_ARCH}`, and adds C++-only `-ffp-contract=off`.

The architecture rewrite mutates the directory-scope variable, not the cache entry.
Consequently, `CMakeCache.txt` can retain unstripped `-march` and `-mtune` values while
generated `flags.make` files contain the sanitized effective flags. Phase 0.0 captures
both files, so audits must treat `flags.make` as the effective per-target evidence. The
rewrite intentionally strips only `-march` and `-mtune`; it does not strip independent
`-mfma` or `-mavx*` options.

The `ABISS_ALLOCATOR=system` default is justified by the coordinator evidence recorded in
`review_v0`: the live production build defined neither `USE_MIMALLOC` nor
`USE_ABSL_HASHMAP`, and `agg`, `meme`, `ws2`, and `acme` each linked zero allocator
libraries. Therefore `system` matches the production allocator configuration and no
`auto` mode is needed. This stage did not re-investigate that resolved finding.

The mimalloc build uses `<RUN>/deps/lib64/pkgconfig`. This is within the dependency
approval recorded in `run.md` under `## Approvals`, user direction dated 2026-07-29:
“approve dependencies and continue to code.” Neither `pytc` nor `base` was modified.

The composite harness now resolves physical paths for the run folder, ABISS repository,
source scripts, build directory, and each possible binary before sourcing `init.sh`.
`WORKER_HOME` is fixed to the validated isolated repository. The profiling wrapper
receives that repository identity and independently rejects a command whose physical
binary path is outside it. The manifest helper applies the same run/repository checks.
Symlinks resolving to the live checkout therefore fail the physical-path checks.

## Files Changed
| File | Purpose |
|---|---|
| `work/abiss/CMakeLists.txt` | Make FP contraction control opt-in with `ABISS_ARCH` and C++-only; retain architecture and allocator controls. |
| `work/abiss/.gitignore` | Prevent the three generated build trees from being swept into a later `git add`. |
| `state/profiling/capture_baseline_manifest.sh` | Require the repository, build, and captured binaries to resolve inside the run-local isolated copy. |
| `state/profiling/composite_chunk_me.sh` | Resolve and validate the isolated source/build paths and all seven possible ME binaries before sourcing or execution. |
| `state/profiling/composite_chunk_ws.sh` | Resolve and validate the isolated source/build paths and `ws2` before sourcing or execution. |
| `state/profiling/profile_command.sh` | Independently reject profiled ME/WS binaries outside the validated run-local repository. |
| `state/profiling/test_run_local_isolation.sh` | Add positive physical-path checks and negative live-checkout rejection tests. |
| `state/profiling/README.md` | Document the enforced isolation boundary and test command. |
| `work/abiss/build-system/` | Regenerated ignored Release/system build with `EXTRACT_SIZE=ON` and empty `ABISS_ARCH`. |
| `work/abiss/build-mimalloc/` | Regenerated ignored Release/mimalloc build from the approved run-local dependency prefix. |
| `work/abiss/build-arch-sanitize/` | Regenerated ignored configure-only tree with non-empty `ABISS_ARCH=nocona`. |

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
current_head: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

## Verification

The following configurations and builds completed in `work/abiss`:

```text
conda run -n pytc cmake -S . -B build-system \
  -DCMAKE_BUILD_TYPE=Release -DEXTRACT_SIZE=ON \
  -DABISS_ALLOCATOR=system -DABISS_ARCH= \
  -DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE
conda run -n pytc cmake --build build-system --parallel 8

PKG_CONFIG_PATH=<RUN>/deps/lib64/pkgconfig conda run -n pytc cmake \
  -S . -B build-mimalloc -DCMAKE_BUILD_TYPE=Release -DEXTRACT_SIZE=ON \
  -DABISS_ALLOCATOR=mimalloc -DABISS_ARCH= \
  -DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE
conda run -n pytc cmake --build build-mimalloc --parallel 8

conda run -n pytc cmake -S . -B build-arch-sanitize \
  -DCMAKE_BUILD_TYPE=Release -DEXTRACT_SIZE=ON \
  -DABISS_ALLOCATOR=system -DABISS_ARCH=nocona \
  -DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE
```

The asserted audit reported:

```text
build-system audited=17 failures=0
build-mimalloc audited=17 failures=0
build-arch-sanitize audited=17 failures=0
build-system mimalloc-linked=0 jemalloc-linked=0
build-mimalloc mimalloc-linked=17 jemalloc-linked=0
```

The full deployable-build per-target flag/definition table is:

| Build | Target | `-march` | `-ffp-contract=off` | `EXTRACT_SIZE` | unsafe FP | `USE_ABSL_HASHMAP` | `USE_MIMALLOC` |
|---|---|---:|---:|---:|---:|---:|---:|
| system | accs | 1 | 0 | 1 | 0 | 0 | 0 |
| system | acme | 1 | 0 | 1 | 0 | 0 | 0 |
| system | agg | 1 | 0 | 1 | 0 | 0 | 0 |
| system | agg_extra | 1 | 0 | 1 | 0 | 0 | 0 |
| system | agg_nonoverlap | 1 | 0 | 1 | 0 | 0 | 0 |
| system | agg_overlap | 1 | 0 | 1 | 0 | 0 | 0 |
| system | assort | 1 | 0 | 1 | 0 | 0 | 0 |
| system | evaluate | 1 | 0 | 1 | 0 | 0 | 0 |
| system | match_chunks | 1 | 0 | 1 | 0 | 0 | 0 |
| system | mecs | 1 | 0 | 1 | 0 | 0 | 0 |
| system | meme | 1 | 0 | 1 | 0 | 0 | 0 |
| system | reduce_chunk | 1 | 0 | 1 | 0 | 0 | 0 |
| system | size_map | 1 | 0 | 1 | 0 | 0 | 0 |
| system | split_remap | 1 | 0 | 1 | 0 | 0 | 0 |
| system | ws | 1 | 0 | 1 | 0 | 0 | 0 |
| system | ws2 | 1 | 0 | 1 | 0 | 0 | 0 |
| system | ws3 | 1 | 0 | 1 | 0 | 0 | 0 |
| mimalloc | accs | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | acme | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | agg | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | agg_extra | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | agg_nonoverlap | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | agg_overlap | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | assort | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | evaluate | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | match_chunks | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | mecs | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | meme | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | reduce_chunk | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | size_map | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | split_remap | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | ws | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | ws2 | 1 | 0 | 1 | 0 | 0 | 1 |
| mimalloc | ws3 | 1 | 0 | 1 | 0 | 0 | 1 |

Here `unsafe FP` counts
`-ffast-math|-Ofast|-funsafe-math-optimizations`. The configure-only non-empty
architecture tree separately showed one `-march`, zero `-mtune`, and one
`-ffp-contract=off` on each of 17 targets. Its cache/effective-flags comparison was:

```text
CMakeCache.txt:
ABISS_ARCH:STRING=nocona
CMAKE_CXX_FLAGS:STRING=... -march=nocona -mtune=haswell ...

CMakeFiles/agg.dir/flags.make:
CXX_FLAGS = ... -march=nocona -O3 -fopenmp -std=gnu++20 -ffp-contract=off
```

The isolation test passed:

```text
bash -n state/profiling/*.sh
state/profiling/test_run_local_isolation.sh

resolved_binary: <RUN>/work/abiss/build-system/match_chunks
resolved_binary: <RUN>/work/abiss/build-system/meme
resolved_binary: <RUN>/work/abiss/build-system/agg_overlap
resolved_binary: <RUN>/work/abiss/build-system/agg_nonoverlap
resolved_binary: <RUN>/work/abiss/build-system/agg
resolved_binary: <RUN>/work/abiss/build-system/split_remap
resolved_binary: <RUN>/work/abiss/build-system/assort
resolved_binary: <RUN>/work/abiss/build-system/ws2
resolved_build_dir: <RUN>/work/abiss/build-system
Refusing ABISS_REPO outside <RUN>: <PROJECT>/lib/abiss
Refusing ABISS_BIN_PATH outside <RUN>/work/abiss: <PROJECT>/lib/abiss/build
Refusing profiled ABISS repo outside CCC run: <PROJECT>/lib/abiss
Isolation checks passed for <RUN>/work/abiss
```

The test asserts seven ME binary resolutions, one WS binary resolution, all 17 manifest
binary resolutions, and the profiling wrapper's `agg` resolution. It also expects nonzero
exits and the shown diagnostics for live `ABISS_REPO`, `ABISS_BIN_PATH`, manifest capture,
and profiling overrides.

Repository checks passed:

```text
git diff --check
git diff --quiet -- scripts
work_scripts_diff_exit=0

git check-ignore -v build-system build-mimalloc build-arch-sanitize
.gitignore:1:/build-system/        build-system
.gitignore:2:/build-mimalloc/      build-mimalloc
.gitignore:3:/build-arch-sanitize/ build-arch-sanitize
```

No cluster benchmark, full-volume replay, or segmentation comparison was run.

## Review Focus

- Confirm the empty `ABISS_ARCH` branch adds no FP directive and the non-empty branch adds
  C++-only `-ffp-contract=off`.
- Confirm the physical-path checks reject both direct live paths and symlinks resolving
  outside the run-local isolated repository before sourcing or execution.
- Confirm the per-target `USE_ABSL_HASHMAP` and `USE_MIMALLOC` columns match the configured
  allocator variants.
- Treat complete unrelabelled segmentation equality as the blocking release gate.

## Risks and Unknowns

Segmentation bit-identity remains **UNVERIFIED** overall. The next required step is the
blocking fidelity gate: capture the immutable baseline manifest, run complete affected
outputs at least twice per candidate, and compare shape, dtype, per-block SHA-256, and
every unrelabelled label value exhaustively.

No speed or memory improvement was measured in this stage. The mimalloc build and
non-empty architecture path are only build candidates. Cluster profiling, representative
L4/L5 sampling, runtime mapping, and node-ISA inventory remain pending. A non-empty
production `ABISS_ARCH` must not be selected before the node inventory.

The directory-scope/cache distinction can mislead an audit that reads only
`CMakeCache.txt`; effective flags must continue to be read from per-target `flags.make`.
Independent inherited `-mfma` or `-mavx*` flags are not sanitized by `ABISS_ARCH`.

## Changes Since Previous Code Version

1. Chose the preferred construction fix: `-ffp-contract=off` is conditional on non-empty
   `ABISS_ARCH`. The default adds no FP directive; the flag still applies whenever
   `ABISS_ARCH` is set, which is the case where FMA can appear.
2. Cited the resolved allocator finding from the coordinator: the live production build
   defined no `USE_MIMALLOC`/`USE_ABSL_HASHMAP`, and `agg`, `meme`, `ws2`, and `acme`
   linked zero allocator libraries. The `system` default therefore remains; no `auto`
   mode was added and no live allocator re-investigation was performed.
3. Added explicit per-target `USE_ABSL_HASHMAP` and `USE_MIMALLOC` columns. The table shows
   0/17 for both in the system build, 0/17 `USE_ABSL_HASHMAP` in mimalloc, and 17/17
   `USE_MIMALLOC` in mimalloc.
4. Added `.gitignore` entries for `build-system/`, `build-mimalloc/`, and
   `build-arch-sanitize/`; `git check-ignore -v` passes for all three.
5. Added positive physical-path resolution checks and negative live-checkout tests.
   Repository, build, manifest, and profiling overrides pointing at `lib/abiss` fail
   loudly before source or execution.
6. Cited `run.md` `## Approvals` and the user's 2026-07-29 dependency approval when
   describing the run-local mimalloc build.
7. Recorded that architecture stripping changes directory scope rather than the cache,
   demonstrated the cache/`flags.make` difference, and stated that only `-march` and
   `-mtune` are stripped, not `-mfma` or `-mavx*`.
8. Guarded `-ffp-contract=off` with `$<COMPILE_LANGUAGE:CXX>`.
