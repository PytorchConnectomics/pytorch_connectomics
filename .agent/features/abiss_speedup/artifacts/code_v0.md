# Code v0
## Overview

Implemented the approved build controls and run-local profiling tooling without touching or
building the live `lib/abiss` checkout. The isolated system and mimalloc configurations both
compiled all 17 ABISS targets with `Release` and `EXTRACT_SIZE=ON`. No commit was created.

Abseil remains rejected: it was not installed or enabled, both builds explicitly disabled
package discovery, and neither build emitted `USE_ABSL_HASHMAP`.

## What Changed

- Added `ABISS_ARCH` as an empty-by-default cache variable. Empty preserves the inherited
  architecture selection; non-empty values remove inherited `-march`/`-mtune` tokens before
  adding one `-march`.
- Removed the existing duplication of global C++ flags in the Release/Debug flag variables.
- Added `-ffp-contract=off` to every C++ target.
- Added validated `ABISS_ALLOCATOR=system|mimalloc|jemalloc` selection, defaulting to
  `system`, with dependency discovery restricted to the selected non-system allocator.
- Added the Phase 0.0 manifest capture and Phase 0.1 per-binary profiling harness under the
  CCC run folder, not under either ABISS `scripts/` directory.
- Built isolated system and mimalloc variants and created a configure-only sanitizer check
  using the existing `nocona` baseline.

## Implementation Details

`ABISS_ARCH` accepts a single architecture token. The default system build leaves it empty,
so the conda toolchain's inherited `-march=nocona -mtune=haswell` remains in effect. Release
flags no longer embed a second copy of `CMAKE_CXX_FLAGS`, eliminating the prior duplicate
`-march`. A configure-only `ABISS_ARCH=nocona` check confirmed that the non-empty branch
removes inherited `-march` and `-mtune` before adding exactly one `-march=nocona`; it does
not propose a new node baseline.

`ABISS_ALLOCATOR=system` performs no allocator package lookup and adds no allocator
definition or library. The mimalloc branch requires its pkg-config module, defines
`USE_MIMALLOC`, and links the imported pkg-config target. The jemalloc branch is symmetric
but was not built because only mimalloc was already available in the isolated dependency
prefix.

The manifest helper records the compiler, CMake version, per-target `flags.make`,
`CMakeCache.txt`, `ldd` output, `conda list -n pytc`, binary SHA-256 values, git state, and
dereferenced read-only copies of supplied baseline outputs. The profiling copies wrap all
seven possible ME executable names and `ws2` via `ABISS_PROFILE_PREFIX`. Each record names
its baseline manifest and captures `/usr/bin/time -v`, the command PID,
`Cpus_allowed_list` from `/proc/<pid>/status`, and `/proc/<pid>/maps` when available.

## Files Changed

| File | Purpose |
|---|---|
| `work/abiss/CMakeLists.txt` | Add architecture, FP contraction, and allocator controls. |
| `state/profiling/capture_baseline_manifest.sh` | Capture the Phase 0.0 immutable build/environment baseline and output copies. |
| `state/profiling/profile_command.sh` | Record per-command timing, RSS, PID affinity, maps, and baseline-manifest identity. |
| `state/profiling/profile_exec.sh` | Preserve the profiled executable PID across `exec`. |
| `state/profiling/composite_chunk_me.sh` | Run-local ME composite copy with all seven possible ABISS binary calls wrapped. |
| `state/profiling/composite_chunk_ws.sh` | Run-local WS composite copy with `ws2` wrapped. |
| `state/profiling/README.md` | Document isolated defaults, manifest capture, sampling, and admission rules. |
| `work/abiss/build-system/` | Generated Release/system build with `EXTRACT_SIZE=ON`. |
| `work/abiss/build-mimalloc/` | Generated Release/mimalloc build using the isolated dependency prefix. |
| `work/abiss/build-arch-sanitize/` | Generated configure-only check of non-empty architecture sanitization. |

## Git Baseline

run_start_ref: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
current_head: 3c4f56219488441edb4c51ee7ca4f07cdef2cd8f

## Verification

Both build commands completed all 17 targets:

```text
conda run -n pytc cmake -S . -B build-system \
  -DCMAKE_BUILD_TYPE=Release -DEXTRACT_SIZE=ON \
  -DABISS_ALLOCATOR=system -DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE
conda run -n pytc cmake --build build-system --parallel 8

PKG_CONFIG_PATH=<RUN>/deps/lib64/pkgconfig conda run -n pytc cmake \
  -S . -B build-mimalloc -DCMAKE_BUILD_TYPE=Release -DEXTRACT_SIZE=ON \
  -DABISS_ALLOCATOR=mimalloc -DCMAKE_DISABLE_FIND_PACKAGE_absl=TRUE
conda run -n pytc cmake --build build-mimalloc --parallel 8
```

The grep-based audit of every system-build `flags.make` produced:

```text
accs            march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
acme            march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
agg             march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
agg_extra       march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
agg_nonoverlap  march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
agg_overlap     march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
assort          march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
evaluate        march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
match_chunks    march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
mecs            march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
meme            march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
reduce_chunk    march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
size_map        march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
split_remap     march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
ws              march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
ws2             march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
ws3             march=1 ffp-contract-off=1 EXTRACT_SIZE=1 unsafe-fp=0
```

The cache grep showed the required default configuration:

```text
ABISS_ALLOCATOR:STRING=system
ABISS_ARCH:STRING=
CMAKE_BUILD_TYPE:STRING=Release
EXTRACT_SIZE:UNINITIALIZED=ON
```

The direct unsafe-FP grep and the system-build allocator `ldd` grep were both empty:

```text
grep -HEn -- '-ffast-math|-Ofast|-funsafe-math-optimizations' \
  build-system/CMakeFiles/*.dir/flags.make
grep_exit=1

for <all 17 system binaries>; do ldd "$binary"; done |
  grep -Ei 'mimalloc|jemalloc'
grep_exit=1
```

The mimalloc build enabled `USE_MIMALLOC` on all 17 targets. Its direct `ldd` proof was:

```text
libmimalloc.so.2 => <RUN>/deps/lib64/libmimalloc.so.2 (0x00007ffff7ed9000)
```

All 17 mimalloc binaries resolved `libmimalloc.so.2`; none resolved jemalloc. This is a
linker-resolution check, not a benchmark or a segmentation-fidelity result.

The non-empty architecture configure check produced one `-march=nocona` and zero `-mtune`
tokens for each of the 17 targets. The default deployable build remained `ABISS_ARCH=""`.

The profiling tools passed `bash -n`; `git diff --check` passed; and
`git diff --quiet -- scripts` returned zero. A local `/bin/sleep 1` smoke test of the
profiler returned exit status 0 and captured:

```text
executable: /usr/bin/sleep
Cpus_allowed_list: 0-6
Percent of CPU this job got: 1%
Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.13
Maximum resident set size (kbytes): 3712
maps_capture=present
```

No cluster profiling sweep or ABISS benchmark was run.

## Review Focus

- Confirm the `CMAKE_CXX_FLAGS` sanitization handles the supported non-empty architecture
  tokens while the empty default preserves the inherited architecture choice.
- Confirm allocator selection cannot silently fall through from one discovered allocator to
  another and that the system default has no allocator dependency.
- Confirm the run-local composite copies cover every requested ABISS executable and cannot
  default to the live checkout.
- Treat the full-volume fidelity gate as blocking before either build is used for production.

## Risks and Unknowns

Segmentation bit-identity is **UNVERIFIED** in this stage. No preserved whole-volume output
was supplied to this isolated build, so neither exhaustive raw-label comparison nor repeat
determinism testing was possible. The next required step is to capture a named immutable
baseline manifest, run at least two complete affected-output reproductions per candidate,
and compare shape, dtype, per-block SHA-256, and every unrelabelled label value exhaustively.

No speed or memory payoff was measured. The cluster sweep, representative L4/L5 sampling,
heap attribution, and Phase A ranking remain pending. The mimalloc runtime map should also
be captured from an actual candidate workload even though `ldd` resolves the intended
library. No non-default production `ABISS_ARCH` should be selected before the node-ISA
inventory.

## Changes Since Previous Code Version

Initial implementation.
