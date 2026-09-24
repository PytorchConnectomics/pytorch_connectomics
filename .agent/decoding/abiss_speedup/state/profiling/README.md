# ABISS profiling harness

This directory contains non-production copies of the ME and WS composite
scripts. The copies leave both the live and isolated `scripts/` directories
unchanged and route every ABISS binary invocation through the executable named
by `ABISS_PROFILE_PREFIX`. By default, every helper resolves ABISS to the
isolated checkout at `<RUN>/work/abiss`. Physical repository, source-script,
build-directory, and binary paths must remain below `<RUN>`; symlinks or
overrides that resolve to the live checkout fail before `init.sh` is sourced.
Set `ABISS_REPO` explicitly only when the isolated checkout has been copied
elsewhere within the same run folder.

Before profiling, capture an immutable build/environment baseline and copies of
the existing segmentation outputs:

```bash
./capture_baseline_manifest.sh \
  ../baseline_manifest_before_allocator \
  /path/to/preserved/segmentation-output
```

`capture_baseline_manifest.sh` records compiler and CMake versions, every
target's `flags.make`, `CMakeCache.txt`, `ldd` for every built executable,
`conda list -n pytc`, binary SHA-256 values, git state, and read-only copies of
the supplied baseline outputs. It never regenerates those outputs. It reads
`<RUN>/work/abiss/build-system` by default; set `ABISS_BUILD_DIR` to capture a
different isolated configuration.

To profile a composite run:

```bash
export ABISS_PROFILE_PREFIX="$PWD/profile_command.sh"
export ABISS_PROFILE_DIR=/path/to/new/profile-records
export ABISS_PROFILE_MANIFEST=/path/to/baseline_manifest_before_allocator
export ABISS_BIN_PATH=/path/to/isolated/abiss/build-system
export SECRETS=/path/to/existing/.cloudvolume/secrets
./composite_chunk_me.sh /path/to/chunk.json
./composite_chunk_ws.sh /path/to/chunk.json
```

The prefix must name one executable, not a shell fragment. Each binary gets a
separate record directory containing `/usr/bin/time -v` output, the command and
exit status, the executing PID, `Cpus_allowed_list` read from
`/proc/<pid>/status`, and `/proc/<pid>/maps` when the process remains live long
enough to capture it. Each record names the required immutable baseline
manifest. The ME copy wraps all seven possible ABISS executable names
(`match_chunks`, `meme`, three `agg` variants, `split_remap`, and `assort`);
the WS copy wraps `ws2`.

Run the isolation test before submitting profiling jobs:

```bash
./test_run_local_isolation.sh
```

It positively checks the physical paths of all eight composite binaries and
negatively checks that repository, build, manifest-capture, and profiling
overrides pointing at the live `lib/abiss` checkout fail loudly.

The cluster sweep is intentionally not automated here. Select at least five L4
chunks spanning the observed cost spread and the single L5 chunk. Treat L5 as
latency-only. Before each L4 concurrency point, require:

```text
concurrency * measured per-chunk RSS <= 0.8 * node memory
```

For the dominant TBB-linked binary, compare 1, 4, and 16 CPUs per task. More
than 100% in the `Percent of CPU this job got` field is the required evidence
that more than one CPU was actually used. Run the highest admitted setting
twice and compare the complete unrelabelled segmentation arrays bit-for-bit.
