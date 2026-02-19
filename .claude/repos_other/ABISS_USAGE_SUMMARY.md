# ABISS Usage Summary

This note summarizes how to build and run ABISS from `/Users/weidf/Code/lib/abiss`, based on the current scripts and source in that repo.

## What ABISS Is

ABISS (Affinity Based Image Segmentation System) is a chunked 3D segmentation pipeline:

1. Watershed over affinity maps (`ws` stage).
2. Agglomeration/merge of watershed fragments (`agg` stage, typically `me` op).
3. Optional contact-surface extraction (`cs` op).

Pipeline orchestration is driven by shell scripts in `/Users/weidf/Code/lib/abiss/scripts`.

## Build

From the ABISS repo:

```bash
cd /Users/weidf/Code/lib/abiss
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
```

Key binaries produced in `build/`:

- `ws`, `ws2`, `ws3`
- `acme`, `meme`, `agg`, `agg_nonoverlap`, `agg_overlap`, `agg_extra`
- `split_remap`, `match_chunks`, `reduce_chunk`, `size_map`, `evaluate`
- `accs`, `mecs`, `assort`

## Runtime Layout and Entry Scripts

Primary entrypoints:

- `scripts/run_batch.sh <op> <num_composite_layers> <root_tag>`
- `scripts/remap_batch.sh <op> <num_composite_layers_unused> <root_tag>`

Where:

- `<op>` maps to script names:
  - `ws` -> `atomic_chunk_ws.sh`, `composite_chunk_ws.sh`, `remap_chunk_ws.sh`
  - `me` -> `atomic_chunk_me.sh`, `composite_chunk_me.sh`, `remap_chunk_agg.sh`
  - `cs` -> `atomic_chunk_cs.sh`, `composite_chunk_cs.sh` (no remap script in batch wrapper)
- `<root_tag>` is chunk tag format: `mip_x_y_z` (example: `0_0_0_0`).

## Required Environment Conventions

`scripts/init.sh` expects:

- `WORKER_HOME` (defaults to `/workspace/seg`)
- `SECRETS` directory containing a parameter JSON file named `param`
  - `PARAM_JSON` is set to `$SECRETS/param`
- `STAGE` must be exported by caller (`ws`, `agg`, optionally `cs`)
- For `me` scripts, `OVERLAP` must be exported (`0`, `1`, or `2`)
  - `atomic_chunk_me.sh` and `composite_chunk_me.sh` use `set -u`, so unset `OVERLAP` will fail.

`init.sh` auto-generates `$SECRETS/config.sh` by running `scripts/set_env.py $PARAM_JSON` (once), then sources it.

## Parameter JSON (`$SECRETS/param`)

### Core keys (practically required)

- `NAME`
- `BBOX`
- `CHUNK_SIZE`
- `AFF_PATH`
- `AFF_RESOLUTION`
- `WS_HIGH_THRESHOLD`
- `WS_LOW_THRESHOLD`
- `WS_SIZE_THRESHOLD`
- `AGG_THRESHOLD`
- `SCRATCH_PATH`
- `WS_PREFIX`, `SEG_PREFIX` (or explicit `WS_PATH`, `SEG_PATH`)

### Highly recommended / stage-specific

- `CHUNKMAP_OUTPUT` for watershed remap output upload
- `CHUNKMAP_INPUT` (optional; defaults to `${SCRATCH_PATH}/ws/chunkmap`)
- `WS_DUST_THRESHOLD` (defaults to `WS_SIZE_THRESHOLD`)
- `REMAP_SIZE_MAP_THRESHOLD` (defaults to `100000`)
- `SEM_PATH`, `SEMANTIC_WS`, `SEM_FILL_MISSING`
- `AFF_FILL_MISSING`, `WS_FILL_MISSING`, `SEG_FILL_MISSING`
- `GT_PATH`, `CLEFT_PATH` (used by eval path in remap agg)
- `CHUNKED_AGG_OUTPUT`, `CHUNKED_SEG_PATH`
- `UPLOAD_CMD`, `DOWNLOAD_CMD` (auto-derived if missing)
- `REDIS_SERVER`, `REDIS_DB` (task state tracking; otherwise fallback to scratch `done/` files)

### Affinity channel expectations

In `cut_chunk_common.py`:

- 1 channel: interpreted as probability map and converted to 3 affinities.
- 3 channels: used as affinities.
- 4 channels: first 3 affinity + 1 myelin.
- N channels: first `AFF_CHANNELS` channels (default `3`).

## Minimal End-to-End Run Sequence

Assuming:

- ABISS repo at `/Users/weidf/Code/lib/abiss`
- your param JSON available at `$SECRETS/param`
- root chunk tag `0_0_0_0`
- composite layers `3` (example)

```bash
export WORKER_HOME=/Users/weidf/Code/lib/abiss
export SECRETS=/path/to/secrets_dir
export OVERLAP=0

# 1) Watershed atomic+composite
export STAGE=ws
/Users/weidf/Code/lib/abiss/scripts/run_batch.sh ws 3 0_0_0_0

# 2) Watershed remap (writes to WS_PATH and CHUNKMAP_OUTPUT)
export STAGE=ws
/Users/weidf/Code/lib/abiss/scripts/remap_batch.sh ws 3 0_0_0_0

# 3) Agglomeration atomic+composite (ME path)
export STAGE=agg
/Users/weidf/Code/lib/abiss/scripts/run_batch.sh me 3 0_0_0_0

# 4) Agg remap (writes to SEG_PATH and size map)
export STAGE=agg
/Users/weidf/Code/lib/abiss/scripts/remap_batch.sh agg 3 0_0_0_0
```

## Outputs (High Level)

- `ws` remap uploads chunked watershed segmentation to `WS_PATH`.
- `ws` remap uploads chunkmaps to `CHUNKMAP_OUTPUT`.
- `agg` remap uploads final segmentation to `SEG_PATH`.
- `agg` remap uploads size map to `${SEG_PATH}/size_map`.

## Common Failure Points

- `STAGE` not set: scripts depend on it for pathing and remap logic.
- `OVERLAP` not set for `me`: shell exits due to `set -u`.
- `CHUNKMAP_OUTPUT` missing: watershed remap upload path becomes invalid.
- Wrong `AFF_RESOLUTION` / bbox alignment: cutout and upload mismatches.
- Missing cloud credentials or invalid `UPLOAD_CMD` / `DOWNLOAD_CMD`.

## Optional Paths

- Contact surface pipeline:
  - `export STAGE=cs`
  - `scripts/run_batch.sh cs <layers> <root_tag>`
- Legacy `rlme` scripts exist, but they reference binaries (`ac`, `me`) not defined in current `CMakeLists.txt`; treat as legacy unless you add/build those tools.
