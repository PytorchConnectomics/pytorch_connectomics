# ABISS Reference

**ABISS** (Affinity Based Image Segmentation System) is a chunked 3D segmentation pipeline.

**Location**: `/Users/weidf/Code/lib/abiss`

## Pipeline Stages

1. **Watershed** (`ws`): Over affinity maps
2. **Agglomeration** (`agg`/`me`): Merge watershed fragments
3. **Contact Surface** (`cs`): Optional surface extraction

## Build

```bash
cd /path/to/abiss && mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

## Runtime

```bash
export WORKER_HOME=/path/to/abiss
export SECRETS=/path/to/secrets_dir  # Contains param JSON
export OVERLAP=0

# 1) Watershed
export STAGE=ws && scripts/run_batch.sh ws 3 0_0_0_0
scripts/remap_batch.sh ws 3 0_0_0_0

# 2) Agglomeration
export STAGE=agg && scripts/run_batch.sh me 3 0_0_0_0
scripts/remap_batch.sh agg 3 0_0_0_0
```

## Key Parameters (param JSON)

- `CHUNK_SIZE`, `BBOX`: Volume chunking
- `AFF_PATH`: Affinity map location
- `WS_HIGH_THRESHOLD`, `WS_LOW_THRESHOLD`: Watershed thresholds
- `AGG_THRESHOLD`: Agglomeration threshold
- `SCRATCH_PATH`: Working directory

## Affinity Channels

- 1 channel: probability map -> converted to 3 affinities
- 3 channels: used directly as affinities
- 4 channels: 3 affinity + 1 myelin

## PyTC Integration

Used via `connectomics/decoding/decoders/abiss.py` -- wraps ABISS as an external decoder in the decoding pipeline.
