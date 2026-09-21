# LICONN ingest — raw drop → uint8 OME-Zarr → GCS

The stage that sat in front of [`neuron_liconn_moe/`](../neuron_liconn_moe/README.md)
and was never in the repo. That tutorial starts from
`preprocessed/<clip_variant>/zarr/`; **this produces those.** Until now the ND2 →
uint8 conversion lived only as a script on BC, which is why its provenance is
documented in a README on `/projects` rather than here.

```
fetch → preprocess → publish → verify → prune
```

Each stage is idempotent, writes `provenance.json`, and drops a `COMPLETE.<stage>`
marker. `prune` refuses to delete anything the remote has not confirmed.

## Why this is its own stage

`prepare_volume.py` already explains it: **CLAHE is applied per XY plane at native
resolution, before any block averaging.** Level 0 of the published zarr is exactly
that. Keeping the resample onto the checkpoint grid as a separate zarr → zarr step
means changing the training grid never means re-reading ND2 — which matters now that
the grid is itself contested.

Published output is **level 0 only**, at the finest (native) resolution. No pyramid:
a Neuroglancer slice view of a 2200³ cube is ~4.8 MB and fine; zoomed-out 3D will be
slow. Two levels is a cheap retrofit.

## Usage

```bash
# 0. one-time, interactive: rclone config   (Drive OAuth — not automatable)
rclone config                                   # create a remote named `gdrive`

# 1. Drive folder -> local inbox
python tutorials/liconn_ingest/ingest.py --inbox ./inbox \
    fetch --folder "<drive folder name>" --dry-run

# 2. one ND2 -> uint8 OME-Zarr at native resolution
python tutorials/liconn_ingest/ingest.py --work ./work \
    preprocess ./inbox/ExPID71_120ms-30ms_600nm_40XW02.nd2 \
    --clip-variant clip_percentile_1_99 \
    --optics-spacing-nm 600 162.5 162.5 \
    --fold 32 --exposure-ms 30

# 3-5. publish, verify, then delete the raw
python tutorials/liconn_ingest/ingest.py --work ./work publish <cube_id>
python tutorials/liconn_ingest/ingest.py --work ./work verify  <cube_id>
python tutorials/liconn_ingest/ingest.py --work ./work --inbox ./inbox \
    prune <cube_id> --dry-run
```

Target: `gs://donglai_public/liconn/moe/<clip_variant>/zarr/<cube_id>.zarr`, a
sibling of the twelve image groups already there.

## Three things this tool will not do for you

**`--clip-variant` has no default.** Two variants with identical dataset names
already exist, and the component is load-bearing — see `volumes.py::GCS_PREFIX`.

**Exposure is not read from the filename.** `ExPID71_120ms-30ms_...` lists 120 first;
Moe states the plane was illuminated 30 ms **first**. `naming.parse_stem` returns the
exposures as an unordered `frozenset` precisely so no caller can accidentally imply an
order. Pass `--exposure-ms` from the ND2 metadata.

**`prune` will not run on an unverified upload.** It re-hashes the raw against
`provenance.json:source_sha256` before unlinking. Once the ND2 is gone that hash is
the only surviving evidence of what was processed.

## Unresolved, and it blocks correctness not just convenience

**The clip range is exposure-dependent.** A fixed `120–350` window cannot be right for
both a 30 ms and a 120 ms acquisition of the same plane. Apply one window to both and
the sweep measures normalization rather than the microscope; re-derive per exposure and
two things change at once. The existing published layers used `clip_percentile_1_99`,
while `lessons/liconn_preprocessing.md` calls fixed `120–350` the reference recipe.
Pick deliberately and record it — `provenance.json` carries both the variant name and
the actual numbers.

## Docker

Layers on the repo's own base, same as `dispim_gcloud` reuses `pytc:snemi-abiss`.
CPU-only; nothing here assumes CUDA, so `PYTC_IMAGE` can point at a slimmer base.

```bash
docker build -f tutorials/liconn_ingest/Dockerfile -t pytc:liconn-ingest \
    --build-arg PYTC_IMAGE=pytc:gpu .
```

The build runs `run_tests.py` and refuses to produce an image if the filename grammar
or spacing arithmetic regresses — those fail silently rather than loudly, so they gate
the image.

## What is verified

- `naming.py` — 9/9 unit tests pass (`python tutorials/liconn_ingest/run_tests.py`),
  including that anisotropy is invariant to expansion fold, that `0.1625 µm / 32`
  reproduces the recorded `5.078125` nm, and that `_28xx` (a real typo in the ExPID96
  set) and fractional `_14p5x` both parse.
- `ingest.py` — CLI parses; **the ND2, zarr and GCS paths have not been executed.**
  Neither Docker nor rclone is installed on the machine this was written on, and no
  drop has been fetched.
