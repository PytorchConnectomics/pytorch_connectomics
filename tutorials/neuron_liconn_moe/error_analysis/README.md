# No-GT error analysis for moe segmentations

Skeleton-based semantic type (axon/dendrite caliber) and completeness (free ends
against the crop faces) for every label >= 1000 voxels, plus split-link
proposals and a Neuroglancer `segment_properties/` sidecar. There is no ground
truth for any moe volume; nothing here is a score.

These are the scripts behind the published ExPID96 S1 eb2 `error_analysis.json`
(run 2026-09-17, `analysis_script_sha256` a5d72f66...), promoted from the
untracked `dev/liconn_moe/`. `volume.py` now selects the volume:

- unset `LICONN_EA_RUN`: the S1 constants, exactly as published;
- `LICONN_EA_RUN=<run dir>`: name, segmentation, spacing (`mt_sweep.json`),
  shape (the h5) and affinity (`mt_sweep.json`, or `LICONN_EA_AFFINITY`)
  come from that directory. Never reuse S1's spacing: the 22x/28x/32x volumes
  are near [24, 18, 18] nm, the 18x ones [22.22, 18.06, 18.06].

Settings are the published run's: 10 nm skeleton simplification, caliber gates
0.15/0.20 um, terminal windows 0.10/0.15 um, 0.4 um pieces. The gates were
calibrated on S1; each volume's `caliber_survey.txt` is kept to check them.
`--max-voxels 5e6` reproduces S1's one skipped giant; skipped labels are listed
in `metadata.skeletonization.labels_skipped_oversized` and come out `unmeasured`.

Outputs land in `<run dir>/error_analysis/`. `finish` refuses to overwrite an
existing `error_analysis.json`.

## Five classes

`semantic/semantic_segmentation.json` assigns every label one of `axon`
(terminals included), `dendrite` (neurons with soma included), `glia`,
`blood_vessel`, `unclassified`. Glia and vessels have no detector yet: zero means
not assessed. Size gates, recorded in the catalog metadata: an axon call needs a
skeleton >= 3 um or volume >= 0.1 um3 (thin processes can be long with little
volume; short-and-small pieces, often false splits, are penalized); a thick object with a skeleton < 2 um is
a blob (a detached axon terminal as easily as a dendrite piece). Both go to
`unclassified`. The classes reach the viewer as `class:` tags in
`segment_properties/info`.

`run_volume.sh reports` rebuilds everything downstream of `error_analysis.json`
(catalog, sidecar, `reports/<cube>/report.{json,md}`) and overwrites in place.
`publish.py` then uploads it all beside the seg layer, skipping unchanged objects
and adding `"segment_properties"` to the layer `info` when missing.

## BC (SLURM)

```bash
bash tutorials/neuron_liconn_moe/error_analysis/slurm/submit.sh <run dir>...
```

One sizes job, a 16-shard skeleton array, then `finish` (merge, caliber
survey, analyze, link, plot, segment properties), chained with `afterok`.

## gcloud (container)

CPU-only; no credentials inside. Stage the run directory and its affinity
onto the VM, run, copy `error_analysis/` back out.

```bash
docker build -f tutorials/neuron_liconn_moe/error_analysis/Dockerfile -t liconn-ea .
docker run --rm -v /data/<name>:/data/<name> -v /data/aff:/aff \
  -e LICONN_EA_RUN=/data/<name> -e LICONN_EA_AFFINITY=/aff/raw_x1_ch0-1-2.h5 \
  liconn-ea all 16
```

Memory: about 64 GB per Gvoxel for skeletonization and 100 GB per Gvoxel for
`finish` (S1, 0.78 Gvoxel: 32 GB and < 80 GB peaks).
