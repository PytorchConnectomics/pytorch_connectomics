# corgie_deprecated

**GitHub:** https://github.com/seung-lab/corgie_deprecated
**Language:** Python | **Stars:** 0

COnnectomics Registration Generalizable Inference Engine (corgie). A CLI tool for registration of very large 3D volumes, designed for aligning EM serial sections at scale.

## Key Features
- Large-scale 3D volume registration
- Cloud storage integration (GCS, S3)
- Chunk-based processing with configurable MIP levels
- Fold detection and mask support

## API
```bash
corgie copy --src_layer_spec '{"path": "..."}' \
  --dst_folder "gs://bucket/output" \
  --start_coord "150000, 150000, 17000" \
  --end_coord "250000, 250000, 17020" \
  --mip 6 --chunk_xy 1024 --chunk_z 1
```

## Relevance to Connectomics
Handles section-to-section registration for aligning EM volumes during reconstruction (deprecated in favor of newer tools).
