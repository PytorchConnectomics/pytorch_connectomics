# corgie

**GitHub:** https://github.com/seung-lab/corgie
**Language:** Python | **Stars:** 16

COnnectomics Registration Generalizable Inference Engine. A toolkit for registration (alignment) of very large 3D serial section EM volumes at petascale.

## Key Features
- Petascale serial section EM alignment
- CloudVolume-based data I/O (local, GCS, S3)
- Multi-MIP progressive alignment
- Distributed execution via Amazon SQS
- Supports copy, normalize, align, and render operations
- Paired with MetroEM for model training

## Usage
```bash
pip install -e .

corgie copy \
    --src_layer_spec '{"name": "unaligned", "path": "..."}' \
    --dst_folder "gs://corgie/demo/my_first_stack" \
    --start_coord "150000, 150000, 17000" \
    --end_coord "250000, 250000, 17020" \
    --mip 6 --chunk_xy 1024 --chunk_z 1
```

## Citation
Popovych et al., "Petascale pipeline for precise alignment of images from serial section electron microscopy," bioRxiv, 2022.

## Relevance to Connectomics
Provides the alignment/registration step that precedes segmentation in the EM connectomics pipeline, ensuring serial sections are properly registered before neuron tracing.
