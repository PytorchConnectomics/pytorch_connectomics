# metroem

**GitHub:** https://github.com/seung-lab/metroem
**Language:** Jupyter Notebook | **Stars:** 9

Training framework for EM volume alignment models. Trains multi-scale alignment modules that progressively refine registration from coarse to fine MIP levels, using displacement fields.

## Key Features
- Multi-stage, multi-MIP alignment model training
- Built on modelhouse, artificery, and scalenet packages
- Image dataset creation from CloudVolumes
- Displacement field dataset generation between stages
- Supports progressive refinement across resolution levels

## Usage
```bash
# Download a model
modelhouse load gs://corgie/models/pyramid_m4m6m9

# Create training image dataset from CloudVolume
python download_image.py --dst_folder ~/data --z_start 8175 --z_end 8180 \
    --mip 3 5 --patch_size 1536 1536 --cv_path <path>
```

## Relevance to Connectomics
Trains the alignment models used by corgie for petascale serial section EM registration, a critical preprocessing step before segmentation.
