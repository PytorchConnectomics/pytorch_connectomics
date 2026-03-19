# feabas

**GitHub:** https://github.com/seung-lab/feabas
**Language:** Python | **Stars:** 0

FEABAS (Finite-Element Assisted Brain Assembly System) -- a Python library powered by finite-element analysis for stitching and alignment of serial-sectioning electron microscopy connectomics datasets.

## Key Features
- Tile-based image stitching with matching, optimization, and rendering stages
- Section-to-section alignment using finite-element methods
- Thumbnail generation for coarse alignment
- Configurable via per-project YAML configs
- Handles arbitrary tile arrangements (not just rectilinear grids)
- Checkpoint-based pipeline with error recovery

## Usage
```bash
python scripts/stitch_main.py --mode matching
python scripts/stitch_main.py --mode optimization
python scripts/stitch_main.py --mode rendering
```

## Relevance to Connectomics
Core tool for stitching and aligning serial-section EM image volumes, an essential preprocessing step before segmentation.
