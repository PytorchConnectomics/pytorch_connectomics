# StitchEM

**GitHub:** https://github.com/seung-lab/StitchEM
**Language:** Matlab | **Stars:** 1

A set of tools for serial electron microscopy image stitching and alignment. Generates rough affine alignment via feature matching in MATLAB, then refines with piecewise affine (elastic) alignment using TrakEM2 in FIJI.

## Key Features
- Tile-to-overview alignment (rough_xy)
- Within-section tile stitching (xy)
- Cross-section (z) alignment of overviews and tiles
- Elastic alignment via TrakEM2/FIJI block matching
- Full pipeline from raw tile images to registered stack

## Relevance to Connectomics
Essential preprocessing step for assembling EM tile images into aligned volumes before segmentation.
