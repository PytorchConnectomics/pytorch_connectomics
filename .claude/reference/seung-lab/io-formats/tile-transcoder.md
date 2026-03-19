# tile-transcoder

**GitHub:** https://github.com/seung-lab/tile-transcoder
**Language:** Python | **Stars:** 3

Utility for bulk moving and re-encoding 2D image tiles between formats and locations, with integrated resin/tissue detection for TEM data.

## Key Features
- Bulk image tile format conversion (.bmp, .png, .jpeg, .jxl, .tiff)
- Database-backed job tracking with parallel workers
- Integrated resin/tissue detector for TEM tiles
- SLURM-compatible worker model with lease-based processing
- Supports remote destinations via cloud-files

## Usage
```bash
transcode init $SRC $DEST --encoding jxl --db xfer.db
transcode worker xfer.db --parallel 2 --progress
```

## Relevance to Connectomics
Efficiently converts and transfers large collections of EM image tiles between storage formats during data pipeline processing.
