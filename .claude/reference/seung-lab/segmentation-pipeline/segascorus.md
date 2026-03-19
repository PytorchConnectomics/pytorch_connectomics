# segascorus

**GitHub:** https://github.com/seung-lab/segascorus
**Language:** Python | **Stars:** 6

Error metrics for volumetric segmentation. Computes Rand Index and Variation of Information between two segmentations, with options for foreground restriction and singleton splitting.

## Key Features
- Rand Index and Variation of Information metrics
- Foreground restriction (ignore background voxels)
- Splitting of "0" segment into singletons for watershed outputs
- Error curve computation over watershed MST thresholds
- Plotting utilities for error curves

## API
```python
# One-shot scoring
python score.py --seg1 pred.tif --seg2 gt.tif

# Error curves over thresholds
python curve.py --help
```

## Relevance to Connectomics
Standard evaluation metrics (Rand, VOI) for comparing predicted EM segmentations against ground truth.
