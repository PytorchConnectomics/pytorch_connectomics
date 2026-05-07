# tinybrain

**GitHub:** https://github.com/seung-lab/tinybrain
**Language:** Python | **Stars:** 11

Image pyramid generation specialized for connectomics data types. Fast 2x2 and 2x2x2 downsampling with averaging (grayscale) and mode pooling (segmentation labels).

## Key Features
- Optimized 2x2 and 2x2x2 downsampling on fast C++ path
- Average pooling for grayscale images (uint8, uint16, float32, float64)
- Mode pooling for segmentation labels (preserves label identity)
- Min/max/striding pooling options
- Sparse mode to prevent ghosting at z-boundaries
- Multi-mip generation in single pass (reduces integer truncation)
- ~27x faster than naive numpy on benchmarks

## API
```python
import tinybrain

img_pyramid = tinybrain.downsample_with_averaging(img, factor=(2,2,1), num_mips=5)
label_pyramid = tinybrain.downsample_segmentation(labels, factor=(2,2,1), num_mips=5)
```

## Relevance to Connectomics
Generates multi-resolution image pyramids for visualization and analysis of large EM volumes and their segmentations in tools like Neuroglancer.
