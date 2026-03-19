# pyspng-seunglab

**GitHub:** https://github.com/seung-lab/pyspng-seunglab
**Language:** Python | **Stars:** 4

Fork of pyspng for efficiently loading and saving PNG files to/from numpy arrays. Uses libspng for 2-3x faster PNG decoding than Pillow, with added encoding support.

## Key Features
- Fast PNG decode/encode via libspng
- 2-3x faster than Pillow for uncompressed PNGs
- Encoding with progressive and interlaced modes
- CLI for npy-to-PNG conversion and header inspection
- Binary wheels for Linux, macOS, and Windows

## API
```python
import pyspng
from pyspng import ProgressiveMode

# Decode
with open('test.png', 'rb') as f:
    arr = pyspng.load(f.read())

# Encode
binary = pyspng.encode(arr, progressive=ProgressiveMode.PROGRESSIVE, compress_level=6)
```

## CLI
```bash
pyspng example.npy --level 9 --progressive  # npy -> png
pyspng -d example.png                        # png -> npy
pyspng --header example.png                  # read header
```

## Relevance to Connectomics
Provides fast PNG I/O for EM image tiles stored in Precomputed/CloudVolume format.
