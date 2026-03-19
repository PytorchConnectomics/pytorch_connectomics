# deflate

**GitHub:** https://github.com/seung-lab/deflate
**Language:** C | **Stars:** 0

Thin Python wrapper around Eric Biggers' libdeflate library for fast gzip compression and decompression.

## Key Features
- Gzip compress/decompress with libdeflate backend
- Compression levels 1-12
- Simple two-function API

## API
```python
import deflate
compressed = deflate.gzip_compress(b"data", level=9)
original = deflate.gzip_decompress(compressed)
```

## Relevance to Connectomics
Provides fast compression for volumetric data I/O in the CloudVolume/connectomics data pipeline.
