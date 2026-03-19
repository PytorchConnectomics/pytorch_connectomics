# BigArrays.jl

**GitHub:** https://github.com/seung-lab/BigArrays.jl
**Language:** Julia | **Stars:** 13

Serverless Julia library for storing and accessing arbitrarily large arrays using local filesystem, Google Cloud Storage, or AWS S3 backends. Provides standard Julia Array indexing over chunked, compressed data stored in neuroglancer-compatible precomputed format.

## Key Features
- Julia Array interface for arbitrarily large datasets (tested up to ~9 TB)
- Backends: local filesystem, Google Cloud Storage, AWS S3
- Neuroglancer-compatible precomputed format for direct visualization
- Multiple compression: gzip, zstd, blosclz, jpeg
- Multi-process parallel I/O
- Supports Bool, UInt8/16/32/64, Float32/64

## API
```julia
using BigArrays, BigArrays.BinDicts
ba = BigArray(BinDict("/path/of/dataset"))
chunk = ba[1:128, 1:128, 1:128]  # standard array indexing
```

## Relevance to Connectomics
Provides array-like access to large EM volumes stored in cloud or local storage, enabling chunk-based reading/writing for distributed processing pipelines.
