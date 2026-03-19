# BOSSArrays.jl

**GitHub:** https://github.com/seung-lab/BOSSArrays.jl
**Language:** Julia | **Stars:** 0

Julia interface to BOSS (Brain Observatory Storage Service) for large-scale image cutout and chunk saving.

## Key Features
- Read and write image volumes to BOSS as standard Julia arrays
- Blosc compression for fast data transfer
- Error handling for write operations

## API
```julia
using BOSSArrays
ba = BOSSArray(collectionName="col", experimentName="exp", channelName="ch")
ba[10001:10200, 10001:10200, 1:3] = rand(UInt8, 200, 200, 3)
b = ba[10001:10200, 10001:10200, 1:3]
```

## Relevance to Connectomics
Enables Julia-based access to BOSS, a cloud storage backend for large-scale EM brain imaging data.
