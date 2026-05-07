# CloudVolume.jl

**GitHub:** https://github.com/seung-lab/CloudVolume.jl
**Language:** Julia | **Stars:** 2

Julia wrapper for Python's cloud-volume package. Read and write Neuroglancer Precomputed volumes using Julia array syntax via PyCall.

## Key Features
- Julia array indexing interface over CloudVolume
- CloudVolumeWrapper for image read/write
- StorageWrapper for key-value file access
- Chunk-aligned upload support

## API
```julia
using CloudVolume
vol = CloudVolumeWrapper("<path to precomputed>")
img = vol[1000:1100, 2000:2100, 100:200]   # download
vol[1000:1100, 2000:2100, 100:200] = img    # upload
```

## Relevance to Connectomics
Enables Julia-based workflows to access Neuroglancer Precomputed volumes, bridging Julia analysis code with the seung-lab cloud storage ecosystem.
