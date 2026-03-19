# DVIDArrays.jl

**GitHub:** https://github.com/seung-lab/DVIDArrays.jl
**Language:** Jupyter Notebook (Julia) | **Stars:** 1

Julia interface to DVID (Distributed, Versioned, Image-oriented Dataservice) for accessing EM image volumes.

## Key Features
- Access DVID image data as standard Julia arrays
- Supports ImageTileArrays for tile-based access
- Arbitrary subvolume cutouts via array indexing

## API
```julia
using DVIDArrays
a = ImageTileArray("server_address", 8000, "5c")
arr = a[2305:2604, 1473:1772, 1281:1282]
```

## Relevance to Connectomics
Provides Julia-based access to DVID, a key data service for large-scale EM connectomics datasets (e.g., FlyEM at Janelia).
