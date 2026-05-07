# H5SectionsArrays.jl

**GitHub:** https://github.com/seung-lab/H5SectionsArrays.jl
**Language:** Julia | **Stars:** 0

Cutout arbitrary chunks from serial 2D image sections stored in HDF5 format, accessed as standard Julia arrays.

## Key Features
- Read-only array interface over HDF5 section stacks
- Supports negative coordinates
- Registry file defines section offsets and sizes
- Used as a normal Julia array with indexing

## API
```julia
using H5SectionsArrays
ba = H5SectionsArray("path/of/dataset/")
a = ba[101:300, -99:100, 1:3]
```

## Relevance to Connectomics
Provides Julia-based random access to serial-section EM datasets stored as HDF5, a common storage format for aligned EM stacks.
